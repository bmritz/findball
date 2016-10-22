# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 10:53:56 2015

@author: brianritz

changes
1: original code
2: took out extraneous code from the objects
3: incorporated the find_speed function into the objects as much as possible
4: delete the stuff after the objects, reorg the call of the objects
5: put in angle detection, simplified code by passing objects to lower objects
6: put constraints on the optimization
"""


import pandas as pd
import numpy as np
import cv2
import os
from sklearn import linear_model
import statsmodels.api as sm
from math import cos, radians, sin, degrees, pi, acos, sqrt, atan

import scipy.optimize as opt
import copy

# CONSTANTS

#VIDEO_PATH = "./pitch_41.m4v"
#VIDEO_PATH = '//Users/brianritz/projects/findball/captured_videos/outing_20151004/videos/pitch_41.m4v'

#FRAMES_PER_SECOND = 240
#DISTANCE_TO_PITCH = 30.
#ZOOM = 0
#RECT = ((0, 170), (600, 430))

# these were calculated from taking video
FIELD_OF_VIEW_X = 52.
FIELD_OF_VIEW_Y = 30.88211

# calculated from measurements
FIELD_OF_VIEW_X_ZOOM = 18.47188362
FIELD_OF_VIEW_Y_ZOOM = 10.52105669

# this is the degrees of the angle it will look for another contour to continute the trajectory
DEGREES = 40

# helper function
def distance_spherical_coords(point1, point2):
    """ 
    distance in our spherical coordinates between two points
    used in the straight_line_velo method of ContourBlob
    """
    r1 = point1[0]
    th1 = radians(point1[1])
    ph1 = radians(point1[2])
    r2 = point2[0]
    th2 = radians(point2[1])
    ph2 = radians(point2[2])
    return sqrt(r1**2 + r2**2 - ((2*r1*r2)*(sin(th1)*sin(th2)*cos(ph1-ph2) + cos(th1)*cos(th2))))


def smooth_vector_ols(v, order=1, x=None):
    """
    this smooths the vector v based on ols-- may want to use thisto smooth velocity
    this is used in the smooth_data method of Trajectory object
    """
    y = pd.Series(v)
    x = pd.Series(range(len(v))) if x is None else pd.Series(x)
    if order == 1:
        model = pd.ols(y=v, x=x, intercept=True)
        y_hat = model.y_fitted
    elif order==2:
        model = pd.ols(y=y, x=pd.DataFrame({'x':x, 'x^2':[x2**2 for x2 in x]}))
        y_hat = model.y_fitted
    else:
        raise "Only order 1 or 2 is supported."
    return (model.beta, y_hat.tolist())

def simple_drag_func(pars, t):
    """
    crude estimate of the drag equation we will optimize to fit our velo data
    pars is a tuple: (v0, c_over_m)
    v0 - initial velocity
    c_over_m -- drag coefficient over the mass of the baseball
    """
    v0, c_over_m = pars
    return 1/((1/v0) + c_over_m*t)

def smooth_vector_drag(v, x=None):
    """
    smooths the data based on an estimate of drag func - simple_drag_func
    this is used in the smooth_data method of Trajectory object
    """
    y = pd.Series(v)
    x = pd.Series(range(len(v))) if x is None else pd.Series(x)
    def objective(pars):
        vo, c_over_m = pars
        return np.sum((y-x.apply(lambda(t): simple_drag_func(pars, t)))**2)
    optimized = opt.minimize(objective, x0=np.array([60.0, 0.003]), method='Nelder-Mead', options={'disp': True})
    parameter_estimates = optimized.x
    reconstructed_velos_drag = x.apply(lambda(t): simple_drag_func(parameter_estimates, t) )
    return (parameter_estimates, reconstructed_velos_drag)
    
    
def smooth_vector_drag_constrained(v, x=None, lbound=0.000015, ubound=0.000085):
    y = pd.Series(v)
    x = pd.Series(range(len(v))) if x is None else pd.Series(x)
    
    ## least squares way http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares
    def fun_resids(pars, t, y):
        return simple_drag_func(pars, t) - y

    x0=np.array([60.0, (ubound+lbound)/2.])
    constraints = ([35., lbound], [105., ubound])
    res_2 = opt.least_squares(fun_resids, x0, args=(x, y), bounds=constraints)    
    parameter_estimates = res_2.x
    reconstructed_velos_drag = x.apply(lambda(t): simple_drag_func(parameter_estimates, t) )
    return (parameter_estimates, reconstructed_velos_drag)
    
# class objects

class Video(object):
    
    def __init__(self, filename, rect=None, zoom=100, frame_rate=240,
                 distance=50, s=0):      

        """
        Make a video object full of frames from a .m4v
        filename:
        filename is the path to the m4v file
        rect:
        rect is top left corner, top right corner--default is entire image
        ((horz cord of top left, vert cord of top left), (horz cord of bottom right, vert cord of bottom right))
        
        s (offset) is the distance from the pitcher to the perpendicular line 
        from camera to pitcher-catcher line:

        distance (perpendicular distance) is the perpendicular distance from the camera to the 
        pitcher-catcher line
        
        |----- s ---|
        P------------------------------Catcher
                    |        
                    |d
                    |
                   cam
        """
        self.num_frames=0
        self.distance=distance
        self.frame_rate = frame_rate
        self.s=s
        self.frames = []
            
        self.trajectories = []
        self.max_trajectory_id=0
        self.zoom=zoom
        if zoom == 100:
            self.field_of_view_x = FIELD_OF_VIEW_X_ZOOM
            self.field_of_view_y = FIELD_OF_VIEW_Y_ZOOM
        elif zoom == 0:
            self.field_of_view_x = FIELD_OF_VIEW_X
            self.field_of_view_y = FIELD_OF_VIEW_Y
        else:
            raise ValueError("Zoom must be 0 or 100. Other values not yet supported.")
        # bring in the frames --  set rectangle default with first frame
        vc = cv2.VideoCapture(filename)
        if vc.isOpened():
            rval , frame = vc.read()
            self.rect = ((0,0), (frame.shape[0:2])) if rect is None else rect
        else:
            rval = False
        
        while rval:
            self.num_frames += 1
            self.frames.append(Frame(frame, id = self.num_frames-1, video=self))
            rval, frame = vc.read()
            cv2.waitKey(1)
        
            
    def apply_difference(self, history = 5, nmixtures=3, backgroundRatio=0.85,
                         noiseSigma=5, learningRate=.1):
        """ 
        we look at the difference from the last frames using the background subtractor from opencv
        """
        fgbg = cv2.BackgroundSubtractorMOG(history = history,
                                           nmixtures = nmixtures,
                                           backgroundRatio = backgroundRatio,
                                           noiseSigma = noiseSigma)
        for frame in self.frames:
            fgmask = fgbg.apply(frame.orig_image, learningRate=learningRate)
            frame.fgmask = fgmask
           
    def find_contours(self, image="fgmask"):
        """
        find the contours of an image
        """
        for frame in self.frames:
            frame_gr = frame.fgmask
            ret, thresh = cv2.threshold(frame_gr, 127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contours = [ContourBlob(x, frame=frame) for x in contours]
            for contour in contours:
                # self.in_rect adds attribute to contour to indicate if in rect 
                self.in_rect(contour)
            frame.contours.extend(contours)
            
    def in_rect(self, contour):
        """
        sets the in_rect property of the contour to True or False depending if it is in the rect of the video
        """
        contour.in_rect = self.rect[0][0] <= contour.position[0] <= self.rect[1][0] and self.rect[0][1] <= contour.position[1] <= self.rect[1][1]
        return contour.in_rect
    
    def blur_frames(self, ksize, image="original", method="median", sigmaX=1, sigmaY=1):
        """ 
        this calls the blur method for every frame in the video
        ksize: the number of neighbors to consider when blurring
        image: which image do you want to blur? "original", "fgmask", or "gray"
        method: which blurring method? "median", "gaussian"
        sigmaX, sigmaY: parameters for gaussian blur -- ignored for other blurs
        """
        for frame in self.frames:
            frame.blur(ksize, image=image, method=method, sigmaX=sigmaX, sigmaY=sigmaY)
    
    def findAllTrajectories(self, check_movement=True):
        """ 
        this is the function to 'solve' the trajectories problem in a video once all the frames are in a video
        
        The check_movement parameter is actually for assign new trajectories 
        method and if true will not assign new trajectories if there are > 10
        new trajectories because we assume that the camera moved when that happens
        """
        # the holding pen holds trajectory ids that did not find a contour but may find a contour in subsequent frames
        # this is to get around obstructions for a portion of the balls flight
        self.holding_pen = []
        for i in range(len(self.frames)-1):
            F1 = self.frames[i]
            F2 = self.frames[i+1]
            self.extendTrajectories(F1, F2, check_movement=check_movement)
            
            # for each frame read in, tickdown the trajectories in the holding pen
            for traj_id in self.holding_pen:
                self.trajectories[traj_id].tickdown()
            
            # remove the trajectories that have been killed from the holding pen
            self.holding_pen = [traj for traj in self.holding_pen if self.trajectories[traj].killed is False]
        
    def extendTrajectories(self, frame1, frame2, check_movement=True):
        """
        extend the trajectories from frame1 into frame 2 by looking at closest contour
        if there are too many contours in frame2, create a new trajectory
        """
        # copy the holding ben at the beginning ofthe loop through contours, 
        # because we don't want to add a trajectory to the holding pen and then search on that trajectory
        frame_copy_holding_pen = copy.copy(self.holding_pen)
        for contour in frame1.contours:
            # this list holds the contours in the previous frame before frame1 that were already part of a trajectory extended
            # i do this because i do not want trajectories to repeat -- so if two trajectories have the same last two contours, we do not extend
            prev_contours = []
            trajectories_to_extend = [traj for traj in self.trajectories if traj.id in contour.trajectories]
            # sort to extend the longest trajectories first
            trajectories_to_extend = sorted(trajectories_to_extend, key=lambda(x): len(x.contours), reverse=True)
            for traj in trajectories_to_extend:
                # check if the previous contour is in prev_contours, so we don't overlap trajectories
                # the -1 index is the current contour of the for loop, -2 index goes to the one before
                go_ahead = False
                try:
                    if traj.contours[-2] not in prev_contours:
                        prev_contours.append(traj.contours[-2])
                        go_ahead = True 
                except IndexError:
                    go_ahead = True
                if go_ahead:
                    angles = frame2.findLinearities(traj)
                    # potentials and distances will be in same order
                    distances = frame2.findDistances(contour)
                    potentials = frame2.contours
                    adp = zip(angles, distances, potentials)
                    # max angle requirement and max distance requirement
                    valid_distances = [(z[1], z[2]) for z in adp if z.count(None) == 0 and z[0] >= (180 - DEGREES) and z[2].haspos]

                    try:
                        best_contour = valid_distances[np.argmin([x[0] for x in valid_distances])][1]
                        if best_contour:
                            traj.add_contour(best_contour)
                            try:
                                # since we extended that trajectory, remove it from the holding pen if it is there
                                self.holding_pen.remove(traj.id)
                                frame_copy_holding_pen.remove(traj.id)
                            except ValueError:
                                pass
                    except ValueError:
                        # if we cannot find a best contour because there are no valid distances, add the trajectory to the holding pen
                        if traj.id not in self.holding_pen:
                            self.holding_pen.append(traj.id)

        holding_trajectories_to_extend = [traj for traj in self.trajectories if traj.id in frame_copy_holding_pen]
        # try to continue the trajectories in the holding pen
        for traj in holding_trajectories_to_extend:
                # reverse them so if we have 2 contours in one list and one contour in other list, they will be still aligned correctly
                avg_dist = np.mean([c1.distance(c2) / (c2.frame_id-c1.frame_id) for c1, c2 in zip(reversed(traj.contours[-3:-1]),reversed(traj.contours[-2:]))])
                avg_dist_rng = (traj.countup*avg_dist*.5, traj.countup*avg_dist*1.5)
                angles = frame2.findLinearities(traj)
                # potentials and distances will be in same order
                distances = frame2.findDistances(traj.contours[-1])
                potentials = frame2.contours
                adp = zip(angles, distances, potentials)
                
                valid_distances = [(z[1], z[2]) for z in adp if z.count(None) == 0 and z[0] >= (180 - DEGREES) and z[1]< avg_dist_rng[1] and z[1]>avg_dist_rng[0] and z[2].haspos]

                try:
                    best_contour = valid_distances[np.argmin([x[0]-avg_dist*traj.countup for x in valid_distances])][1]
                    if best_contour:
                        traj.add_contour(best_contour)
                        traj.reset_tickdown()
                        self.holding_pen.remove(traj.id)
                        frame_copy_holding_pen.remove(traj.id)
                except ValueError:
                    pass

        assigned_new = self.assignNewTrajectories(frame2, check_movement=check_movement)
        return assigned_new


    def assignNewTrajectories(self, frame, check_movement=True):
        """
        for a frame, if a contour in the frame does not have a trajectory -- add a trajectory to the contour
        add a trajectory object to the video and assign the id to the contour
        then add the contour to the trajectory object
        we only add trajectories that have a position -- we are saying that the ball cannot possibly start as a little point
        """
        trajectoryless_contours = [contour for contour in frame.contours if (not contour.trajectories) and contour.haspos and self.in_rect(contour)]
        # check movement was added in to check for camera movement -- 
        # lots of contours would be in the frame if the camera was moved, 
        # so we do not add trajectories if there are >10 contours and check_movement is True
        if (check_movement and (len(trajectoryless_contours) < 10)) or not check_movement:
            for contour in trajectoryless_contours:
                self.addtrajectory(initial_contours=[contour])
        return len(trajectoryless_contours) > 0


    def addtrajectory(self, initial_contours=[]):
        """
        method to add a trajectory to the video -- 
        keep track of each trajectory's id with max_trajectory_id
        """
        self.trajectories.append(Trajectory(self.max_trajectory_id, contours=initial_contours))
        self.max_trajectory_id += 1
        return self.trajectories[-1]
        

    def find_ball_trajectory(self):
        """
        find which trajectory is most likely the ball
        """
        rsq = []
        distances = []
        lengths = []
        # rsq, distances, lengths are unnecessary but nice to have here for future
        # the neural nets do not run on rcc yet, so rely on  lengths
        for t in self.trajectories:
            rsq.append(t.fit_line().rsquared)
            distances.append(t.total_distance())
            lengths.append(len(t.contours))
        # break ties with distance
        ball_trajectory0 = lengths.index(max(lengths))
        ball_trajectory1 = distances.index(max(distances))
        ball_trajectory2 = rsq.index(max(rsq))
        #ball_trajectory0 = np.argwhere(lengths == np.amax(lengths))
        #ball_trajectory1 = np.argwhere(distances == np.amax(distances))
        #ball_trajectory2 = np.argwhere(rsq == np.amax(rsq))
        lst = [ball_trajectory0]
        ball_trajectory = max(set(lst), key=lst.count)
        self.final_ball=ball_trajectory
        return ball_trajectory
    
    # NOTE: can probably make this a little more efficient
    def mark_trajectory(self, trajectory, frame_filter=None, color=(255, 0, 0), size=4, image="original"):
        """ make a trajectory in the original image of the frames"""
        if frame_filter is None:
            frame_filter = range(self.num_frames)
        for frame in self.frames:
            if frame.id in frame_filter:
                for j, con in enumerate(self.trajectories[trajectory].contours):
                    if con.frame_id == frame.id:
                        if image =="original":
                            cv2.circle(frame.orig_image, con.position, size, color, 3)
                            cv2.putText(frame.orig_image, str(trajectory), tuple([x+20 for x in con.position]), cv2.FONT_HERSHEY_COMPLEX, .5, color, )
                        if image =="fgmask":
                            cv2.circle(frame.fgmask, con.position, size, color, 3)
                            cv2.putText(frame.fgmask, str(trajectory), tuple([x+20 for x in con.position]), cv2.FONT_HERSHEY_COMPLEX, .5, color, )
                 
    def write_orig_image(self, directory, file_prefix, frame_num=None):
        """ write out the images to a directory"""
        clean_dirpath = os.path.abspath(directory)
        if frame_num is None:
            for i, frame in enumerate(self.frames):
                full_path = os.path.join(clean_dirpath, file_prefix+str(i)+".png")
                cv2.imwrite(full_path, frame.orig_image)
        else:
            for i, frame in enumerate([f for f in self.frames if f.id in frame_num]):
                full_path = os.path.join(clean_dirpath, file_prefix+str(i)+".png")
                cv2.imwrite(full_path, frame.orig_image)                
        
    def write_fgmask(self, directory, file_prefix):
        """ write out the images to a directory"""
        clean_dirpath = os.path.abspath(directory)
        for i, frame in enumerate(self.frames):
            full_path = os.path.join(clean_dirpath, file_prefix+str(i)+".png")
            cv2.imwrite(full_path, frame.fgmask)
        
    def write_image_gray(self, directory, file_prefix):
        """ write out the images to a directory"""
        clean_dirpath = os.path.abspath(directory)
        for i, frame in enumerate(self.frames):
            full_path = os.path.join(clean_dirpath, file_prefix+str(i)+".png")
            cv2.imwrite(full_path, frame.orig_image_gr)
            
    def draw_rect(self):
        for frame in self.frames:
            cv2.rectangle(frame.orig_image, self.rect[0], self.rect[1], (255, 0, 0))


class Frame(object):
    """
    a frame is one picture...it is a collection of contours as well
    there are many frames in a video, and they are sequential, as defined by self.id here
    """
    def __init__(self, pic_file, id, video=None):
        self.video=video
        self.orig_image = pic_file
        self.orig_image_gr = cv2.cvtColor(self.orig_image, cv2.COLOR_BGR2GRAY)
        self.contours = []
        self.size = (self.orig_image.shape[1], self.orig_image.shape[0])
        self.id = id
        
    def blur(self, ksize, image="original", method="median", sigmaX=1, sigmaY=1):
        """ 
        blurs the image using box filter, median filter, or gaussian filter
        
        ksize: the number of neighbors to consider when blurring
        image: which image do you want to blur? "original", "fgmask", or "gray"
        method: which blurring method? "median", "gaussian", "box"
        sigmaX, sigmaY: parameters for gaussian blur -- ignored for other blurs
        """
        if image=="original" or image=="all":
            self.orig_image_archive = self.orig_image
            if method=="median":
                self.orig_image=cv2.medianBlur(self.orig_image, ksize)
            elif method=="box":
                self.orig_image=cv2.blur(self.orig_image, ksize)
            elif method=="gaussian":
                self.orig_image=cv2.GaussianBlur(self.orig_image, ksize, sigmaX=sigmaX, sigmaY=sigmaY)
        if image=="gray" or image=="all":
            self.orig_image_gr_archive = self.orig_image_gr
            if method=="median":
                self.orig_image_gr = cv2.medianBlur(self.orig_image_gr, ksize)
            elif method=="box":
                self.orig_image_gr = cv2.blur(self.orig_image_gr, ksize)
            elif method=="gaussian":
                self.orig_image_gr = cv2.GaussianBlur(self.orig_image_gr, ksize, sigmaX=sigmaX, sigmaY=sigmaY)
        if image=="fgmask" or image=="all":
            self.fgmask_archive = self.fgmask
            if method=="median":
                self.fgmask = cv2.medianBlur(self.fgmask, ksize)
            elif method=="box":
                self.fgmask = cv2.blur(self.fgmask, ksize)
            elif method=="gaussian":
                self.fgmask = cv2.GaussianBlur(self.fgmask, ksize, sigmaX=sigmaX, sigmaY=sigmaY)

    def findLinearities(self, other_trajectory):
        """find the linearity of every contour with another trajectory"""
        try:
            # C1 is the vertex of the angle
            C1 = other_trajectory.contours[-1]
            C2 = other_trajectory.contours[-2]
            return [contour.linearities(C1, C2) for contour in self.contours]
        except IndexError:
            return [180 for contour in self.contours]
    
    def findDistances(self, other_contour):
        """
        find the distance of every contour in this frame to the other contour
        
        without .haspos() this takes a long time...
        """
        return [contour.distance(other_contour) for contour in self.contours]
    
    def crop(self, center, x_len, y_len, image = "original"):
        """
        crop out a portion of the image for further analysis
        
        center is the center of the crop
        x_len: the x lenght of the cropped frame
        y_len the y length of the cropped frame
        image: "original", "fgmask", "gray"        
        """
        x = max(0, center[0] - (x_len/2))
        y = max(0, center[1] - (y_len/2))
        x2 = min(x+x_len, self.size[1])
        y2 = min(y+y_len, self.size[0])
        if image == "original":
            crop_img = self.orig_image[y:y2, x:x2] 
            return crop_img
        elif image == "gray":
            crop_img = self.orig_image_gr[y:y+y2, x:x2]
            return crop_img
        else:
            raise KeyError("image parameter must be 'original' or 'gray'")
        # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
    
    def draw_ellipse(self, contours=None, image="original", color=(0,255,0)):
        c = self.contours if contours is None else contours
        for cont in c:
            if cont.ellipse is not None:
                if image == "original":
                    cv2.ellipse(self.orig_image, cont.ellipse, color)
                if image =="gray":
                    cv2.ellipse(self.orig_image_gr, cont.ellipse, color)
                if image =="fgmask":
                    cv2.ellipse(self.fgmask, cont.ellipse, color)
                    
    def draw_min_area_rect(self, contours=None, image="original", color=(0,255,0)):
        c = self.contours if contours is None else contours
        for cont in c:
            if cont.ellipse is not None:
                if image == "original":
                    cv2.drawContours(self.orig_image, 
                                     [np.int0(cv2.cv.BoxPoints(cont.min_area_rect))],
                                      0, color, 1)
                if image =="gray":
                    cv2.drawContours(self.orig_image_gr, 
                                     [np.int0(cv2.cv.BoxPoints(cont.min_area_rect))],
                                      0, color, 1)
                if image =="fgmask":
                    cv2.drawContours(self.fgmask, 
                                     [np.int0(cv2.cv.BoxPoints(cont.min_area_rect))],
                                      0, color, 1)
    
    def draw_bounding_rect(self, contours=None, image="original", color=(0,255,0)):
        c = self.contours if contours is None else contours
        for cont in c:
            if cont.ellipse is not None:
                if image == "original":
                    cv2.rectangle(self.orig_image, (cont.bounding_rect[0], cont.bounding_rect[1]),
                                  (cont.bounding_rect[0]+cont.bounding_rect[2],
                                   cont.bounding_rect[1]+cont.bounding_rect[3]), color)
                if image =="gray":
                    cv2.rectangle(self.orig_image_gr, (cont.bounding_rect[0], cont.bounding_rect[1]),
                                  (cont.bounding_rect[0]+cont.bounding_rect[2],
                                   cont.bounding_rect[1]+cont.bounding_rect[3]), color)
                if image =="fgmask":
                    cv2.rectangle(self.fgmask, (cont.bounding_rect[0], cont.bounding_rect[1]),
                                  (cont.bounding_rect[0]+cont.bounding_rect[2],
                                   cont.bounding_rect[1]+cont.bounding_rect[3]), color)

# for each trajectory add next on the next frame
# OR make a new trajectory for everything not on the next frame 

class ContourBlob(object):
    """
    a contour is a blob of mass that is found in the picture when 
    we take the difference from the previous frame
    it is found by the cv2 module, but i feed it here to keep track of their properties
    """
    
    def __init__(self, contour, frame=None):
        self.contour_array = contour
        self.frame_id = frame.id if frame is not None else -1
        self.frame=frame
        self.cam_distance=None
        self.moments = cv2.moments(contour)
        self.area = cv2.contourArea(contour)
        self.trajectories = set()
        self.line = (0,0)
        self.min_enclosing_circle = cv2.minEnclosingCircle(self.contour_array)
        self.min_enclosing_circle_area = ((self.min_enclosing_circle[1])**2)*pi
        self.net_ball = False
        if self.moments['m00'] > 0 and self.moments['m00'] > 0:
            self.x_cord = int(self.moments['m10'] / self.moments['m00'])
            self.y_cord = int(self.moments['m01'] / self.moments['m00'])
            self.position = (self.x_cord, self.y_cord)
            self.circularity = self.min_enclosing_circle_area / self.area
            self.haspos = True
        else:
            self.x_cord = None
            self.y_cord = None
            self.position = None
            self.haspos = True #new way to find position below!
            
            leftmost = tuple(contour[contour[:,:,0].argmin()][0])[0]
            rightmost = tuple(contour[contour[:,:,0].argmax()][0])[0]
            topmost = tuple(contour[contour[:,:,1].argmin()][0])[1]
            bottommost = tuple(contour[contour[:,:,1].argmax()][0])[1]
            self.x_cord = int(round((leftmost+rightmost) / 2))
            self.y_cord = int(round((topmost+bottommost) / 2))
            self.position = (self.x_cord, self.y_cord)
        try:
            self.ellipse = cv2.fitEllipse(self.contour_array)
            self.min_area_rect = cv2.minAreaRect(self.contour_array)
            self.bounding_rect = cv2.boundingRect(self.contour_array)
        except:
            self.ellipse=None
    
    def distance(self, other_contour):
        """
        distance from the contour to another contour specified in other_contour
        """
        if self.haspos and other_contour.haspos:
            return sqrt((self.x_cord-other_contour.x_cord)**2 + (self.y_cord-other_contour.y_cord)**2)
        else:
            return None
        
    def distance_to_cam(self, P):
        """
        P origination_frame_pt is a point (x, y) coordinates in the camera frame 
        in which the pitcher first released the ball -- it is where the pitch
        originates
        """
                
        FOV_h = radians(FIELD_OF_VIEW_X) if self.frame.video.zoom==0 else radians(FIELD_OF_VIEW_X_ZOOM)
        theta = atan(float(self.frame.video.s) / float(self.frame.video.distance))
        theta1 = theta - (float(self.position[0] - P[0]) / self.frame.size[0]*FOV_h)
        d1 = self.frame.video.distance / cos(theta1)
        self.cam_distance=d1
        return d1
        
    def linearities(self, C1, C2):
        """how close does this contour fall on the line created the other contours
        C1 is a contour that is the vertex of the angle, and C2 is the other contour point
        http://stackoverflow.com/questions/1211212/how-to-calculate-an-angle-from-three-points
        """
        if self.haspos and C1.haspos and C2.haspos:
            D12 = C1.distance(C2)
            D23 = self.distance(C2)
            D13 = self.distance(C1)
            try:
                return degrees(acos((D12**2 + D13**2 - D23**2) / (2 * D12 * D13)))
            except (ZeroDivisionError, ValueError):
                return 180
        else:
            return None
                
    def add_trajectories(self, trajectory_ids):
        """add a trajectory id to this contours list of trajectories it belongs to"""
        self.trajectories.update(set(trajectory_ids))
        return max(self.trajectories)

    def straight_line_velo(self, other_contour, use_y=True, use_d=True):
        """
        TODO: catch exception for dividing by 0
        This returns the velocity of the contour to another contour in a different frame        
        """
        
        if use_y:
            ball_pos_1 = (self.position[0], self.position[1])
            ball_pos_2 = (other_contour.position[0], other_contour.position[1])
        else:
            ball_pos_1 = (self.position[0], (self.position[1] + other_contour.position[1]) / 2.)
            ball_pos_2 = (other_contour.position[0], (self.position[1] + other_contour.position[1]) / 2.)
        
        if use_d:
            ball_dist_1 = self.cam_distance or self.frame.video.distance
            ball_dist_2 = other_contour.cam_distance or other_contour.distance_to_cam(self.position)
        else:
            ball_dist_1 = ball_dist_2 = float(self.frame.video.distance)

        ball_dist_1 = float(ball_dist_1)
        ball_frame_1 = int(self.frame_id)
        
        ball_dist_2 = float(ball_dist_2)
        ball_frame_2 = int(other_contour.frame_id)
        
        field_of_view_x = self.frame.video.field_of_view_x
        field_of_view_y = self.frame.video.field_of_view_y
        frame_size = self.frame.size
        try:    
            ball_angle_x_1 = (ball_pos_1[0] - (frame_size[0]/2.)) / frame_size[0] * (field_of_view_x)
            ball_angle_x_2 = (ball_pos_2[0] - (frame_size[0]/2.)) / frame_size[0] * (field_of_view_x)
        
            # negative because y comes down from above (coordinates from top left corner)
            ball_angle_y_1 = -(ball_pos_1[1] - (frame_size[1]/2.)) / frame_size[1] * (field_of_view_y)
            ball_angle_y_2 = -(ball_pos_2[1] - (frame_size[1]/2.)) / frame_size[1] * (field_of_view_y)
            
            
            v1 = (ball_dist_1, ball_angle_x_1, ball_angle_y_1)
            v2 = (ball_dist_2, ball_angle_x_2, ball_angle_y_2)
            d= distance_spherical_coords(v2, v1)
            num_frames = abs(ball_frame_2-ball_frame_1)
            ft_per_frame = d / num_frames
            
            # constant 0.68181818 converts ft / sec to miles per hour
            mph = ft_per_frame * self.frame.video.frame_rate * 0.68181818
            return mph
        except ZeroDivisionError:
            return -1

class Trajectory(object):
    """
    A trajectory is an ordered list of contours that belong to eachother
    they belong to one trajectory because they are connected by the closest distance frame to frame
    """
    def __init__(self, id=None, contours=[], disappearance_leeway=20):
        self.contours = []
        self.id = id
        self.killed=False
        self.countup=1
        self.disappearance_leeway=disappearance_leeway
        for contour in contours:
            self.add_contour(contour)
        if len(self.contours) > 0:
            self.contours.sort(key=lambda x: x.frame_id)
    
    def add_contour(self, contour):
        """add a contour to the trajectory"""
        self.contours.append(contour)
        if self.id is not None:
            contour.add_trajectories([self.id])
        self.contours.sort(key=lambda x: x.frame_id)
        #contour.distance_to_cam(self.contours[0].position)

    def get_cam_distances(self):
        """ get the distances to the cam for every contour in the trajectory"""
        for contour in self.contours:
            contour.distance_to_cam(self.contours[0].position)

    def total_distance(self):
        """
        returns the sum of the distances between every sequential
        contour in the trajectory
        """
        return sum([c1.distance(c0) for (c0, c1) in zip(self.contours[:-1], 
                    self.contours[1:])])

    def fit_line(self):
        """ 
        fit a line to the positions of the contours in the trajectory
        """
        ys = np.array([s.y_cord for s in self.contours])
        xs = np.array([s.x_cord for s in self.contours])
        xs = sm.add_constant(xs)
        self.results = sm.OLS(ys, xs).fit()
        return self.results
            
    def tickdown(self):
        self.countup += 1
        if self.countup>=self.disappearance_leeway:
            self.killed=True
            
    def reset_tickdown(self):
        self.countup=1
            
    def remove_outliers(self):
        """
        remove the outlier contours using RANSAC robust linear regression
        on the positions of the contour
        """
        
        horz_center = np.array([c.position[0] for c in self.contours])
        vert_center = np.array([c.position[1] for c in self.contours])
        model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold=4)
        Xs = pd.DataFrame(np.column_stack([horz_center, horz_center**2]))
        model_ransac.fit(Xs, vert_center)
        inlier_mask = model_ransac.inlier_mask_
        self.contours = [c for (c, keep) in zip(self.contours, inlier_mask) if keep]
        
    
    def find_velos(self, max_frame_diff=6):
        """
        This finds the velocity from each contour of the trajectory to other
        contours surrounding it
        
        distance: distance from the camera to the ball
        max_frame_diff: maximum difference in frames to look at the velocity between them
        """
        # contours will be sorted because we sort after every add
        all_velos = []
        num_contours = len(self.contours)
        first_frame = self.contours[0].frame_id
        for i in range(num_contours):
            c1 = self.contours[i]
            x = c1.frame_id - first_frame
            for j in range(i+1, min(i+max_frame_diff+1, num_contours)):
                c2 = self.contours[j]
                y = c2.frame_id - first_frame
                if x - y <= max_frame_diff:
                    f1 = c1.frame_id
                    f2 = c2.frame_id
                    h1 = c1.position[0]
                    h2 = c2.position[0]
                    v1 = c1.position[1]
                    v2 = c2.position[1]
                    d1 = c1.cam_distance
                    d2 = c2.cam_distance
                    fd = f2-f1
                    t = (x + y) / 2.
                    
                    velo = c1.straight_line_velo(c2)
                    velo_no_y = c1.straight_line_velo(c2, use_y=False)
                    velo_no_d = c1.straight_line_velo(c2, use_d=False)
                    velo_no_d_no_y = c1.straight_line_velo(c2, use_y=False, use_d=False)
                    # velo will be -1 if it errorred inside straight_line_velo
                    if velo > 0:
                        all_velos.append((x, y, f1, f2, h1, h2, v1, v2, fd, d1, d2, t, velo, velo_no_y, velo_no_d, velo_no_d_no_y))

        self.all_velos = pd.DataFrame(all_velos, 
                                      columns=['num1', 'num2', 'frame1', 'frame2', 
                                      'h1', 'h2', 'v1', 'v2','fd', 'd1', 'd2', 't', 'velo', 'velo_no_y', 'velo_no_d', 'velo_no_d_no_y'])
        return self.all_velos

    def smooth_data(self, velo_type="velo", ignore_beginning=-1, lbound=0.000015, ubound=0.000085):
        """
        this smooths the position data and the distance data based on 
        the drag equations in the helper functions in the beginning
        
        velo type says which column on all_velos to use to smooth and return
        """
        
        filtered=self.all_velos[self.all_velos.num1>ignore_beginning]
        velo = filtered[velo_type]
        ts = filtered.t
        vname = 'drag_'+velo_type+'/'+str(ignore_beginning) +"/"+str(lbound)+'/'+str(ubound)
        drag_params, reconstructed_velos_drag = smooth_vector_drag_constrained(velo, x=ts, lbound=lbound, ubound=ubound)
        filtered[vname] = reconstructed_velos_drag
        self.all_velos = pd.merge(self.all_velos, filtered[["num1","num2",vname]], on=["num1","num2"], how='left')

        self.v0 = drag_params[0]
        self.c_over_m = drag_params[1]
        
        return (drag_params, reconstructed_velos_drag)

#import cProfile
#
#def do_cprofile(func):
#    def profiled_func(*args, **kwargs):
#        profile = cProfile.Profile()
#        try:
#            profile.enable()
#            result = func(*args, **kwargs)
#            profile.disable()
#            return result
#        finally:
#            profile.print_stats()
#    return profiled_func
#    
    
#@do_cprofile
def find_speed(img_path, rect, distance, zoom, fps=120, s=10., constrained=True):
    """
    The actual function that wil define the speed
    img_path = path to the video
    rect = rectangle around which you think the ball will emerge from  
    distance is distance in feet from camera to pitcher-catcher line
    zoom = 100 or 0 
    fps frames per second
    
    output is a tuple: '
    first position is the estimate of the velocity, 
    second is dataframe of velocities/positions throughout the video
    """
    v2 = Video(img_path, rect=rect, frame_rate=fps, zoom=zoom, distance=distance, s=s)
    print "video loaded"
    # find black and white background subtractor
    v2.apply_difference(history = 75, backgroundRatio=0.95, 
                        nmixtures=3, learningRate=.5)
    print "difference applied"
    # blur frames to get rid of graininess
    v2.blur_frames(9, image="fgmask")
    print "blurred"
    v2.find_contours()
    print "contours found"
    v2.findAllTrajectories()
    print "trajectories found"
    
    # find the most ballyest trajectory
    if len(v2.trajectories) > 0:
        bt = v2.find_ball_trajectory()
        print "ball trajectory is %s" % bt
        ball_traj = v2.trajectories[bt]
        ball_traj.get_cam_distances()
        ball_traj.remove_outliers()
        print "outliers removed"
        ball_traj.find_velos(max_frame_diff=2)
        print "velos found"
        results_df = pd.DataFrame(columns=['pred_velo', 'drag_param', 'velo_type', 'ignore_beg', 'lbound'])
        for velo_type in ['velo']:
            for beg in [0]:
                for lbound, ubound in [(0.00003,0.000085), (0.000047, 0.000048)]:
                    # fix beginning if we did not find many instances
                    beg0 = beg if ball_traj.all_velos.shape[0] > 6 else -1
                    ((pred_velo, drag_param),reconstructed)  = ball_traj.smooth_data(velo_type=velo_type, ignore_beginning=beg0, lbound=lbound, ubound=ubound)
                    results_df.loc[results_df.shape[0]]=[pred_velo, drag_param, velo_type, beg, lbound]            
                    #drag_params_res_constrained, smooth_velos = ball_traj.smooth_data(constrained=True)
                    #drag_params_res_unconstrained, smooth_velos = ball_traj.smooth_data(constrained=False)
        print "data smoothed"
        all_velos = ball_traj.all_velos
        return (results_df, all_velos, v2)
    else:
        return (None, None)
# test the rect 
#common_rect=((0,150),(600,445))
#pitch_info = pd.read_csv('/Users/brianritz/projects/findball/captured_videos/outing_20151206/pitch_info.csv')
#distances = pitch_info.ix[(pitch_info.angle==90)&(pitch_info.fps!=60), ['pitch_no', 'distance_ft', 'fps', 'zoom', 'angle', 'velocity']]
###
###for ind, pitch_no, distance_ft, fps, zoom, deg, velocity in distances.itertuples():
###    print pitch_no
###    d = "".join(['//Users/brianritz/projects/findball/captured_videos/outing_20151206/videos/pitch_',str(pitch_no),'.m4v'])
###    dout = '//Users/brianritz/temp/test_rect'
###    v2 = Video(d, rect=common_rect)
###    v2.draw_rect()
###    v2.write_orig_image(dout, str(pitch_no)+"_", frame_num=[0])
#
#resdf, allv, v2 = find_speed(d, common_rect, distance_ft, zoom, s=15. )
#pitch_no=280
#d = "".join(['//Users/brianritz/projects/findball/captured_videos/outing_20151206/videos/pitch_',str(pitch_no),'.m4v'])
##common_rect=((0,170),(600,430))
#distance_ft=distances.ix[pitch_no, "distance_ft"]
#zoom=distances.ix[pitch_no, "zoom"]
#fps=distances.ix[pitch_no, "fps"]
#v2= Video(d, rect=common_rect, zoom=zoom, distance=distance_ft, s=15., frame_rate=120)
#v2.apply_difference(history = 75, backgroundRatio=0.95, nmixtures=3, learningRate=.5)
#v2.blur_frames(9, image="fgmask")
#print "blurred"
#v2.find_contours()
#print "contours found"
#v2.findAllTrajectories()
#bt = v2.find_ball_trajectory()
#ball_traj = v2.trajectories[bt]
#ball_traj.get_cam_distances()
#ball_traj.remove_outliers()
#ball_traj.find_velos(max_frame_diff=2)
#((pred_velo, drag_param),reconstructed) = ball_traj.smooth_data(velo_type='velo', ignore_beginning=0, lbound=0.00003)
#test41=find_speed(VIDEO_PATH, RECT, DISTANCE_TO_PITCH, ZOOM, fps=FRAMES_PER_SECOND)
#print "Velocity is: %s" % test41[0]
