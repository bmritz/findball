# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 10:53:56 2015

@author: brianritz

changes
1: original code
2: took out extraneous code from the objects
"""


import pandas as pd
import numpy as np
import cv2
import os
import sys
from sklearn import linear_model
from sklearn import cluster
import statsmodels.api as sm
from math import cos, radians, sin, degrees, pi
import itertools
from math import pi, acos, sqrt, degrees, sin, cos, radians, tan
                    
import copy

crop_size= 40

# CONSTANTS
# these were calculated from taking video
FIELD_OF_VIEW_X = 52
FIELD_OF_VIEW_Y = 30.88211

# calculated from measurements
FIELD_OF_VIEW_X_ZOOM = 18.47188362
FIELD_OF_VIEW_Y_ZOOM = 10.52105669

# this is the degrees of the angle it will look for another contour to continute the trajectory
DEGREES = 90

# class objects


class Video(object):
    
    def __init__(self, filename, rect=None, zoom=100, frame_rate=240):
        """
        Make a video object full of frames from a .m4v
        filename:
        filename is the path to the m4v file
        rect:
        rect is top left corner, top right corner--default is entire image
        ((horz cord of top left, vert cord of top left), (horz cord of bottom right, vert cord of bottom right))
        """
        
        vc = cv2.VideoCapture(filename)
        
        self.num_frames=0
        self.frames = []
        # bring in the frames --  set rectangle default with first frame
        if vc.isOpened():
            rval , frame = vc.read()
            self.rect = ((0,0), (frame.shape[0:2])) if rect is None else rect
        else:
            rval = False
        
        while rval:
            self.num_frames += 1
            self.frames.append(Frame(frame, id = self.num_frames-1))
            rval, frame = vc.read()
            cv2.waitKey(1)
            
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
            contours = [ContourBlob(x, frame_id=frame.id) for x in contours]
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
        final_ball = -1
        ball_trajectory = np.argwhere(lengths == np.amax(lengths))
        if len(ball_trajectory)==1:
            final_ball = ball_trajectory[0]
        else:
            max_dist=0
            for i in ball_trajectory:
                if distances[i[0]] > max_dist:
                    max_dist = distances[i[0]]
                    final_ball=i[0]
        self.final_ball = final_ball
        return final_ball
    
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
                            cv2.circle(frame.orig_image, con.position, size, color, -1)
                        if image =="fgmask":
                            cv2.circle(frame.fgmask, con.position, size, color, -1)
    
    def refine_ball_positions(self, crop_size=40):
        """
        refine the positions of the contours that we think are the ball
        once we find the trajectory that is the ball
        """
        for t in self.trajectories:
            if t.id == self.final_ball:
                for c in t.contours:
                    self.frames[c.frame_id].refine_positions(contours = [c], crop_size=crop_size)
                    
    def write_orig_image(self, directory, file_prefix):
        """ write out the images to a directory"""
        clean_dirpath = os.path.abspath(directory)
        for i, frame in enumerate(self.frames):
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
    def __init__(self, pic_file, id):
        self.orig_image = pic_file
        self.orig_image_gr = cv2.cvtColor(self.orig_image, cv2.COLOR_BGR2GRAY)
        self.contours = []
        self.size = self.orig_image.shape
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
    
    PROGRAMMING NOTE: PERHAPS CONTOUR SHOULD TAKE A FRAME OBJECT AS AN ARGUMENT
    """
    
    def __init__(self, contour, frame_id=None):
        self.contour_array = contour
        self.frame_id=frame_id
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
    
    def add_contour(self, contour):
        """add a contour to the trajectory"""
        self.contours.append(contour)
        if self.id is not None:
            contour.add_trajectories([self.id])
        # sorts it for every add -- delete this if worried about speed but its nice to always know we're sorted
        self.contours.sort(key=lambda x: x.frame_id)

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
        self.results = sm.OLS(ys, xs).fit()
        return self.results
            
    def tickdown(self):
        self.countup += 1
        if self.countup>=self.disappearance_leeway:
            self.killed=True
            
    def reset_tickdown(self):
        self.countup=1
            
            
    def smooth_data(self):
        """
        this smooths the position data and the distance data based on linear regression
        """
        df = pd.DataFrame([(c.refined_position+(c.camera_distance,)) for c in self.contours], columns=['x','y','raw_distance'])
        df['smooth_distance'] = pd.ols(y=df.raw_distance, x=pd.Series(range(len(df.raw_distance))), intercept=True).y_fitted
        df_smooth = df.apply(smooth_vector_ols, 0)
        self.smooth_data
        return df_smooth



# some helper functions that will help find the speed

# these were calculated from taking video
FIELD_OF_VIEW_X = 52
FIELD_OF_VIEW_Y = 30.88211

# calculated from measurements
FIELD_OF_VIEW_X_ZOOM = 18.47188362
FIELD_OF_VIEW_Y_ZOOM = 10.52105669
 
zoom=0
frame_size=(1280, 720)

def smooth_vector_ols(v, order=1, x=None):
    """
    this smooths the vector v based on ols
    """
    y = pd.Series(v)
    x = pd.Series(range(len(v))) if x is None else x
    if order == 1:
        y_hat = pd.ols(y=v, x=x, intercept=True).y_fitted
    elif order==2:
        y_hat = pd.ols(y=y, x=pd.DataFrame({'x':x, 'x^2':[x2**2 for x2 in x]})).y_fitted
    else:
        raise "Only order 1 or 2 is supported."
    return y_hat.tolist()

def distance_spherical_coords(point1, point2):
    """ distance in our spherical coordinates between two points """
    r1 = point1[0]
    th1 = radians(point1[1])
    ph1 = radians(point1[2])
    r2 = point2[0]
    th2 = radians(point2[1])
    ph2 = radians(point2[2])
    return sqrt(r1**2 + r2**2 - ((2*r1*r2)*(sin(th1)*sin(th2)*cos(ph1-ph2) + cos(th1)*cos(th2))))


def straight_line_velo(tup1, tup2, frame_rate, zoom):
    """
    tups are (frame_id, camera_distance(mm), x, y)
    """ 
    ball_pos_1 = (tup1[2], tup2[3])
    ball_dist_1 = tup1[1]
    ball_frame_1 = int(tup1[0])
    
    ball_pos_2 = (tup2[2], tup2[3])
    ball_dist_2 = tup2[1]
    ball_frame_2 = int(tup2[0])
    
    if zoom == 100:
        field_of_view_x = FIELD_OF_VIEW_X_ZOOM
        field_of_view_y = FIELD_OF_VIEW_Y_ZOOM
    elif zoom == 0:
        field_of_view_x = FIELD_OF_VIEW_X
        field_of_view_y = FIELD_OF_VIEW_Y
    else:
        raise ValueError("Zoom must be 0 or 100. Other values not yet supported.")
        
    ball_angle_x_1 = (ball_pos_1[0] - (frame_size[0]/2.)) / frame_size[0] * (field_of_view_x)
    ball_angle_x_2 = (ball_pos_2[0] - (frame_size[0]/2.)) / frame_size[0] * (field_of_view_x)

    # negative because y comes down from above (coordinates from top left corner)
    ball_angle_y_1 = -(ball_pos_1[1] - (frame_size[1]/2.)) / frame_size[1] * (field_of_view_y)
    ball_angle_y_2 = -(ball_pos_2[1] - (frame_size[1]/2.)) / frame_size[1] * (field_of_view_y)
    
    
    v1 = (ball_dist_1, ball_angle_x_1, ball_angle_y_1)
    v2 = (ball_dist_2, ball_angle_x_2, ball_angle_y_2)
    distance = distance_spherical_coords(v2, v1)
    num_frames = abs(ball_frame_2-ball_frame_1)
    mm_per_frame = distance / num_frames
    
    # constant 0.00223693629 converts mm / sec to miles per hour
    mph = mm_per_frame * frame_rate * 0.00223693629
    return mph
    

def find_speed(img_path, rect, distance, zoom, fps=120):
    """
    The actual function that wil define the speed
    img_path = path to the video
    rect = rectangle around which you think the ball will emerge from  
    distance is distance in feet from camera to pitcher-catcher line
    zoom = 100 or 0 
    fps frames per second
    """
    v2 = Video(img_path, rect=rect)
    # find black and white background subtractor
    v2.apply_difference(history = 75, backgroundRatio=0.95, 
                        nmixtures=3, learningRate=.5)
    # blur frames to get rid of graininess
    v2.blur_frames(9, image="fgmask")
    v2.find_contours()
    v2.findAllTrajectories()
    
    # find the most ballyest trajectory
    if len(v2.trajectories) > 0:
        bt = v2.find_ball_trajectory()
        ball_traj = v2.trajectories[bt]
        # loop through the trajectory and find the positions
        alllog=[]
        for i, c in enumerate(ball_traj.contours):
            frame_id = c.frame_id
            try:
                ellipse=c.ellipse
                min_rect=c.min_area_rect
                vert_center = ellipse[0][1]
                horz_center = ellipse[0][0]
                vert_center_r = min_rect[0][1]
                horz_center_r = min_rect[0][0]
                alllog.append((frame_id, vert_center, horz_center, vert_center_r, horz_center_r))

            except:
                alllog.append((-1,-1,-1,-1,-1))
    else:
        print "No Trajectories Found"
        return (pd.DataFrame([[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]],columns=['num1', 'num2', 'frame1', 'frame2', 'h1', 'h2', 'v1', 'v2','fd', 'velo']),
                pd.DataFrame([[-1,-1,-1,-1,-1]], columns=['frame_id', 'vert_center', 'horz_center', 'vert_center_r',
                                            'horz_center_r']))

    # now we find hte velocities
    results = pd.DataFrame(alllog, columns=['frame_id', 'vert_center', 'horz_center', 'vert_center_r',
                                            'horz_center_r'])
    results['distance_mm'] = float(distance)/0.00328084
    all_velos = []
    results.reset_index(inplace=True, drop=False)
    
    if results.shape[0] > 4:
        print "we are in results"
        # remove outliers using RANSAC robust linear regression
        model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold=4)
        Xs = pd.DataFrame(np.column_stack([results.horz_center, results.horz_center**2]))
        model_ransac.fit(Xs, results.vert_center)
        inlier_mask = model_ransac.inlier_mask_
        results_in = results[inlier_mask]        
        
        for x, y in itertools.combinations(list(results_in.index), 2):
            if y - x > 0:
                s = 10  # this is the distance from teh pitcher to the point that is the vertex of the perpendicular angle to the camera on the pitcher-catcher line
                f1 = results_in.loc[x, 'frame_id']
                f2 = results_in.loc[y, 'frame_id']
                h1 = results_in.loc[x, 'horz_center']
                h2 = results_in.loc[y, 'horz_center']
                v1 = results_in.loc[x, 'vert_center']
                v2 = results_in.loc[y, 'vert_center']
                fd = f2-f1
                b1 = results_in.loc[x,["frame_id", "distance_mm", "horz_center", "vert_center"]].tolist()
                b2 = results_in.loc[y,["frame_id", "distance_mm", "horz_center", "vert_center"]].tolist()
                try:
                    all_velos.append((x,y,f1,f2,h1,h2,v1,v2,fd,straight_line_velo(b1, b2, fps, zoom)))
                except ZeroDivisionError:
                    print "we got zerodiverror"
                    all_velos.append((-1,-1,-1,-1,-1,-1,-1,-1,-1,-1))
        return (pd.DataFrame(all_velos, columns=['num1', 'num2', 'frame1', 'frame2', 'h1', 'h2', 'v1', 'v2','fd', 'velo']),results)
    else:
        print "we did notmee shape reqs"
        return (pd.DataFrame([[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]],columns=['num1', 'num2', 'frame1', 'frame2', 'h1', 'h2', 'v1', 'v2','fd', 'velo']), results)
# TESTING

common_rect=((0,170),(600,430))
pitch_vid='//Users/brianritz/projects/findball/captured_videos/outing_20151004/videos/pitch_41.m4v'
z=0
dist=30
test41=find_speed(pitch_vid, common_rect, dist, z, fps=240)

testvids = range(641, 660)
testvids.remove(643)
testdists=[30,30,30,30,30,30,30,30,30,50,50,50,50,50,70,70,70,70]

testdirs = map(lambda x:"".join(['//Users/brianritz/projects/findball/captured_videos/outing_20151004/videos/pitch_',str(x),'.m4v']),testvids)
testparams = zip(testdirs, testdists, testvids)

outlist=[]
for d, dis,v in testparams:
    sp, res = find_speed(d, common_rect, dis, z)
    sp['pitch_no']=v
    res['pitch_no'] = v
    outlist.append((sp, res))
    print d

# TODO: change distance to the ball based on trigonometry

all_results_sp = pd.concat([x[0] for x in outlist])
all_results_res = pd.concat([x[1] for x in outlist])
all_results_res.to_csv("//Users/brianritz/projects/findball/temp/all_results_res.csv")
all_results_sp.to_csv("//Users/brianritz/projects/findball/temp/all_results_sp.csv")


common_rect=((0,170),(600,430))
pitch_info = pd.read_csv('/Users/brianritz/projects/findball/captured_videos/outing_20151004/pitch_info.csv')
distances = pitch_info.ix[(pitch_info.degrees==90)&(pitch_info.fps!=60), ['pitch_no', 'distance_ft', 'fps', 'zoom', 'degrees', 'velocity']]
outlist=[]
for ind, pitch_no, distance_ft, fps, zoom, deg, velocity in distances.itertuples():
    d = "".join(['//Users/brianritz/projects/findball/captured_videos/outing_20151004/videos/pitch_',str(pitch_no),'.m4v'])
    sp, res = find_speed(d, common_rect, distance_ft, zoom, fps)
    sp['pitch_no']=pitch_no
    res['pitch_no'] = pitch_no
    outlist.append((sp, res))
    print d

all_results_sp = pd.concat([x[0] for x in outlist])
all_results_res = pd.concat([x[1] for x in outlist])
all_results_res.to_csv("//Users/brianritz/projects/findball/output/all_results_res5.csv")
all_results_sp.to_csv("//Users/brianritz/projects/findball/output/all_results_sp5.csv")
    
all_results_sp = pd.read_csv("//Users/brianritz/projects/findball/output/all_results_sp5.csv", index_col=0)
# assess accuracy

predictions = all_results_sp.groupby(['pitch_no', 'fd'])['velo'].max()

predictions = predictions.reset_index(level=-1)
predictions_and_actuals = pd.merge(predictions, distances, left_index=True, right_index=True)
predictions_and_actuals['resid'] = predictions_and_actuals.velo - predictions_and_actuals.velocity

predictions_and_actuals.to_csv("/Users/brianritz/temp/pred_and_act.csv")
fps240 = predictions_and_actuals[(predictions_and_actuals.fd==3) & (predictions_and_actuals.fps==240)]
fps120 = predictions_and_actuals[(predictions_and_actuals.fd==3) & (predictions_and_actuals.fps==120)]



import scipy.optimize as opt

# smooths out the distance assuming constant velocity and stable camera
y = df.raw_distance
maxy = np.argmax(y)
miny = np.argmin(y)
keeps = [x not in (miny, maxy) for x in df.index]
#smoothed_positions = df.loc[keeps, ['frame_id','raw_distance','x','y']].apply(smooth_vector_ols, 0, args=(2,df.frame_id[keeps]))
#smoothed_positions.frame_id 
#y = y.drop(maxy).drop(miny)
## perhaps take out outliers here
#x = df.frame_id
#x = x.drop(maxy).drop(miny)
def fun(pars, x):
    xs, dxdt, ys, dydt, zs, dzdt = pars
    return ((xs+(x*dxdt))**2+((ys+(x*dydt))**2)+((zs+(x*dzdt))**2))**(1./2)


def objective(pars):
    xs, dxdt, ys, dydt, zs, dzdt = pars
    #return np.sum((y-(((xs+(x*dxdt))**2+((ys+(x*dydt))**2))**(1./2)))**2)
    return np.sum((y-pd.Series(x).apply(lambda(t): fun(pars, t)))**2)
res1 = opt.minimize(objective, x0=np.array([100, 50, -50, -70, -2, 10]), method='BFGS', options={'disp': True ,'eps' : 1e0})
res.x

# fit a function that represents drag to the equation
def fun2(pars, t):
    v0, c_over_m = pars
    return 1/((1/v0) + c_over_m*t)

pitch66=all_results_sp[(all_results_sp.pitch_no==117)&(all_results_sp.fd==1)]
velo66 = pitch66.velo
frame66 = pitch66.num2
def objective2(pars):
    v0, c_over_m = pars
    return np.sum((velo66-pd.Series(frame66).apply(lambda(t): fun2(pars, t)))**2)

res66 = opt.minimize(objective2, x0=np.array([60.0, 0.003]), method='Nelder-Mead', options={'disp': True})

def smooth_vector_ols(v, order=1, x=None):
    """
    this smooths the vector v based on ols-- may want to use thisto smooth velocity
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

def simple_drag_func(t, pars):
    v0, c_over_m = pars
    return 1/((1/v0) + c_over_m*t)

def smooth_vector_drag(v, x=None):
    y = pd.Series(v)
    x = pd.Series(range(len(v))) if x is None else pd.Series(x)
    def objective(pars):
        vo, c_over_m = pars
        return np.sum((y-x.apply(lambda(t): simple_drag_func(t, pars)))**2)
    optimized = opt.minimize(objective, x0=np.array([60.0, 0.003]), method='Nelder-Mead', options={'disp': True})
    parameter_estimates = optimized.x
    reconstructed_velos_drag = x.apply(lambda(t): simple_drag_func(t, parameter_estimates) )
    return (parameter_estimates, reconstructed_velos_drag)
    
def smooth_estimates(ds):
    """all three smoothed estimates"""
    velo=ds.velo
    frame=ds.frame1
    ts = ds.t
    pitch_no=ds.pitch_no
    
    def objective(pars):
        v0, c_over_m = pars
        return np.sum((velo-pd.Series(frame).apply(lambda(t): simple_drag_func(t, pars)))**2)
    
    optimized = opt.minimize(objective, x0=np.array([60.0, 0.003]), method='Nelder-Mead', options={'disp': True})
    parameter_estimates = optimized.x
    
    reconstructed_velos_drag = smooth_vector_drag(velo, x=frame)[1]
    reconstructed_velos_sm1 = smooth_vector_ols(velo, order=1, x=frame)
    reconstructed_velos_sm2 = smooth_vector_ols(velo, order=2, x=frame)
    
    return pd.DataFrame({"pitch_no":pitch_no, "frame":frame, "orig_velo": velo, "drag_velo":reconstructed_velos_drag,
                         "smooth1_velo":reconstructed_velos_sm1,
                         "smooth2_velo":reconstructed_velos_sm2})


### use these to make new predictions
groups = all_results_sp[all_results_sp.fd==1].groupby('pitch_no')

smoothed_velos=  groups.apply(smooth_estimates)

smoothed_predictions = smoothed_velos.groupby("pitch_no")[['drag_velo', 'orig_velo', 'smooth1_velo', 'smooth2_velo']].max()

sm_predictions_and_actuals = pd.merge(smoothed_predictions, distances, left_index=True, right_index=True)
sm_predictions_and_actuals['resid'] = sm_predictions_and_actuals.velo - predictions_and_actuals.velocity

####


#### use all measurements to try tofit a line

all_results_sp['t'] = (all_results_sp['num1'] + all_results_sp['num2']) / 2
all_results_sp = all_results_sp[all_results_sp.t != -1]

test = all_results_sp.sort(columns = ['pitch_no', 't', 'fd']).groupby(['pitch_no','t'])['velo'].mean()


def smooth_estimates2(ds):
    """all three smoothed estimates"""
    velo=ds.velo
    frame=ds.frame1
    frame2=ds.frame2
    fd = ds.fd
    ts = ds.t
    pitch_no=ds.pitch_no
    
    #def objective(pars):
    #    v0, c_over_m = pars
    #    return np.sum((velo-pd.Series(frame).apply(lambda(t): simple_drag_func(t, pars)))**2)
    
    #optimized = opt.minimize(objective, x0=np.array([60.0, 0.003]), method='Nelder-Mead', options={'disp': True})
    #parameter_estimates = optimized.x
    
    drag_params, reconstructed_velos_drag = smooth_vector_drag(velo, x=ts)
    reconstructed_velos_sm1 = smooth_vector_ols(velo, order=1, x=ts)
    reconstructed_velos_sm2 = smooth_vector_ols(velo, order=2, x=ts)
    
    return pd.DataFrame({"pitch_no":pitch_no, "frame1":frame,"frame2":frame2, "fd":fd,"orig_velo": velo, "drag_velo":reconstructed_velos_drag,
                         "drag_max_velo":drag_params[0],"drag_param_1":drag_params[1],
                         "smooth1_velo":reconstructed_velos_sm1[1],"smooth1_beta0":reconstructed_velos_sm1[0][0],
                         "smooth1_beta1":reconstructed_velos_sm1[0][1], 
                         "smooth2_velo":reconstructed_velos_sm2[1], "smooth2_beta0":reconstructed_velos_sm2[0][0], 
                         "smooth2_beta1":reconstructed_velos_sm2[0][1], "smooth2_beta2":reconstructed_velos_sm2[0][2],
                         "t":ts})

smooth_velos2 = all_results_sp[all_results_sp.fd<3].groupby('pitch_no').apply(smooth_estimates2)

smooth_velos2.to_csv('//Users/brianritz/temp/smoothed_velos2.csv', index=False )
preds = smooth_velos2.groupby('pitch_no')['drag_max_velo'].mean()

smooth_predictions_and_actuals = pd.merge(preds, distances, left_index=True, right_index=True,how='left')
