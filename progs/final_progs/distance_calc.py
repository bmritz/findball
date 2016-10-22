# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 15:24:39 2015

@author: brianritz
"""


from math import sin, cos, acos, radians, degrees, sqrt, tan
import numpy as np
import pandas as pd
import numpy.linalg as linalg
from scipy import optimize

# pitcher to catcher distance ft
PITCHER_CATCHER_DISTANCE=56.4375

# x, y coordinates selected from photo
PITCHERS_PLATE=(966, 1247)
HOME_PLATE=(4533, 1235)

# height of camera in ft
H=5.


# field of view in degrees
FOVx=63.54
FOVy=44.95886879
PIC_LENGTH=3264.
PIC_HEIGHT=2448.
p_c_meters = PITCHER_CATCHER_DISTANCE*0.3048


X_pixels = PITCHERS_PLATE[0] - HOME_PLATE[0]

X_proportion = abs(X_pixels / PIC_LENGTH)
X_angle = radians(X_proportion*FOVx)

distance_meters = p_c_meters / tan(X_angle)

distance_ft = distance_meters*(1/0.3048)
print distance_ft


def simple_x(x, y, pitcher_catcher_distance=60.5):
    pitchers_plate=x
    home_plate=y
    
    p_c_meters = pitcher_catcher_distance*0.3048
    X_pixels = pitchers_plate[0] - home_plate[0]

    X_proportion = abs(X_pixels / PIC_LENGTH)
    X_angle = radians(X_proportion*FOVx)
    print tan(X_angle)
    distance_meters = p_c_meters / tan(X_angle)

    distance_ft = distance_meters*(1/0.3048)
    return distance_ft

def more_complex(X_pixels , y_pixels, pitcher_catcher_distance=60.5, s=0, h=0):
    """ s is number of feet in front of pitchers plate the camera is"""
    #pitchers_plate=x
    #home_plate=y

    p_c_meters = pitcher_catcher_distance*0.3048
    s_meters = s*0.3048
    #X_pixels = pitchers_plate[0] - home_plate[0]
    #y_pixels = pitchers_plate[1] - home_plate[1]
    X_proportion = abs(X_pixels / PIC_LENGTH)
    y_proportion = abs(y_pixels / PIC_HEIGHT)
    X_angle_degrees = X_proportion*FOVx
    X_angle_degrees_front = X_angle_degrees * (pitcher_catcher_distance - s) / pitcher_catcher_distance
    X_angle_radians_front = radians(X_angle_degrees_front)
    y_angle = radians(y_proportion*FOVy)
    distance_meters = (p_c_meters-s_meters) / tan(X_angle_radians_front)

    distance_ft = distance_meters*(1/0.3048)

    # correct for h
    d = sqrt(distance_ft**2 - h**2)    
    
    return d

def panorama(x, y, pitcher_catcher_distance=60.5, s=0):
    #pitchers_plate=x
    #home_plate=y
    
    pano_degrees_per_pixel = FOVx/PIC_LENGTH    
    
    p_c_meters = pitcher_catcher_distance*0.3048
    s_meters = s*0.3048
    X_pixels = x
    y_pixels = y
    X_angle_degrees = X_pixels*pano_degrees_per_pixel
    X_angle_degrees_front = X_angle_degrees * (pitcher_catcher_distance - s) / pitcher_catcher_distance
    X_angle_radians_front = radians(X_angle_degrees_front)
    y_proportion = abs(y_pixels / PIC_LENGTH)
    y_angle = radians(y_proportion*FOVy)
    distance_meters = (p_c_meters-s_meters) / tan(X_angle_radians_front)

    distance_ft = -distance_meters*(1/0.3048)
    return distance_ft
        
def find_dist(X_pixels, y_pixels, pitcher_catcher_distance=60.5, s=0):
    #pitchers_plate=x
    #home_plate=y
    #X_pixels = abs(pitchers_plate[0] - home_plate[0])
    #y_pixels = pitchers_plate[1] - home_plate[1]
    if X_pixels > PIC_LENGTH:
        return panorama(X_pixels, y_pixels, pitcher_catcher_distance, s)
    elif X_pixels < PIC_LENGTH:
        return more_complex(X_pixels, y_pixels, pitcher_catcher_distance, s)
    else:
        raise ValueError
    
def three_d_way(X_diff, y_diff, pitcher_catcher_distance=PITCHER_CATCHER_DISTANCE, s=0., h=H):
    
pd.Series(np.arange(0, PIC_LENGTH))
    
###this was the second outing
    
training = pd.read_excel("/Users/brianritz/projects/findball/distance_calibration/photos_20160210/positions.xlsx")
    
all_pred = []
for row, ser in training.iterrows():
    p1 = (ser.xmin, ser.ymin)
    p2 = (ser.xmax, ser.ymax)
    pred = more_complex(p1, p2, s=0)
    all_pred.append(pred)

training['predicted_dist'] = pd.Series(all_pred)
training['resid'] = training.predicted_dist - training.distance_ft
training['ydiff'] = training.ymax - training.ymin
df =training[['predicted_dist', 'distance_ft', 'resid','ydiff']]
    
#### this was teh first way I did things
training = pd.read_excel("/Users/brianritz/projects/findball/distance_calibration/photo_info.xlsx")
for S in map(lambda x: x/10., range(70, 100)):
    all_pred2=[]
    for row, ser in training.iterrows():
        p1 = (ser.x1, ser.y1)
        p2 = (ser.x2, ser.y2)
        try:
            if ser.pano ==1:
                pred = panorama(p1, p2, s=S)
            else:
                pred=more_complex(p1, p2, s=S)
        except:
            pred = -1
        all_pred2.append(pred)
    
    training['predicted_'+str(S)] = pd.Series(all_pred2)
    training['resid_'+str(S)] = training['predicted_'+str(S)] - training.distance
training['xdiff'] = training['x2'] - training['x1']
training['ydiff'] = training['y2'] - training['y1']
training['resid'] = training['predicted'] - training.distance
training['center'] = (training['x1'] + training['x2'] ) / 2
training.to_csv("/Users/brianritz/temp/training_tuning.csv")


filter_col = [col for col in list(training) if col.startswith('resid')]
filter_col
training[filter_col]
training[filter_col].apply(lambda x: x.pow(2).mean(), axis=0).argmin()

## NOTE: Higher residuals wwhen we are more centered