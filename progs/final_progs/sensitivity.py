# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 20:52:25 2016

@author: brianritz
"""

from math import cos, radians, sin, degrees, pi, sqrt

# sensitivity analysis

theta2s = np.arange(0, 2*pi, 2*pi/50)
theta1s = np.arange(0, 2*pi, 2*pi/50)
phi1s = np.arange(0, 2*pi, 2*pi/50)
phi2s = np.arange(0,2*pi, 2*pi/50)
x = {'phi1':phi1s, 'phi2':phi2s, 'theta1':theta1s, 'theta2':theta2s}
all_combos = pd.DataFrame(list(itertools.product(*x.values())), columns=x.keys())

def deriv(phi1, phi2, theta1, theta2):
    """ I found this derivative on in my notebook """
    
    x = sqrt(2 - 2*((sin(theta1)*sin(theta2)*cos(phi1-phi2))+(cos(theta1)*cos(theta2))))
    return x
    
def row_derive(row):
    try:
        return deriv(row['phi1'], row['phi2'], row['theta1'], row['theta2'])
    except:
        return -99
    
derivatives = all_combos.apply(row_derive, axis=1)

derivatives.max()
derivatives.min()
derivatives[derivatives!=-99].min()

all_combos['derivative'] = derivatives

#####
frame_size= (1280, 720)
FIELD_OF_VIEW_X = 52
FIELD_OF_VIEW_Y = 30.88211

# calculated from measurements
FIELD_OF_VIEW_X_ZOOM = 18.47188362
FIELD_OF_VIEW_Y_ZOOM = 10.52105669

mask_theta = ((all_combos.theta2 - all_combos.theta1) < radians(FIELD_OF_VIEW_X)) & (all_combos.theta2 > all_combos.theta1)
mask_phi = ((all_combos.phi2 - all_combos.phi1) < radians(FIELD_OF_VIEW_Y)) & (all_combos.phi2 > all_combos.phi1)
feasible_combos = all_combos[(mask_theta) & (mask_phi)]


# look at possible angles we actuall found

all_res = pd.read_csv("//Users/brianritz/projects/findball/output/all_results_res5.csv", index_col=0)
vertical_diffs = all_res.groupby('pitch_no')[['vert_center']].apply(lambda x: np.max(np.abs(x.diff())))
horizontal_diffs = all_res.groupby('pitch_no')[['horz_center']].apply(lambda x: np.max(np.abs(x.diff())))

pitch_info = pd.read_csv('/Users/brianritz/projects/findball/captured_videos/outing_20151004/pitch_info.csv')
distances = pitch_info.ix[(pitch_info.degrees==90)&(pitch_info.fps!=60), ['pitch_no', 'distance_ft', 'fps', 'zoom', 'degrees', 'velocity']]

zoomed_out=distances.ix[(distances.zoom==0) &(distances.fps==120),['pitch_no', 'distance_ft']]
zoomed_in=distances.ix[distances.zoom==100),'pitch_no']

# take 11 as the vertical difference
# take 50 as the horizontal difference
vd = 3.
hd = 30.

probable_combos = feasible_combos[feasible_combos.theta2 - feasible_combos.theta1 < radians(FIELD_OF_VIEW_X*0.04)]
probable_combos.shape

probable_max_derivative = deriv(0, radians(vd/frame_size[1]*FIELD_OF_VIEW_Y),
                                0, radians(hd/frame_size[0]*FIELD_OF_VIEW_X))

#feet per sec\

 to mph
fps_to_mph = lambda fps: 0.681818*fps

fps_to_mph(probable_max_derivative*120)