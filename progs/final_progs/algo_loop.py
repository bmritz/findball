# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 11:16:21 2016

@author: brianritz
"""

# call algo



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
