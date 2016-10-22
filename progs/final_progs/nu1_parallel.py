# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 11:21:42 2016

@author: brianritz
"""
import find_speed6
#import find_speed
import pandas as pd
#from multiprocessing import Process
import multiprocessing as mp
import time
import os

#mp.sys.executable = '/Users/brianritz/.virtualenvs/findball/bin/python2.7'

VIDEO_PATH = "./pitch_41.m4v"
VIDEO_PATH = '//Users/brianritz/projects/findball/captured_videos/outing_20151004/videos/pitch_41.m4v'

#FRAMES_PER_SECOND = 240
#DISTANCE_TO_PITCH = 30.
#ZOOM = 0
#RECT = ((0, 170), (600, 430))

common_rect=((0,170),(600,430))
pitch_info = pd.read_csv('/Users/brianritz/projects/findball/captured_videos/outing_20151004/pitch_info.csv')
distances = pitch_info.ix[(pitch_info.degrees==90)&(pitch_info.fps!=60), ['pitch_no', 'distance_ft', 'fps', 'zoom', 'degrees', 'velocity']]
outlist1=[]
outlist2=[]
#for ind, pitch_no, distance_ft, fps, zoom, deg, velocity in distances.iloc[:20,:].itertuples():
#    d = "".join(['//Users/brianritz/projects/findball/captured_videos/outing_20151004/videos/pitch_',str(pitch_no),'.m4v'])
#    param1, res1 = find_speed6.find_speed(d, common_rect, distance_ft, zoom, fps, s=9.)
#    #param2, res2 = find_speed.find_speed(d, common_rect, distance_ft, zoom, fps)
#    outlist1.append((pitch_no, param1, res1))
#    #outlist2.append((pitch_no, param, res))
#    print d
    
## attempt at multiprocessing

def info(title):
    print title
    print 'module name:', __name__
    if hasattr(os, 'getppid'):  # only available on Unix
        print 'parent process:', os.getppid()
    print 'process id:', os.getpid()

def fspeed(pars):
    info('function fspeed')
    pitch_no, distance_ft, zoom, fps = pars
    os.system('echo $VIRTUAL_ENV')
    os.system('which ffmpeg')
    d = "".join(['//Users/brianritz/projects/findball/captured_videos/outing_20151004/videos/pitch_',str(pitch_no),'.m4v'])
    x = find_speed6.find_speed(d, common_rect, distance_ft, zoom, fps, s=9.)
    return x

if __name__=="__main__":
    #p = mp.Pool(2)
    top2 = distances.iloc[:2,:]
    start = time.time()
    #results = p.map(fspeed, zip(top2.pitch_no, top2.distance_ft, top2.zoom, top2.fps))
    #results = map(fspeed, zip(top2.pitch_no, top2.distance_ft, top2.zoom, top2.fps))
    info('main line')    
    #multiple_results = [p.apply_async(fspeed, ((w,x,y,z),)) for w,x,y,z in zip(top2.pitch_no, top2.distance_ft, top2.zoom, top2.fps)]
    #print [res.get(timeout=1) for res in multiple_results]
    p = mp.Process(target=fspeed, args=((20,30, 0, 120),))
    p.start()
    p.join()
    
    #print [x[0] for x in results]
    end = time.time()
    print(end - start)
