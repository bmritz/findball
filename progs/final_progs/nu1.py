# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 11:21:42 2016

@author: brianritz
"""
import find_speed6
import time
import os
import pandas as pd

POSITIONS_RESULTS_OUTPUT = '/Users/brianritz/projects/findball/output/nu1_findball6_may18/positions2.csv'
PREDICTIONS_RESULTS_OUTPUT = '/Users/brianritz/projects/findball/output/nu1_findball6_may18/predictions2.csv'

import logging
now = time.strftime("%Y_%m_%d_%H_%M_%S")
log_dir = '/Users/brianritz/projects/findball/logs/'
prog_name = os.path.basename(__file__).strip('.py')
log_filename = "".join([log_dir, prog_name, now, ".log"])
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

logging.info("Script Started")

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)


if __name__=="__main__":
    common_rect=((0,170),(600,430))
    pitch_info = pd.read_csv('/Users/brianritz/projects/findball/captured_videos/outing_20151004/pitch_info.csv')
    distances = pitch_info.ix[(pitch_info.degrees==90)&(pitch_info.fps!=60)&(pitch_info.pitch_no==41), ['pitch_no', 'distance_ft', 'fps', 'zoom', 'degrees', 'velocity']]
    print distances    
    outlist1=[]
    ensure_dir(PREDICTIONS_RESULTS_OUTPUT)
    for ind, pitch_no, distance_ft, fps, zoom, deg, velocity in distances.itertuples():
        try:
            logging.info("Pitch %s Started" % (pitch_no, )) 
            d = "".join(['//Users/brianritz/projects/findball/captured_videos/outing_20151004/videos/pitch_',str(int(pitch_no)),'.m4v'])
            print d            
            start = time.time()    
            param1, res1 = find_speed6.find_speed(d, common_rect, distance_ft, zoom, fps, s=9.)
            end = time.time()
            elapsed = end - start
            outlist1.append((pitch_no, elapsed, param1, res1))
            logging.info("Pitch Number %s done." % (pitch_no,))
            results = pd.concat([x[3].assign(pitch_no=x[0]) for x in outlist1])
            results.to_csv(POSITIONS_RESULTS_OUTPUT, index=False)
    
            predictions= pd.DataFrame([(x[0], x[1], x[2][0][0], x[2][0][1], x[2][1][0], x[2][1][1]) for x in outlist1], 
                                       columns=['pitch_no', 'elapsed_time', 'pred_velo_constrained', 'drag_param_constrained', 'pred_velo_unconstrained', 'drag_param_unconstrained'])
                                       
            predictions = predictions.merge(distances, how='inner')
            predictions.to_csv(PREDICTIONS_RESULTS_OUTPUT, index=False)
            logging.info("Pitch Number %s written." % (pitch_no,))
        except:
            pass