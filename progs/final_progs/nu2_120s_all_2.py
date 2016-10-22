# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 11:21:42 2016

@author: brianritz
"""
import find_speed7
import time
import os
import pandas as pd
import cPickle as pkl

GROUP=2

OUTPUT_DIRECTORY = '/Users/brianritz/projects/findball/output/nu2_findball_all/'
OUTPUT_PREFIX = 'nu2_120s_all_s15_'+str(GROUP)

POSITIONS_RESULTS_OUTPUT = OUTPUT_DIRECTORY+OUTPUT_PREFIX+'_positions.csv'
PREDICTIONS_RESULTS_OUTPUT = OUTPUT_DIRECTORY+OUTPUT_PREFIX+'_predictions.csv'
TIMING_OUTPUT = OUTPUT_DIRECTORY+OUTPUT_PREFIX+'_timing.csv'
ERRORS_OUTPUT = OUTPUT_DIRECTORY+OUTPUT_PREFIX+'_errors.csv'
PICKLE_OUTPUT = OUTPUT_DIRECTORY+OUTPUT_PREFIX+'_pickle.pkl'
IMAGE_OUTPUT = OUTPUT_DIRECTORY+"images/"

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
    pitch_info = pd.read_csv('/Users/brianritz/projects/findball/captured_videos/outing_20151206/pitch_info.csv')
    #pitch_info=pitch_info[(pitch_info.training_set)]
    pitch_info = pitch_info[pitch_info.group == GROUP]
    distances = pitch_info.ix[(pitch_info.angle==90), ['pitch_no', 'distance_ft', 'fps', 'zoom', 'angle', 'velocity']]
    common_rect=((0,160),(600,445))   
    outlist1=[]
    errors = []
    ensure_dir(PREDICTIONS_RESULTS_OUTPUT)
    for ind, pitch_no, distance_ft, fps, zoom, deg, velocity in distances.itertuples():
        try:
            logging.info("Pitch %s Started" % (pitch_no, )) 
            d = "".join(['//Users/brianritz/projects/findball/captured_videos/outing_20151206/videos/pitch_',str(pitch_no),'.m4v'])      
            start = time.time()
            param1, res1, v2 = find_speed7.find_speed(d, common_rect, distance_ft, zoom, fps, s=15.)
            end = time.time()
            for traj in v2.trajectories:
                if traj.id != v2.final_ball:
                    v2.mark_trajectory(traj.id, size=15)
                else:
                    v2.mark_trajectory(traj.id, size=15, color=(0,255,0))
            image_dirname = IMAGE_OUTPUT+'pitch_'+str(pitch_no)+"/"
            ensure_dir(image_dirname)
            #v2.write_orig_image(image_dirname, 'pitch_')
            elapsed = end - start
            if param1 is not None and res1 is not None:
                outlist1.append((pitch_no, elapsed, param1, res1))
                logging.info("Pitch Number %s done." % (pitch_no,))
                results = pd.concat([x[3].assign(pitch_no=x[0]) for x in outlist1])
                results.to_csv(POSITIONS_RESULTS_OUTPUT, index=False)
        
                predictions =pd.concat([x[2].assign(pitch_no=x[0]) for x in outlist1])
                          
                predictions = predictions.merge(distances.rename(columns={'pitch number': 'pitch_no'}), how='inner')
                predictions.to_csv(PREDICTIONS_RESULTS_OUTPUT, index=False)
                
                pd.DataFrame([(x[0], x[1]) for x in outlist1], columns=['pitch_no', 'elapsed_time']).to_csv(TIMING_OUTPUT, index=False)
                logging.info("Pitch Number %s written." % (pitch_no,))
        except:
            errors.append(pitch_no)
    pd.Series(errors).to_csv(ERRORS_OUTPUT, index=False)
    pkl.dump(outlist1, open(PICKLE_OUTPUT, 'wb'))
    