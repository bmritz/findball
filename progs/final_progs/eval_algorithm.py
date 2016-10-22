# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 11:14:04 2016

@author: brianritz
"""

# evaluate videos
import pandas as pd

   
all_results_sp = pd.read_csv("//Users/brianritz/projects/findball/output/nu1_findball6/nu1_120s_positions.csv")
# assess accuracy

#predictions = all_results_sp.groupby(['pitch_no', 'fd'])['velo'].max()
#
#predictions = predictions.reset_index(level=-1)
#predictions_and_actuals = pd.merge(predictions, distances, left_index=True, right_index=True)
#predictions_and_actuals['resid'] = predictions_and_actuals.velo - predictions_and_actuals.velocity
#
#predictions_and_actuals.to_csv("/Users/brianritz/temp/pred_and_act.csv")
#fps240 = predictions_and_actuals[(predictions_and_actuals.fd==3) & (predictions_and_actuals.fps==240)]
#fps120 = predictions_and_actuals[(predictions_and_actuals.fd==3) & (predictions_and_actuals.fps==120)]

import numpy as np
import scipy.optimize as opt

# smooths out the distance assuming constant velocity and stable camera
#y = df.raw_distance
#maxy = np.argmax(y)
#miny = np.argmin(y)
#keeps = [x not in (miny, maxy) for x in df.index]
#smoothed_positions = df.loc[keeps, ['frame_id','raw_distance','x','y']].apply(smooth_vector_ols, 0, args=(2,df.frame_id[keeps]))
#smoothed_positions.frame_id 
#y = y.drop(maxy).drop(miny)
## perhaps take out outliers here
#x = df.frame_id
#x = x.drop(maxy).drop(miny)
#def fun(pars, x):
#    xs, dxdt, ys, dydt, zs, dzdt = pars
#    return ((xs+(x*dxdt))**2+((ys+(x*dydt))**2)+((zs+(x*dzdt))**2))**(1./2)
#

#def objective(pars):
#    xs, dxdt, ys, dydt, zs, dzdt = pars
#    #return np.sum((y-(((xs+(x*dxdt))**2+((ys+(x*dydt))**2))**(1./2)))**2)
#    return np.sum((y-pd.Series(x).apply(lambda(t): fun(pars, t)))**2)
#res1 = opt.minimize(objective, x0=np.array([100, 50, -50, -70, -2, 10]), method='BFGS', options={'disp': True ,'eps' : 1e0})
#res.x
#
## fit a function that represents drag to the equation
#def fun2(pars, t):
#    v0, c_over_m = pars
#    return 1/((1/v0) + c_over_m*t)
#
#pitch66=all_results_sp[(all_results_sp.pitch_no==117)&(all_results_sp.fd==1)]
#velo66 = pitch66.velo
#frame66 = pitch66.num2
#def objective2(pars):
#    v0, c_over_m = pars
#    return np.sum((velo66-pd.Series(frame66).apply(lambda(t): fun2(pars, t)))**2)
#
#res66 = opt.minimize(objective2, x0=np.array([60.0, 0.003]), method='Nelder-Mead', options={'disp': True})

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

def simple_drag_func(pars, t):
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
    
#def smooth_estimates(ds):
#    """all three smoothed estimates"""
#    velo=ds.velo
#    frame=ds.frame1
#    ts = ds.t
#    pitch_no=ds.pitch_no
#    
#    def objective(pars):
#        v0, c_over_m = pars
#        return np.sum((velo-pd.Series(frame).apply(lambda(t): simple_drag_func(t, pars)))**2)
#    
#    optimized = opt.minimize(objective, x0=np.array([60.0, 0.003]), method='Nelder-Mead', options={'disp': True})
#    parameter_estimates = optimized.x
#    
#    reconstructed_velos_drag = smooth_vector_drag(velo, x=frame)[1]
#    reconstructed_velos_sm1 = smooth_vector_ols(velo, order=1, x=frame)
#    reconstructed_velos_sm2 = smooth_vector_ols(velo, order=2, x=frame)
#    
#    return pd.DataFrame({"pitch_no":pitch_no, "frame":frame, "orig_velo": velo, "drag_velo":reconstructed_velos_drag,
#                         "smooth1_velo":reconstructed_velos_sm1,
#                         "smooth2_velo":reconstructed_velos_sm2})


### use these to make new predictions
#groups = all_results_sp[all_results_sp.fd==1].groupby('pitch_no')
#
#smoothed_velos=  groups.apply(smooth_estimates)
#
#smoothed_predictions = smoothed_velos.groupby("pitch_no")[['drag_velo', 'orig_velo', 'smooth1_velo', 'smooth2_velo']].max()
#
#sm_predictions_and_actuals = pd.merge(smoothed_predictions, distances, left_index=True, right_index=True)
#sm_predictions_and_actuals['resid'] = sm_predictions_and_actuals.velo - predictions_and_actuals.velocity

####


#### use all measurements to try tofit a line

all_results_sp['t'] = (all_results_sp['num1'] + all_results_sp['num2']) / 2
all_results_sp = all_results_sp[all_results_sp.t != -1]

test = all_results_sp.sort(columns = ['pitch_no', 't', 'fd']).groupby(['pitch_no','t'])['velo'].mean()

def smooth_vector_drag_constrained(v, x=None, lbound=0.000015, ubound=0.000085):
    y = pd.Series(v)
    x = pd.Series(range(len(v))) if x is None else pd.Series(x)
    
    ## least squares way http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares
    def fun_resids(pars, t, y):
        return simple_drag_func(pars, t) - y

    x0=np.array([60.0, ((lbound+ubound)/2.)])
    constraints = ([35., lbound], [105., ubound])
    res_2 = opt.least_squares(fun_resids, x0, args=(x, y), bounds=constraints)    
    parameter_estimates = res_2.x
    reconstructed_velos_drag = x.apply(lambda(t): simple_drag_func(parameter_estimates, t) )
    return (parameter_estimates, reconstructed_velos_drag)

def smooth_estimates2(ds, velo_type='velo', ignore_beginning=0, ignore_end=20, lbound=0.00003,ubound=0.000085):
    """all three smoothed estimates"""
    filtered=ds[ds.num1>ignore_beginning]
    filtered = filtered.iloc[0:ignore_end,:]
    velo=filtered[velo_type]
    frame=filtered.frame1
    frame2=filtered.frame2
    fd = filtered.fd
    ts = (filtered['num1'] + filtered['num2']) / 2.
    pitch_no=filtered.pitch_no
    
    #def objective(pars):
    #    v0, c_over_m = pars
    #    return np.sum((velo-pd.Series(frame).apply(lambda(t): simple_drag_func(t, pars)))**2)
    
    #optimized = opt.minimize(objective, x0=np.array([60.0, 0.003]), method='Nelder-Mead', options={'disp': True})
    #parameter_estimates = optimized.x
    
    drag_params, reconstructed_velos_drag = smooth_vector_drag_constrained(velo, x=ts, lbound=lbound,ubound=ubound)
    reconstructed_velos_sm1 = smooth_vector_ols(velo, order=1, x=ts)
    reconstructed_velos_sm2 = smooth_vector_ols(velo, order=2, x=ts)
    
    return pd.DataFrame({"pitch_no":pitch_no, "frame1":frame,"frame2":frame2, "fd":fd,"orig_velo": velo, "drag_velo":reconstructed_velos_drag,
                         "drag_max_velo":drag_params[0],"drag_param_1":drag_params[1],
                         "smooth1_velo":reconstructed_velos_sm1[1],"smooth1_beta0":reconstructed_velos_sm1[0][0],
                         "smooth1_beta1":reconstructed_velos_sm1[0][1], 
                         "smooth2_velo":reconstructed_velos_sm2[1], "smooth2_beta0":reconstructed_velos_sm2[0][0], 
                         "smooth2_beta1":reconstructed_velos_sm2[0][1], "smooth2_beta2":reconstructed_velos_sm2[0][2],
                         "t":ts})

smooth_velos2 = [all_results_sp[all_results_sp.fd<3].groupby('pitch_no').apply(smooth_estimates2, ignore_end=60, lbound=x-0.0000001, ubound=x).assign(ubound=x) for x in np.arange(0.000044, 0.000055,0.0000005)]
all_velos = pd.concat(smooth_velos2)

preds = all_velos.groupby(['pitch_no', 'ubound'])['drag_max_velo', 'drag_param_1', 'smooth1_beta0',  'smooth1_beta1'].mean()

pred_and_actual = pd.merge(preds.reset_index(), distances, how='left', on='pitch_no')
pred_and_actual['resid_drag']= pred_and_actual.drag_max_velo-pred_and_actual.velocity
pred_and_actual['beta_drag']= pred_and_actual.smooth1_beta1-pred_and_actual.velocity
pred_and_actual = pred_and_actual[pred_and_actual.velocity>0]
test = pred_and_actual.groupby('ubound')['resid_drag', 'beta_drag'].agg(lambda x: (x**2).mean())

# use0.000047 as c_over_m