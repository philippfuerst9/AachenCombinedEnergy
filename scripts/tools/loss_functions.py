#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/icetray-start
#METAPROJECT /home/pfuerst/i3_software_py3/combo/build

import xgboost as xgb
import numpy as np

def rmse(pred, dtrain):
    grad = pred - dtrain.get_label()
    hess = np.ones(len(grad))
    return grad, hess
        
def rmse_err(pred, dtrain):
    '''
    return a pair (name, result) name cannot contain : or space
    '''
    truth = dtrain.get_label()
    return 'rmse_err', np.sqrt( np.mean(np.power(pred-truth, 2)))



#pseudo huber loss error function
def pseudo_huber_loss(preds, dtrain):
    '''
    Huber-Loss: delta^2 * (sqrt(1 + (x/delta)^2)-1)
    
    '''
    x = preds - dtrain.get_label() # .get_labels() for sklearn
    delta = 1  #delta is the approached slope for big x
    scale = 1 + (x / delta) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = x / scale_sqrt
    hess = np.abs(delta)**3 / scale / scale_sqrt
    return grad, hess

def pseudo_huber_loss_err(preds, dtrain):
    #get it from wikipedia /w delta = 1
    #check compatibility -> yes!
    x = preds - dtrain.get_label() # .get_labels() for sklearn
    delta = 1  #delta is the approached slope for big x
    scale = 1 + (x / delta) ** 2
    scale_sqrt = np.sqrt(scale)
    huber = delta**2 * (scale_sqrt -1)
    return 'Cherr', np.mean(huber)

#pseudo huber loss error function with adjustable slope 

def pseudo_huber_loss_k(preds, dtrain, delta = 1):
    '''
    Huber-Loss: delta^2 * (sqrt(1 + (x/delta)^2)-1)
    #this is exactly like wikipedia pseudo huber loss
    '''
    x = preds - dtrain.get_label() 
    scale = 1 + (x / delta) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = x / scale_sqrt
    hess = np.abs(delta)**3 / scale / scale_sqrt
    return grad, hess

def pseudo_huber_loss_err_k(preds, dtrain, delta = 1):
    x = preds - dtrain.get_label() 
    scale = 1 + (x / delta) ** 2
    scale_sqrt = np.sqrt(scale)
    huber = delta**2 * (scale_sqrt -1)
    return ('pshe_delta'+str(delta)), np.mean(huber)



#my own weighted squared error function
def custom_relative_rmse(pred, dtrain):
    '''
    rrmse = 0.5 (pred-truth)^2 / truth
    '''
    truth = dtrain.get_label()
    
    grad = (pred - truth)/truth
    hess = 1./truth
    return grad, hess

def custom__relative_rmse_err(pred, dtrain = xgb.DMatrix):
    '''
    return a pair (name, result) name cannot contain : or space for each wanted evaluation error
    returns
    -----
    mae: mean absolute error
    rmse: mean squared error
    custom_relative_rmse_error: relative mean squared error normed to true energy
    in order to not undervalue low energy outliers
    '''
    truth = dtrain.get_label()
    rrmse = np.sqrt( np.mean(np.power(pred-truth, 2) / truth) )
    return 'rrmse_err', rrmse




