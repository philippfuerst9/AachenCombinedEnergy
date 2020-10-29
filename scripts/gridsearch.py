#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v3.1.1/icetray-start
#METAPROJECT /home/pfuerst/i3_software/combo/build

"""
Performs a grid search over hyperparameters to find the best values.
"""
# -- internal packages -- 
import argparse
import matplotlib.pyplot as plt
import numpy   as np
import os
import pickle
import sys

# -- external packages --
#e.g. pip install pyyaml --user
import pandas  as pd
import sklearn
from   sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
import yaml
import xgboost as xgb

# -- icecube software --
from icecube.weighting.weighting import from_simprod, NeutrinoGenerator

# -- custom imports --
full_path = "/home/pfuerst/master_thesis/software/combienergy"
sys.path.append(os.path.join(full_path))
import scripts.tools.loss_functions as func
from scripts.trainer import one_weight_builder_2019

early_stopping_rounds = 10
fixed_params = {'objective':              'reg:pseudohubererror',
                'booster':                'gbtree',
                'n_estimators':           30
               }
#bfore: reg:linear
#mphe
#rmse

grid_params = {
    'max_depth':         [6],
    'eta':               [0.03],
    'subsample':         [0.8,1.0],
    'gamma':             [0.4,0.5,0.6],
    'min_child_weight':  [8,10,12],
    'colsample_bytree':  [1.0,1.1]
              }
#best params for both pseudohuberloss and rmse: {'colsample_bytree': 1.0, 'min_child_weight': 10, 'subsample': 0.8, 'eta': 0.03, 'max_depth': 6, 'gamma': 0.5}

####
full_dict = pd.read_pickle("/data/user/pfuerst/Reco_Analysis/Simulated_Energies_Lists/feature_dataframes/2019_frames/21217+21220/combined_training_set.pickle")

#build labels and training, validation and testing splits.

cut_dict = full_dict.dropna() #removes all nan entries as nan in label cannot be used.
y = cut_dict["E_entry"]

weights = one_weight_builder_2019(cut_dict["MCPrimaryEnergy"], cut_dict["MCPrimaryType"], cut_dict["MCPrimaryCosZen"], cut_dict["TIntProbW"])   
cut_dict.insert(5,"generator_weights", weights)

X_train, X_test, y_train, y_test = train_test_split(cut_dict, y, test_size = 0.4, random_state=123)

#these are the IDs of the events that later get an energy prediction.
#test_event_ids = X_test["event_id"]

valid_size = 0.2
X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size = valid_size, random_state = 123)


#isolate weights before cleaning dataframes
w_train = X_train["generator_weights"]
w_eval  = X_eval["generator_weights"]
w_test  = X_test["generator_weights"]

#load feature config

config_path = os.path.join(full_path, "config","files", "L5_E_no_sigmapar.yaml")
features = yaml.load(open(config_path,'r'), Loader = yaml.SafeLoader)
for key in features:
    if key not in full_dict.keys():
        print("key not found in pickle file!")
        raise KeyError("trying to use a feature not contained in loaded data!")

#only keep features from feature config in xgboost datamatrices
drop_keys = []
for key in cut_dict.keys():
    if key not in features:
        drop_keys.append(key)

X_train = X_train.drop(columns = drop_keys)
X_eval  = X_eval.drop(columns  = drop_keys)
X_test  = X_test.drop(columns  = drop_keys)  

#training_datamatrix   = xgb.DMatrix(data = X_train, label = y_train)
#validation_datamatrix = xgb.DMatrix(data = X_eval,  label = y_eval )
#testing_datamatrix    = xgb.DMatrix(data = X_test,  label = y_test )#

#training_datamatrix.set_weight(w_train)
#validation_datamatrix.set_weight(w_eval)
#testing_datamatrix.set_weight(w_test)

folds = 5
xgb_regress = xgb.XGBRegressor(params = fixed_params, nthread=1, early_stopping_rounds = early_stopping_rounds)
skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1009)
grid_search = GridSearchCV(xgb_regress, param_grid=grid_params,
                                   n_jobs=-1,
                                   verbose=5)

#grid_search.fit(X=X_train, y=y_train)
grid_search.fit(X=X_train, y=y_train)

grid_search_best_params = np.array(grid_search.best_params_)
np.save('/home/pfuerst/master_thesis/software/grid_search_best_params_2019_test', grid_search_best_params)
