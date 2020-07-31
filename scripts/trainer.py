#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/icetray-start
#METAPROJECT /home/pfuerst/i3_software_py3/combo/build

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
from   sklearn.model_selection import train_test_split
import yaml
import xgboost as xgb

# -- custom imports --
#this is hardcoded and sad. Find another way that still works on condor 
full_path = "/home/pfuerst/master_thesis/software/combienergy"
sys.path.append(os.path.join(full_path))
import scripts.tools.loss_functions as func

def parse_arguments():
    """argument parser to specify configuration"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pandas_dataframe", type=str,
        default = "/data/user/pfuerst/Reco_Analysis/Simulated_Energies_Lists/feature_dataframes/features_dataframe_11029_11060_11070_withNaN_v2_coherent.pkl",
        help="dataframe created by extractor.py")
    parser.add_argument(
        "--feature_config", type = str,
        default = "L5_E.yaml",
        help= ".yaml containing a list of features to be used for training and prediction")
    parser.add_argument(
        "--label_key", type = str, default = "E_entry",
        help="key in dataframe to be used as label (has true E), previously E_truth_label which was in log10E!")
    parser.add_argument(
        "--modelname", type = str, default = 'combienergy',
        help = "custom name tag to be appended to the output model name.")
    
    # -- training hyperparameters -- 
    
    parser.add_argument(
        "--max_depth", type=int, default=6, help="max tree depth")
    parser.add_argument(
        "--learning_rate", type=float, default=0.03)
    parser.add_argument(
        "--early_stopping_rounds", type=int, default=10)
    parser.add_argument(
        "--subsample", type=float, default=0.8)
    parser.add_argument(
        "--gamma", type = float, default = 0.5)
    parser.add_argument(
        "--colsample_bytree", type = float, default = 1.0)
    parser.add_argument(
        "--min_child_weight", type = float, default = 10)
    parser.add_argument(
        "--num_rounds", type=int, default = 2500,
        help="number of boosting rounds")
    parser.add_argument(
        "--test_split_size", type = float, default = 0.1,
        help="percent of data used for testing, i.e. amount of data with predicted energies")
    parser.add_argument(
        "--objective", type = str, default = "rmse", #'pshedelta'
        help="objective function to use, rmse, pshe, pshedelta, rrmse, weightrmse are possible rn.")

    parser.add_argument(
        "--delta", type = float, default = 3,
        help = "for huber loss slope")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()  
    
    #pathname = os.path.dirname(sys.argv[0])    ###this ofc doesnt work running on condor machines  
    #full_path =  os.path.abspath(pathname)
    

    
    config_path = os.path.join(full_path, "config","files", args.feature_config)


    
    #build labels and training, validation and testing splits.
    full_dict = pd.read_pickle(args.pandas_dataframe)
    #remove zeros and NaNs from label array as these cant be handled by xgboost. 
    cut_dict = full_dict.replace(to_replace = {"E_entry": 0.0}, value = pd.np.nan).dropna()
    
    #####################################################
    y = cut_dict[args.label_key]

    y = np.log10(y) # E_entry is in true energy
    #this somehow needs to be done when predicting the energy outcome.
    #####################################################
    
    X_train, X_test, y_train, y_test = train_test_split(cut_dict, y, test_size = args.test_split_size, random_state=123)

    X_save = X_test

    #these are the IDs of the events that later get an energy prediction.

    valid_size = 0.2
    X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size = valid_size, random_state = 123)

    features = yaml.load(open(config_path,'r'), Loader = yaml.SafeLoader)
    for key in features:
        if key not in full_dict.keys():
            print("key not found in pickle file!")
            raise KeyError("trying to use a feature not contained in loaded data")
    
    drop_keys = []
    for key in full_dict.keys():
        if key not in features:
            drop_keys.append(key)
    feature_dict = full_dict.drop(columns = drop_keys)

    X_train = X_train.drop(columns = drop_keys)
    X_eval  = X_eval.drop(columns  = drop_keys)
    X_test  = X_test.drop(columns  = drop_keys)    

    training_datamatrix   = xgb.DMatrix(data = X_train, label = y_train)
    validation_datamatrix = xgb.DMatrix(data = X_eval,  label = y_eval )
    testing_datamatrix    = xgb.DMatrix(data = X_test,  label = y_test )

    param = {'booster':                'gbtree',
             #'verbosity':              0,
             'max_depth':              args.max_depth,
             'eta':                    args.learning_rate,
             #'eval_metric':            args.eval_metrices,
             'early_stopping_rounds':  args.early_stopping_rounds,
             'subsample':              args.subsample,
             'disable_default_eval_metric':1
            }
   
   
    watchlist = [(training_datamatrix, 'train'),(validation_datamatrix, 'eval')]

    evals_result = {}
    
    if args.objective == "rmse":
        obj   = func.rmse
        feval = func.rmse_err
        
    elif args.objective == "pshe":
        obj   = funcpseudo_huber_loss
        feval = func.pseudo_huber_loss_err
        
    elif args.objective == "pshedelta":
        obj   = func.pseudo_huber_loss_k
        feval = func.pseudo_huber_loss_err_k
        delta = args.delta #this is messy
        
    elif args.objective == "rrmse":
        obj   = func.custom_relative_rmse
        feval = func.custom__relative_rmse_err

    
    print("BDT trainer initialized with")
    print(param)
    print("used features:")
    print(training_datamatrix.feature_names)


    print("starting training of "+args.modelname+", max rounds: "+str(args.num_rounds)+" ... ")
    model = xgb.train(params = param, dtrain=training_datamatrix, num_boost_round = args.num_rounds, obj = obj, feval = feval, evals = watchlist,evals_result = evals_result)



    #save model
    mobj = str(args.objective)
    mobj = mobj.replace(":", "_")
    modelname = mobj+'_'+args.modelname+'_N'+str(args.num_rounds)+"_"+str(args.feature_config[:-5])+'.model'
    print("model saved as "+modelname)
    #model.save_model(os.path.join(full_path, "trained_models", modelname))
    #model.dump_model(os.path.join(full_path, "trained_models", modelname))
    
    #repeat after me: DO NOT USE XGBOOSTS INTERNAL .save_model OR .dump_model !!!
    #reason 1) .save_model forgets feature names and order and if they are messed up
    #          it doesnt care and predicts random numbers (e.g. it could think E_dnn is cog_rho or sth.
    #reason 2) loading models saved with .dump_model models instantly crashes ipython notebooks.
    
    pickle.dump(model, open(os.path.join(full_path, "trained_models", ("PICKLED_"+modelname)),"wb"))


    #save energy predictions. All testing/validation data get a NaN entry.
    ypred = model.predict(testing_datamatrix)


    X_save["E_predicted"]  = ypred
    savepath = "/data/user/pfuerst/Reco_Analysis/Simulated_Energies_Lists/feature_dataframes/withBDT/"

    X_save.to_pickle(savepath+"modeldata_"+modelname+'.pkl')
    print("prediction saved at "+savepath+"modeldata_"+modelname+'.pkl')

    plt.rcParams["figure.figsize"] = (13, 6)
    xgb.plot_importance(model)
    plt.savefig("/home/pfuerst/master_thesis/plots/BDT_validation/"+modelname+"_features.png")

    for key in evals_result["eval"].keys():
        training_rmse = evals_result["train"][key]
        validation_rmse = evals_result["eval"][key]

        plt.figure()
        plt.plot(training_rmse, label = "training")
        plt.plot(validation_rmse, label = "validation")
        plt.xlabel("# Boosting Runds")
        plt.ylabel(str(key))
        plt.ylim(0,0.4)
        plt.legend()
        plt.grid()
        plt.savefig("/home/pfuerst/master_thesis/plots/BDT_validation/"+modelname+str(key)+"_losscurve.png")

    print("plots saved at /home/pfuerst/master_thesis/plots/BDT_validation/")
