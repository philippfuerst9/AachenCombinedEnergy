#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/icetray-start
#METAPROJECT /home/pfuerst/i3_software_py3/combo/build

"""
Trains an xgboost booster model with a custom configuration
on feature data from extractor.py. 
Saves the model and Loss curve + Feature map plots in hardcoded directories.
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
from   sklearn.model_selection import train_test_split
import yaml
import xgboost as xgb

# -- icecube software --
from icecube.weighting.weighting import from_simprod, NeutrinoGenerator

# -- custom imports --
full_path = "/home/pfuerst/master_thesis/software/combienergy"
sys.path.append(os.path.join(full_path))
import scripts.tools.loss_functions as func


def parse_arguments():
    """argument parser to specify configuration"""
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pandas_dataframe", type=str,
        #default = "/data/user/pfuerst/Reco_Analysis/Simulated_Energies_Lists/feature_dataframes/2012_frames/full/full2012_wXY_wGEO.pickle",
        default = "/data/user/pfuerst/Reco_Analysis/Simulated_Energies_Lists/feature_dataframes/2019_frames/21217+21220/combined_training_set.pickle",
        help="dataframe created by extractor.py")
    parser.add_argument("--sim_year", required = True,
                       type = str,
                       help="either 2012 or 2019. implement new weighting function for your own dataset!")
    parser.add_argument(
        "--feature_config", type = str,
        default = "L5_E_no_sigmapar.yaml",
        help= ".yaml containing a list of features to be used for training and prediction")
    parser.add_argument(
        "--label_key", type = str, default = "E_entry",
        help="key in dataframe to be used as label (has true E), previously E_truth_label which was in log10E!")
    parser.add_argument(
        "--modelname", type = str, default = 'my_combienergy_model',
        help = "custom name tag to be appended to the output model name.")
    parser.add_argument(
        "--gpu", action = 'store_true',
        help = "set this flag if you train on cluster and want gpus (x10 faster). Request gpus = 1 in submit script.")
    
    # -- training hyperparameters -- 
    
    parser.add_argument(
        "--max_depth", type=int, default=6, help="max tree depth")
    parser.add_argument(
        "--learning_rate", type=float, default=0.03)
    parser.add_argument(
        "--early_stopping_rounds", type=int, default=1000)
    parser.add_argument(
        "--subsample", type=float, default=0.8)
    parser.add_argument(
        "--gamma", type = float, default = 0.5)
    parser.add_argument(
        "--colsample_bytree", type = float, default = 1.0)
    parser.add_argument(
        "--min_child_weight", type = float, default = 100)
    parser.add_argument(
        "--num_rounds", type=int, default = 2500,
        help="number of boosting rounds")
    parser.add_argument(
        "--test_split_size", type = float, default = 0.1,
        help="percent of data used for quick model testing.")
    parser.add_argument(
        "--objective", type = str, default = "rmse", #'pshedelta'
        help="objective function to use, rmse, pshe, pshedelta, rrmse, weightrmse are implemented.")

    parser.add_argument(
        "--delta", type = float, default = 3,
        help = "for huber loss slope if objective pshedelta is set.")
    
    parser.add_argument(
        "--no_weights", action="store_true",
        help="Turn off OneWeights x 1e-18 as event weights. If not set, make sure they are stored in the pandas_dataframe.")
    args = parser.parse_args()
    return args

def one_weight_builder_2012(prime_E, prime_Type, prime_coszen, total_weight, 
                       ds_nums = [11029, 11069, 11070],
                       ds_nfiles = [3190, 3920, 997]):      
    """builds OneWeights for training on the combined sim sets 11029, 11069 and 11070.
    the generator is basically the #events per energy range
    prime_E: ["MCPrimary1"].energy
    prime_Type: ["MCPrimary1"].type
    prime_coszen: cos(["MCPrimary1"].dir.zenith)
    total_weight: ["I3MCWeightDict"]["TotalInteractionProbabilityWeight"]
    returns the OneWeight/E for this specific event, i.e. weighted with an 
    E**-1 flux for constant weight in log bins
    """
    
    generator_sum = np.sum([from_simprod(ds_num) * ds_nfiles[i] for i, ds_num in enumerate(ds_nums)])
    return total_weight / (generator_sum(prime_E, particle_type = prime_Type, cos_theta = prime_coszen)*prime_E)

def one_weight_builder_2019(prime_E, prime_Type, prime_coszen, total_weight): 
    """builds OneWeights for training on the combined sim sets 21217, 21220.
    the generator is basically the #events per energy range
    prime_E: ["MCPrimary1"].energy
    prime_Type: ["MCPrimary1"].type
    prime_coszen: cos(["MCPrimary1"].dir.zenith)
    total_weight: ["I3MCWeightDict"]["TotalInteractionProbabilityWeight"]
    returns the OneWeight/E for this specific event, i.e. weighted with an 
    E**-1 flux for constant weight in log bins
    """
    generator_21217 = np.sum([NeutrinoGenerator(10000, 100, 1e8, 1.5, "NuMu")]) * 21674
    generator_21220 = np.sum([NeutrinoGenerator(250, 100, 1e8, 1., "NuMu")]) * 5179
    
    generator_sum = generator_21217+generator_21220
    return total_weight / (generator_sum(prime_E, particle_type = prime_Type, cos_theta = prime_coszen)*prime_E)

def cleaner(pandasframe):
    """removes edge cases where calculation of MC truth fails 
    (E_entry = 0.0 and NaN)
    """
    
    cleanframe = pandasframe.replace(to_replace = {"E_entry": 0.0}, value = pd.np.nan).dropna()
    return cleanframe

if __name__ == '__main__':
    
    args = parse_arguments()  
    config_path = os.path.join(full_path, "config","files", args.feature_config)
    full_dict = pd.read_pickle(args.pandas_dataframe)
    
    cut_dict = cleaner(full_dict)
    y = cut_dict[args.label_key]
    y = np.log10(y) # E_entry is in true energy

    #build weights
    if args.sim_year == "2012":
        print("2012 weighting: assuming training on 11029, 11069, 11070 sets.")
        weights = one_weight_builder_2012(cut_dict["MCPrimaryEnergy"], cut_dict["MCPrimaryType"], 
                                     cut_dict["MCPrimaryCosZen"], cut_dict["TIntProbW"])
    elif args.sim_year == "2019":
        print("2019 weighting: assuming training on 21217, 21220 sets.")
        weights = one_weight_builder_2019(cut_dict["MCPrimaryEnergy"], cut_dict["MCPrimaryType"], 
                                     cut_dict["MCPrimaryCosZen"], cut_dict["TIntProbW"])    
    else:
        raise ValueError("Invalid weighting option! Must be 2012 or 2019.")

    cut_dict.insert(5,"generator_weights", weights)
    
    #split test set which will be saved (with prediction) as dataframe
    #set random_state=123 for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(cut_dict, y, test_size = args.test_split_size) 
    X_save = X_test

    #split evaluation set to watch during training
    valid_size = 0.2
    X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size = valid_size)

    #isolate weights before cleaning dataframes
    w_train = X_train["generator_weights"]
    w_eval  = X_eval["generator_weights"]
    w_test  = X_test["generator_weights"]
        
    #load feature config
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

    training_datamatrix   = xgb.DMatrix(data = X_train, label = y_train)
    validation_datamatrix = xgb.DMatrix(data = X_eval,  label = y_eval )
    testing_datamatrix    = xgb.DMatrix(data = X_test,  label = y_test )

    #handle weighting flag
    if args.no_weights == False:
        training_datamatrix.set_weight(w_train)
        validation_datamatrix.set_weight(w_eval)
        testing_datamatrix.set_weight(w_test)
        print("BDT training using generated OneWeights as weights.")
        
    elif args.no_weights == True:
        print("BDT training on unweighted simulation data.")
    
    #build BDT parameter space
    param = {'booster':                'gbtree',
             'verbosity':              3,
             'max_depth':              args.max_depth,
             'eta':                    args.learning_rate,
             #'eval_metric':            args.eval_metrices,
             'subsample':              args.subsample,
             'disable_default_eval_metric':1
            }
   
    if args.gpu == True:
        param['tree_method'] = 'gpu_hist'
        
    watchlist = [(training_datamatrix, 'train'),(validation_datamatrix, 'eval')]

    evals_result = {}
    
    #pick custom loss function
    if args.objective == "rmse":
        obj   = func.rmse
        feval = func.rmse_err
        
    elif args.objective == "pshe":
        obj   = func.pseudo_huber_loss
        feval = func.pseudo_huber_loss_err
        
    elif args.objective == "pshedelta":
        obj   = func.pseudo_huber_loss_k
        feval = func.pseudo_huber_loss_err_k
        delta = args.delta 
        
    elif args.objective == "rrmse":
        obj   = func.custom_relative_rmse
        feval = func.custom__relative_rmse_err

    
    print("BDT trainer initialized with")
    print(param)
    print("used features:")
    print(training_datamatrix.feature_names)
    print("starting training of "+args.modelname+", max rounds: "+str(args.num_rounds)+" ... ")
    
    #cleanup
    del full_dict
    del cut_dict
    del weights
    
    #train booster
    model = xgb.train(params = param, dtrain=training_datamatrix, num_boost_round = args.num_rounds,
                      early_stopping_rounds = args.early_stopping_rounds,
                      obj = obj, feval = feval, evals = watchlist,evals_result = evals_result)
    
    #save model
    mobj = str(args.objective)
    mobj = mobj.replace(":", "_")
    modelname = "PICKLE_"+mobj+'_'+args.modelname+'_N'+str(args.num_rounds)+"_"+str(args.feature_config[:-5])+'_'+str(args.test_split_size)+'.model'
    print("model saved as "+modelname)
    
    #######################################################################################################
    #                                                                                                     #
    # repeat after me: DO NOT USE XGBOOSTS INTERNAL .save_model OR .dump_model !!!                        #
    # reason 1) .save_model forgets feature names and order and if they are messed up                     #
    #           it doesnt care and predicts random numbers (e.g. it could think E_dnn is cog_rho or sth.  #
    # reason 2) loading models saved with .dump_model models instantly crashes ipython notebooks.         #
    #                                                                                                     #
    #######################################################################################################
    
    pickle.dump(model, open(os.path.join(full_path, "trained_models", modelname),"wb"))

    #save energy predictions.
    ypred = model.predict(testing_datamatrix)
    X_save["E_predicted"]  = ypred
    savepath = "/data/user/pfuerst/Reco_Analysis/Simulated_Energies_Lists/feature_dataframes/withBDT/"
    X_save.to_pickle(savepath+"modeldata_"+modelname+'.pickle')
    print("prediction saved at "+savepath+"modeldata_"+modelname+'.pickle')

    #plots to document training/evauluation loss and feature map
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
        #plt.ylim(0,0.4)
        plt.legend()
        plt.grid()
        plt.savefig("/home/pfuerst/master_thesis/plots/BDT_validation/"+modelname+str(key)+"_losscurve.png")

    print("plots saved at /home/pfuerst/master_thesis/plots/BDT_validation/")
