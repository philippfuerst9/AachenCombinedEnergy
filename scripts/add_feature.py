#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/icetray-start
#METAPROJECT /home/pfuerst/i3_software_py3/combo/build


# -- internal packages -- 
import argparse
import math
import numpy   as np
import os
import pickle
import sys
import time

# -- external packages --
#e.g. pip install pyyaml --user
import pandas  as pd
import sklearn #__version__ 0.23.1
from   sklearn.model_selection import train_test_split
import yaml
import xgboost as xgb #__version__ 1.1.1

# -- icetray --
from icecube import dataio, dataclasses, icetray, common_variables, paraboloid
from icecube.icetray import I3Units
from I3Tray import *

# -- custom imports --
#this is hardcoded and will change once it all is a package.
full_path = "/home/pfuerst/master_thesis/software/combienergy"
sys.path.append(os.path.join(full_path))
import scripts.tools.segmented_muon_energy as sme


def parse_arguments():
    """argument parser to specify configuration"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pickle", type = str,
        required = True,
        help= "path+name of the produced full pickle file")

    group = parser.add_mutually_exclusive_group(required = True)

    group.add_argument(
        "--pathlist_config", type=str,
        #default = 'i3_pathlist_v2.yaml',
        help="config .yaml containing python list of paths to i3 files")
    group.add_argument("--pathlist", type = str, nargs="+",
                       help = "if no config .yaml is built, you can also just supply list of filenames.")
    
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_arguments()
    standard_pandas_frame = pd.read_pickle(args.pickle)
    
    #savepath = "/data/user/pfuerst/Reco_Analysis/Simulated_Energies_Lists/feature_dataframes/"
    #standard_pandas_name = 'features_dataframe_11029_11060_11070_withNaN.pkl'

    
    if args.pathlist_config is not None:
        config_path = os.path.join(full_path, "config","files", args.pathlist_config)
        print(config_path)
        pathlist = yaml.load(open(config_path,'r'), Loader = yaml.SafeLoader)
    
    if args.pathlist is not None:
        pathlist = args.pathlist
    
    print(pathlist)
    
    #add prediction to dataframe
    '''
    new_feature_name = "E_predicted"
    new_feature_list = [] 
    list_of_featuredicts = []
    for path in pathlist:
        
        print("--- folder {} ---".format(path))
        for filename in os.listdir(path):
            if filename.endswith(".i3.zst"):
                print("processing file {}".format(filename))
                with dataio.I3File(os.path.join(path,filename)) as f:
                    for currentframe in f:        
                        if str(currentframe.Stop) == "Physics":
                            new_feature = currentframe["ACEnergy_Prediction"].value    
                            new_feature_list.append(new_feature)

    new_feature_dict = {}
    new_feature_dict[new_feature_name] = new_feature_list

    standard_pandas_frame.update(new_feature_dict)
    standard_pandas_frame.to_pickle(args.pickle)  
    '''
    


#append new custom features

    

    TIntProbW_list = []
    MCPrimaryEnergy_list = []
    MCPrimaryType_list = []
    cos_primary_zen_list = []
    
    for path in pathlist:
        
        print("--- folder {} ---".format(path))
        for filename in os.listdir(path):
            if filename.endswith(".i3.zst"):
                print("processing file {}".format(filename))
                with dataio.I3File(os.path.join(path,filename)) as f:
                    for currentframe in f:        
                        if str(currentframe.Stop) == "Physics":
                            TIntProbW = currentframe["I3MCWeightDict"]["TotalInteractionProbabilityWeight"]
                            MCPrimaryEnergy = currentframe["MCPrimary1"].energy,
                            MCPrimaryType = currentframe["MCPrimary1"].type, 
                            cos_primary_zen = np.cos(currentframe["MCPrimary1"].dir.zenith)
                            
                            TIntProbW_list.append(TIntProbW)
                            MCPrimaryEnergy_list.append(MCPrimaryEnergy[0])
                            MCPrimaryType_list.append(MCPrimaryType[0])
                            cos_primary_zen_list.append(cos_primary_zen)



    new_feature_dict = {}
    new_feature_dict["TIntProbW"] = TIntProbW_list
    new_feature_dict["MCPrimaryEnergy"] = MCPrimaryEnergy_list
    new_feature_dict["MCPrimaryType"] = MCPrimaryType_list
    new_feature_dict["MCPrimaryZenith"] = cos_primary_zen_list
    
    standard_pandas_frame.update(new_feature_dict)
    print("keys in updated file:")
    print(standard_pandas_frame.keys())
    standard_pandas_frame.to_pickle(args.pickle)
    print("updated file saved at {}".format(args.pickle))




