#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/icetray-start
#METAPROJECT /home/pfuerst/i3_software_py3/combo/build

from icecube import dataio, dataclasses, icetray, common_variables, paraboloid
from icecube.icetray import I3Units
import numpy as np
import math
import os
import sys
import imp
import argparse
import pandas as pd
full_path = "/home/pfuerst/master_thesis/software/combienergy"
sys.path.append(os.path.join(full_path))
import scripts.tools.segmented_muon_energy as sme
#import scripts.extractor as ex
#this is how it should be done if new features are added to the Pandas DataFrame later

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
    new_feature_name = "E_predicted"
    new_feature_list = []
    
    if args.pathlist_config is not None:
        config_path = os.path.join(full_path, "config","files", args.pathlist_config)
        print(config_path)
        pathlist = yaml.load(open(config_path,'r'), Loader = yaml.SafeLoader)
    
    if args.pathlist is not None:
        pathlist = args.pathlist
    
    print(pathlist)
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

#append new custom features







