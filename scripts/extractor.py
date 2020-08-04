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
import scripts.tools.loss_functions as func
import scripts.tools.segmented_muon_energy as sme


#this program loads all i3 files from the folders supplied by the config file and builds one big pandas dataframe.
def parse_arguments():
    """argument parser to specify configuration"""
    parser = argparse.ArgumentParser()
    
    group = parser.add_mutually_exclusive_group(required = True)

    group.add_argument(
        "--pathlist_config", type=str,
        #default = 'i3_pathlist_v2.yaml',
        help="config .yaml containing python list of paths to i3 files")
    group.add_argument("--pathlist", type = str, nargs="+",
                       help = "if no config .yaml is built, you can also just supply list of filenames.")

    #/home/pfuerst/master_thesis/software/BDT_energy_reconstruction/config/files/
    parser.add_argument(
        "--name_out_pckl", type = str,
        default = '/data/user/pfuerst/Reco_Analysis/Simulated_Energies_Lists/feature_dataframes/features_dataframe_11029_11060_11070_withNaN_v2_coherent.pkl',
        help= "path+name of the produced full pickle file")
    args = parser.parse_args()
    return args

def feature_extractor(frame):
    """reads feature keys from frame into a dictionary
    
    Energies are in log10[GeV], except for energy entry/exit as exit can be 0
    """
    #if true e key does not exist do this
    #e_entry = np.NaN
    #e_exit  = np.NaN
    #try:
    #    e_entry, e_exit = sme.EnergyAtEgdeNoMuonGun(frame)

    #except:
    #    pass
    
    features = {
    "cog_rho"             : frame["L5_cog_rho"].value,
    "cog_z"               : frame["L5_cog_z"].value,
    "lseparation"         : frame["L5_lseparation"].value,
    "nch"                 : frame["L5_nch"].value,
    "bayes_llh_diff"      : frame["L5_bayes_llh_diff"].value,
    "cos_zenith"          : frame["L5_cos_zenith"].value,
    "rlogl"               : frame["L5_rlogl"].value,
    "ldir_c"              : frame["L5_ldir_c"].value,
    "ndir_c"              : frame["L5_ndir_c"].value,
    "sigma_paraboloid"    : frame["L5_sigma_paraboloid"].value,
    "sdir_e"              : frame["L5_sdir_e"].value,
    #"n_string_hits"       : frame["HitMultiplicityValuesIC"].n_hit_strings,
    "E_truncated"         : np.NaN,
    "E_muex"              : np.NaN,
    "E_dnn"               : frame["TUM_dnn_energy_hive"]["mu_E_on_entry"],  #different pulsemap as in training
    #"E_dnn"               : frame["TUM_dnn_energy"]["mu_E_on_entry"],        
    "random_variable"     : np.random.random()*10,
    "E_entry"             : frame["TrueMuoneEnergyAtDetectorEntry"].value,   #e_entry
    "E_exit"              : frame["TrueMuoneEnergyAtDetectorLeave"].value,    #e_exit
    #comment this for i3 files w/o prediction.
    "E_predicted"         : frame["ACEnergy_Prediction"].value 
    } 
    
    try:
        features["E_truncated"]   = np.log10(frame["SplineMPEICTruncatedEnergySPICEMie_AllDOMS_Muon"].energy)
    except:
        pass
    
    try:
        features["E_muex"] = np.log10(frame["SplineMPEICMuEXDifferential"].energy)
    except:
        pass
   
    return features

if __name__ == '__main__':
    
    args = parse_arguments()    
    #pathname = os.path.dirname(sys.argv[0])     
    #full_path =  os.path.abspath(pathname)
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
                            featuredict = feature_extractor(currentframe)
                            list_of_featuredicts.append(featuredict)


    pandasframe = pd.DataFrame(data = list_of_featuredicts)
    pandasframe.to_pickle(args.name_out_pckl)
