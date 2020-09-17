#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/icetray-start
#METAPROJECT /home/pfuerst/i3_software_py3/combo/build

"""
Builds keys defined in feature_extractor from i3 files into pandas dataframe(s)
"""
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

# -- icecube software --
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
                       help = "path do directory containing i3 files.")
    group.add_argument("--filenamelist", type = str, nargs="+",
                       help= "list of filenames ending with .i3.zst.")

    #/home/pfuerst/master_thesis/software/BDT_energy_reconstruction/config/files/
    parser.add_argument(
        "--write", type = str, required = True,
        help= "path+name of the produced full .pickle file at /data/user/pfuerst/Reco_Analysis/Simulated_Energies_Lists/feature_dataframes/")
    
    parser.add_argument(
        "--wACE", action = "store_true",
        help="flag to extract predicted energies. Set this only if prediction keys already exist.")

    args = parser.parse_args()
    return args

def feature_extractor(frame, wACE=False):
    """reads feature keys from frame into a dictionary
    
    Energies are in log10[GeV], except for energy at entry/exit as exit can be 0
    returns all data necessary for correctly weighting events and training the BDT. 
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
    "E_truncated"         : np.NaN,
    "E_muex"              : np.NaN,
    "E_dnn"               : frame["TUM_dnn_energy_hive"]["mu_E_on_entry"],
    "random_variable"     : np.random.random()*10,
    "E_entry"             : frame["TrueMuoneEnergyAtDetectorEntry"].value,   #e_entry
    "E_exit"              : frame["TrueMuoneEnergyAtDetectorLeave"].value,   #e_exit
    "OneWeight"           : frame["I3MCWeightDict"]["OneWeight"],
    "NEvent"              : frame["I3MCWeightDict"]["NEvents"],
    "TIntProbW"           : frame["I3MCWeightDict"]["TotalWeight"],          #TotalInteractionProbabilityWeight for 2012
    "MCPrimaryEnergy"     : frame["MCPrimary1"].energy,
    "MCPrimaryType"       : frame["MCPrimary1"].type,
    "MCPrimaryCosZen"     : np.cos(frame["MCPrimary1"].dir.zenith),
    "cog_x"               : frame["HitStatisticsValuesIC"].cog.x,
    "cog_y"               : frame["HitStatisticsValuesIC"].cog.y,  
    } 
    
    if wACE == True:
        features["E_predicted"] = frame["ACEnergy_Prediction"].value
                                 
    try:
        features["E_truncated"]   = np.log10(frame["SplineMPEICTruncatedEnergySPICEMie_AllDOMS_Muon"].energy)
    except:
        pass
    
    try:
        features["E_muex"] = np.log10(frame["SplineMPEICMuEXDifferential"].energy)
    except:
        pass
   
    return features

def extract_from_pathlist(pathlist):
    list_of_featuredicts = []
    
    for path in pathlist:
        print("--- folder {} ---".format(path))
        for filename in os.listdir(path):
            if filename.endswith(".i3.zst"):
                print("processing file {}".format(filename))
                with dataio.I3File(os.path.join(path,filename)) as f:
                    for currentframe in f:        
                        if str(currentframe.Stop) == "Physics":
                            featuredict = feature_extractor(currentframe, wACE = wACE_bool)
                            list_of_featuredicts.append(featuredict)
    return list_of_featuredicts

def extract_from_filenamelist(filenamelist):
    list_of_featuredicts = []
    for filename in filenamelist:
        print("processing file {}".format(filename))
        with dataio.I3File(filename) as f:
            for currentframe in f:        
                if str(currentframe.Stop) == "Physics":
                    featuredict = feature_extractor(currentframe, wACE = wACE_bool)
                    list_of_featuredicts.append(featuredict)
    return list_of_featuredicts
                            
if __name__ == '__main__':
    
    args = parse_arguments()    
    wACE_bool = args.wACE
    name = args.write
    
    if not name.endswith(".pickle") or name.endswith(".pckl") or name.endswith(".pkl"):
        name+=".pickle"
    
    if args.pathlist_config is not None:
        config_path = os.path.join(full_path, "config","files", args.pathlist_config)
        print(config_path)
        pathlist = yaml.load(open(config_path,'r'), Loader = yaml.SafeLoader)
        list_of_featuredicts = extract_from_pathlist(pathlist)
        
    if args.pathlist is not None:
        pathlist = args.pathlist
        list_of_featuredicts = extract_from_pathlist(pathlist)

    if args.filenamelist is not None:
        filenamelist = args.filenamelist
        list_of_featuredicts = extract_from_filenamelist(filenamelist)

    pandasframe = pd.DataFrame(data = list_of_featuredicts)
    pandasframe.to_pickle(name)