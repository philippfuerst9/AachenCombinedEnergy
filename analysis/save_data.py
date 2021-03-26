# run in py2 venv

# imports and plotting setup
import os
import sys

import scipy
from scipy import stats
import argparse
import cPickle as pickle
from NNMFit import nnm_logger
from NNMFit.analysis.nnm_fitter import NNMFitter
import yaml
import time
from NNMFit.analysis_config import AnalysisConfig
from NNMFit import likelihoods
from NNMFit import minimizer
from NNMFit import graph_builder
from NNMFit import loaders
import numpy as np
import pandas as pd

# main
parser = argparse.ArgumentParser()
parser.add_argument("main_config")
parser.add_argument("--analysis_config", help="Config containing the analysis"\
                    "config", dest="analysis_config", required=True)
parser.add_argument("-o", "--outpath", help="Output path",
                    dest="outpath", required=True)
parser.add_argument("--override_configs", help="Override configs",
                    nargs="+", dest="override_configs", required=True)

args = parser.parse_args()
    

# additional variables to dump
additional_variables_to_dump = ["reco_energy", "reco_zenith", "true_energy",
                     "true_zenith", "event_id", "L5_cog_rho", "L5_cog_z", 
                    "L5_lseparation", "L5_nch", "L5_bayes_llh_diff",
                    "L5_rlogl", "L5_ldir_c", "L5_ndir_c", "L5_sigma_paraboloid", 
                               "L5_sdir_e", "non_reco_ace_energy", "non_reco_dnn_energy", "energy_muex"] 

override_name = os.path.split(args.override_configs[0])[1]

with open(args.analysis_config) as hdl:
    analysis_config = yaml.load(hdl)

config_hdl = AnalysisConfig(args.main_config,
                            analysis_config["detector_configs"],
                            args.override_configs,
                            None)
detector_configs = analysis_config["detector_configs"]
if len(detector_configs)>1:
    print("Provide analysis settings with a single det-config")
    raise NotImplementedError
det_conf = detector_configs[0]

key_mapping, _  = config_hdl.get_key_mapping(det_conf)
dataset_obj, bins = loaders.load_data(config_hdl, det_conf)
dataset = dataset_obj._data_dict

outfile_data = args.outpath+"/Data_"+det_conf+".pickle"
with open(outfile_data, "w") as hdl:
    pickle.dump(dataset, hdl)
    print("Wrote Data events to", outfile_data)
    