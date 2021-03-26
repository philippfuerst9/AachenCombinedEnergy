# run in py2 venv

# imports and plotting setup
import os
import sys

import numpy as np
import scipy
from scipy import stats
import pandas as pd
import imp
import argparse
import pickle
import yaml
from NNMFit import nnm_logger
from NNMFit.analysis.nnm_fitter import NNMFitter

class Tensor_From_HDF(object):
    ''' Initializes the Theano Tensor. Fit variables can be set manually
    And corresponding weights can be calculated.
    '''
    def __init__(self, args):
        self.args = args

        # setup fitter object
        with open(self.args.analysis_config) as hdl:
            self.analysis_config = yaml.load(hdl, Loader=yaml.FullLoader)
        self.fitter = NNMFitter(self.args.main_config,
                       self.args.override_configs,
                       analysis_config=self.analysis_config)

        # get graph handler
        self.graph_hdl = self.fitter.graph_builder

        # set "use_binned_flux" to False to get per event weights 
        self.graph_hdl.use_binned_flux = False
        print("graph_hdl.use_binned_flux set to", self.graph_hdl.use_binned_flux)

        # set "use_binned_sys" to False to jump into the right loop
        self.graph_hdl.use_binned_sys = False
        print("graph_hdl.use_binned_sys set to", self.graph_hdl.use_binned_sys)

        self.det_sys_handler = self.graph_hdl.make_systematics_handler()
        self.all_graphs, self.all_input_vars, self.all_graphs_unbinned, self.all_indices, self.graphs_per_flux = self.graph_hdl.make_expectation_graph_weights(binned_expectation=True, return_per_flux=True)
        self.det_conf = "IC86_pass2"
        # get the theano tensor 
        self.tensor = self.all_graphs_unbinned[self.det_conf]

    def set_input_variables(self, input_variables):
        """Takes dict of input variables like dom_eff, conv_norm
        and sets the tensor variables.
        """
        for self.var in self.all_input_vars:
            if self.var.name in input_variables.keys():
                val = input_variables[self.var.name]
                print("Setting ", self.var.name, val)
                self.var.set_value(val)
        self.input_var_dict = input_variables
        
    def get_variables(self, variables_out, input_variables):
        """ Gets the L5 variables (or whatever is wanted), setting the input variables beforehand.
        """

        self.set_input_variables(input_variables)

        self.variables_to_dump = self.graph_hdl.collect_required_variables(self.det_conf, self.det_sys_handler)
        print(self.variables_to_dump)

        self.variables_to_dump = self.variables_to_dump.union(set(variables_out))
        print("after set")
        print(self.variables_to_dump)
        self.shared_vars, self.bin_indices, _ = self.graph_hdl.make_shared_variables(self.variables_to_dump, self.det_conf)
        
        out = {}
        out["input_variables"] = self.input_var_dict
        for variable in self.variables_to_dump:
            out[variable] = self.shared_vars[variable].eval()
            
        out["weights"] = self.tensor.eval()
        return out
    
    def get_weights_only(self, input_variables):
        """ gets event weights after setting input variables. Saves memory
        """
        self.set_input_variables(input_variables)
        out = {}
        out["input_variables"] = self.input_var_dict
        out["weights"] = self.tensor.eval()
        return out
    
# main
parser = argparse.ArgumentParser()
parser.add_argument("main_config")
parser.add_argument("--analysis_config", help="Config containing the analysis"\
                    "config", dest="analysis_config", required=True)
parser.add_argument("-o", "--outpath", help="Output path",
                    dest="outpath", required=True)
parser.add_argument("-b", "--bestfit", help="Pickle file with best-fit result",
                    dest="bestfit_file", required=True)
parser.add_argument("--override_configs", help="Override configs",
                    nargs="+", dest="override_configs", required=True)

args = parser.parse_args()
    

tensor = Tensor_From_HDF(args)

#load the bestfit file
with open(args.bestfit_file,"r") as bf_file:
    bestfit = pickle.load(bf_file)#["fit-result"][1]

fit_results = bestfit["fit-result"][1]
fixed_vars  = bestfit['fixed-parameters']
fit_results.update(fixed_vars)

# set fitted vars
tensor.set_input_variables(fit_results)

# additional variables to dump
additional_variables_to_dump = ["reco_energy", "reco_zenith", "true_energy",
                     "true_zenith", "event_id", "L5_cog_rho", "L5_cog_z", 
                    "L5_lseparation", "L5_nch", "L5_bayes_llh_diff",
                    "L5_rlogl", "L5_ldir_c", "L5_ndir_c", "L5_sigma_paraboloid", 
                               "L5_sdir_e", "non_reco_ace_energy", "non_reco_dnn_energy", "energy_muex"] 

tensor_vars = tensor.get_variables(additional_variables_to_dump, fit_results)

print("Output looks like this:")
for key, var in tensor_vars.items():
    print key

override_name = os.path.split(args.override_configs[0])[1]
outfile_MC = args.outpath+"/MC_wBestfit_"+override_name+"_from_v1_.pickle"  
with open(outfile_MC, "wc") as hdl:
    pickle.dump(tensor_vars, hdl, protocol=pickle.HIGHEST_PROTOCOL)
    print("Wrote MC events with weights to", outfile_MC) 

# take care of data
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
    