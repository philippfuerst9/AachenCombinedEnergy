#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/icetray-start
#METAPROJECT /home/pfuerst/i3_software_py3/combo/build

"""This program takes a trained model and a directory containing i3 files.
It adds a model prediction and a true energy key to these files and saves them in another directory.
By default also adds an exponated version of TUM DNN energy in GeV.
"""

# -- internal packages -- 
import argparse
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
from icecube import icetray, dataclasses, dataio
from I3Tray import *

# -- custom imports --
#this is hardcoded and will change once it all is a package.
full_path = "/home/pfuerst/master_thesis/software/combienergy"
sys.path.append(os.path.join(full_path))
import scripts.tools.loss_functions as func
import scripts.tools.segmented_muon_energy as sme

def parse_arguments():
    """argument parser to specify configuration"""
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model",
                        type=str,
                        default = "PICKLE_pshedelta_winner_N5000_L5_E_no_sigmapar_0.1.model",
                        help="trained xgboost model used for prediction.")

    parser.add_argument("--infiles", "-i", type = str,
                        nargs="+",
                        help = "Input /path/to/file.")
    
    parser.add_argument("--outfile", "-o",
                        required = "--infile" in sys.argv, type = str,
                        help = "/path/to/name_of_out_file. Directories will be created if necessary.")
    
    parser.add_argument("--no_prediction",
                       action="store_false",
                       help="flag to not write prediction keys")
    parser.add_argument("--no_truth",
                       action="store_false",
                       help="flag to not write truth keys")
    #parser.add_argument("--do_expDNN",
    #                   action="store_true",
    #                   help="flag to write exponated DNN keys")
    #/data/ana/Diffuse/AachenUpgoingTracks/sim/simprod_NuMu2019/21124/wPS_variables/wBDT
    #/data/ana/Diffuse/AachenUpgoingTracks/sim/simprod_NuMu2019/21002/wPS_variables/wBDT
    args = parser.parse_args()
    return args

def bdt_features(frame, wACE=False):
    """reads feature keys from frame into a dictionary
    take care to have this exactly like the feature_extractor from extractor.py for the BDT variables.
    """

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
    "sdir_e"              : frame["L5_sdir_e"].value,
    "E_truncated"         : np.NaN,
    "E_muex"              : np.NaN,
    "E_dnn"               : frame["TUM_dnn_energy_hive"]["mu_E_on_entry"],
    "random_variable"     : np.random.random()*10,
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

class Exponator(icetray.I3ConditionalModule):    
    """I3Module to add a key containging 10**TUM_DNN
    """
    
    def __init__(self,context):
        icetray.I3ConditionalModule.__init__(self, context)

    def Physics(self,frame):
        """takes TUM energy and exponentiates it.
        """
        log_E = frame["TUM_dnn_energy_hive"]["mu_E_on_entry"]
        E = 10**log_E
        frame.Put("TUM_dnn_energy_hive_GeV",      dataclasses.I3Double(E))
        self.PushFrame(frame)   
        
    def DAQ(self,frame):
        self.PushFrame(frame)

    def Finish(self):
        pass

class TrueMuonEnergy(icetray.I3ConditionalModule):
    """I3Module to add several true muon energies.
    
    AtInteraction: the Muon energy when it is first created
    All 3 are in GeV and AtDetectorLeave is = 0 if the Muon does not get out of the detector.
    """
    
    def Physics(self, frame):
        try:
            e_first, _ = sme.EnergyAtEgdeNoMuonGun(frame)
            frame.Put("ACEnergy_Truth",      dataclasses.I3Double(e_first))

        except:
            frame.Put("ACEnergy_Truth",      dataclasses.I3Double(np.NaN))

        self.PushFrame(frame)
        
    def DAQ(self,frame):
        #This runs on Q-Frames
        self.PushFrame(frame)
        
class ACEPredictor(icetray.I3ConditionalModule):
    """I3Module to add an energy prediction key
    Uses a trained BDT and needs the specific features it was trained on
    """
    
    def __init__(self,context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("model_path", "xgbooster model to be used, only pickled model and L5_E config is valid!")
        #rmse_combienergy_N2500_L5_E.model
        
        
    def Configure(self):
        """Loads a trained Booster model. 
        Do not load models that were saved with
        xgboosts .save_model or .dump_model method
        Called before frames start propagation through IceTray.
        """
        
        self.model = pickle.load(open(os.path.join(self.GetParameter("model_path")), "rb"))
        self.model_list = self.model.feature_names
        
    def Physics(self,frame):
        """predicts Energy and writes it to a key
        It builds a feature list for the BDT to make the prediction on. 
        If one of the input recos doesnt exist, it can still predict, see xgboosts docs.
        """
        features = bdt_features(frame)
        pandasframe = pd.DataFrame(data = features, index = [0])  #dataframe with one-line needs an index
        cleanframe = pandasframe[self.model_list]   #removes all non-feature keys from feature list

        #if cleanframe["E_truncated"][0] == np.NaN or cleanframe["E_muex"][0] == np.NaN:
        #    frame.Put("ACEnergy_Prediction", dataclasses.I3Double(np.NaN))
        #else:
        datamatrix = xgb.DMatrix(data = cleanframe)
        prediction = self.model.predict(datamatrix)
        prediction= 10**(prediction[0].astype(np.float64))
        frame.Put("ACEnergy_Prediction", dataclasses.I3Double(prediction))
        
        self.PushFrame(frame)

    def DAQ(self,frame):
        self.PushFrame(frame)

    def Finish(self):
        del self.model
        pass
    
if __name__ == '__main__':
    print("predictor started")
    
    #parse arguments
    args = parse_arguments()    
    model_path = os.path.join(full_path, "trained_models", args.model)
    infiles = args.infiles
    
    if not os.path.exists(os.path.split(args.outfile)[0]):
        os.makedirs(os.path.split(args.outfile)[0])
    
    outfile = args.outfile
    if outfile.endswith(".i3"):
        outfile+=(".zst")
    if not outfile.endswith(".i3.zst"):
        outfile+=(".i3.zst")
        
    pred_bool  = args.no_prediction
    truth_bool = args.no_truth
    #DNN_bool   = args.do_expDNN   
    
    #check there is actually something to be done.
    if not any([pred_bool,truth_bool]):
        print("All key-writing options were turned off. Exiting.")
        sys.exit()
        
    if len(infiles)==1:
        infile = infiles[0]
        print("i3 file will be saved at {}".format(outfile))
        tray = I3Tray()
        tray.Add("I3Reader","source", filename = infile) 

        if truth_bool:
            tray.AddModule(TrueMuonEnergy, "addingCombiEnergTruth")
        #if DNN_bool:
        #    tray.AddModule(Exponator, "addingDNNExp")
        if pred_bool:
            tray.AddModule(ACEPredictor, "addingCombiEnergy", model_path =  model_path)

        tray.Add("I3Writer", Filename = outfile)
        tray.Execute()
        tray.Finish()
        
    """
    #if several are supplied they will be batched
    elif len(infiles)>1:
        print("i3 files will be batched to {}".format(outfile))
        tray = I3Tray()
        tray.Add("I3Reader","source", filenamelist = infiles) 

        if truth_bool:
            tray.AddModule(TrueMuonEnergy, "addingCombiEnergTruth")
        #if DNN_bool:
        #    tray.AddModule(Exponator, "addingDNNExp")
        if pred_bool:
            tray.AddModule(ACEPredictor, "addingCombiEnergy", model_path =  model_path)

        tray.Add("I3Writer", Filename = outfile)
        tray.Execute()
        tray.Finish()
    """