#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/icetray-start
#METAPROJECT /home/pfuerst/i3_software_py3/combo/build

"""This program takes a trained model and a directory containing i3 files.
It adds a model prediction and a true energy key to these files and saves them in another directory.
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
from scripts.extractor import feature_extractor




def parse_arguments():
    """argument parser to specify configuration"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                type=str,
                default = 'PICKLED_rmse_combienergy_N2500_L5_E.model',
                help="trained xgboost model used for prediction.")

    parser.add_argument("--feature_config",
                type = str,
                default = 'L5_E.yaml',
                help= "feature config used to train the loaded model, config name is the end of the model name.")
    
    parser.add_argument("--loadpath",
                        type = str, 
                        required = True,
                        help = "single directory path containing i3 files to be predicted on.")

    parser.add_argument("--savepath",
                        type = str,
                        required = True,
                        help = "directory where i3 files WITH new key should be saved.")

    #/data/ana/Diffuse/AachenUpgoingTracks/sim/simprod_NuMu2019/21124/wPS_variables/wBDT
    #/data/ana/Diffuse/AachenUpgoingTracks/sim/simprod_NuMu2019/21002/wPS_variables/wBDT
    args = parser.parse_args()
    return args



class TrueMuonEnergy(icetray.I3ConditionalModule):
    
    """I3Module to add several true muon energies.
    
    AtInteraction: the Muon energy when it is first created
    All 3 are in GeV and AtDetectorLeave is = 0 if the Muon does not get out of the detector.
    """
    
    def Physics(self, frame):
        try:
            e_first, e_last = sme.EnergyAtEgdeNoMuonGun(frame)
            frame.Put("TrueMuoneEnergyAtDetectorEntry", dataclasses.I3Double(e_first))
            frame.Put("TrueMuoneEnergyAtDetectorLeave", dataclasses.I3Double(e_last))
        except:
            frame.Put("TrueMuoneEnergyAtDetectorEntry", dataclasses.I3Double(np.NaN))
            frame.Put("TrueMuoneEnergyAtDetectorLeave", dataclasses.I3Double(np.NaN))
        nu, lepton, hadrons, nu_out = sme.get_interacting_neutrino_and_daughters(frame)
        mu_e = np.nan if lepton is None else lepton.energy
        frame.Put("TrueMuonEnergyAtInteraction", dataclasses.I3Double(mu_e))
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
        It builds a feature list for the BDT to make the prediction on. Care needs to be taken 
        to have this exactly the same as it was for training.
        It sometimes fails if entry energy cannot be calculated or one of the other keys is NaN or missing.
        """
        #try: 

        features = feature_extractor(frame)
        pandasframe = pd.DataFrame(data = features, index = [0])  #dataframe with one-line needs an index
        truth = pandasframe["E_entry"][0]
        cleanframe = pandasframe.drop(columns = ["E_entry", "E_exit"])
        cleanframe = cleanframe[self.model_list]   #this is superduper important. 
        datamatrix = xgb.DMatrix(data = cleanframe)
        prediction = self.model.predict(datamatrix)
        prediction= 10**(prediction[0].astype(np.float64))

        #except: 
        #    truth = np.NaN
        #    prediction = np.NaN
            
        self.PushFrame(frame)      
        frame.Put("ACEnergy_Truth",      dataclasses.I3Double(truth))
        frame.Put("ACEnergy_Prediction", dataclasses.I3Double(prediction))
        
    def DAQ(self,frame):
        #This runs on Q-Frames
        self.PushFrame(frame)

    def Finish(self):
        #Here we can perform cleanup work (closing file handles etc.)
        del self.model
        pass



    
if __name__ == '__main__':
    
    print("predictor started")
    
    args = parse_arguments()    
    model_path = os.path.join(full_path, "trained_models", args.model)
    config_path = os.path.join(full_path, "config", "files", args.feature_config)   
    folder = args.loadpath
    save = args.savepath
    
    print("model loaded from {}".format(model_path))
    print("files are being read from {}".format(folder))
    print("files are being saved at {} with keys ACEnergy_Truth and ACEnergy_Prediction".format(save))

    filenamelist = []
    for filename in os.listdir(folder):
        if filename.endswith(".i3.zst"):
            filenamelist.append(os.path.join(filename))

    for filename in filenamelist:
        print(filename)
        tray = I3Tray()

        tray.Add("I3Reader","source", filename = os.path.join(folder,filename)) #SkipKeys

        
        tray.AddModule(TrueMuonEnergy)
        tray.AddModule(ACEPredictor, "addingCombiEnergy", model_path =  model_path)

        tray.Add("I3Writer", Filename = os.path.join(save,filename))
        tray.Execute()
        tray.Finish()
