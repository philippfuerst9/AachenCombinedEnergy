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

def parse_arguments():
    """argument parser to specify configuration"""
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model",
                        type=str,
                        default = "PICKLE_pshedelta_winner_N5000_L5_E_no_sigmapar_0.1.model",
                        help="trained xgboost model used for prediction.")

    parser.add_argument("-source",
                        type = str, 
                        required = True,
                        help = "directory path containing i3 files to be predicted on.")

    parser.add_argument("-dest",
                        type = str,
                        required = True,
                        help = "directory where new i3 files with new keys should be saved.")
    
    parser.add_argument("--loadfiles",
                        nargs = "+", type = str,
                        help = "list of filenames to be processed, can be batched into one file.")

    parser.add_argument("--n_i3batch",
                        type = int,
                        default = 10,
                        help = "No. of files to combine into new files if only directory is supplied.")
    
    parser.add_argument("--batching",
                        action="store_true",
                        help = "flag to batch i3 files into one bigger file.")
    
    parser.add_argument("--batchname", 
                        type = str,
                        required = ["--batching", "--loadfiles"] in sys.argv,
                        help = "if --loadfiles and --batching is set, this is the batched i3 files name.")
    
    parser.add_argument("--no_prediction",
                       action="store_false",
                       help="flag to not write prediction keys")
    parser.add_argument("--no_truth",
                       action="store_false",
                       help="flag to not write truth keys")
    parser.add_argument("--no_expDNN",
                       action="store_false",
                       help="flag to not write exponated DNN keys")
    #/data/ana/Diffuse/AachenUpgoingTracks/sim/simprod_NuMu2019/21124/wPS_variables/wBDT
    #/data/ana/Diffuse/AachenUpgoingTracks/sim/simprod_NuMu2019/21002/wPS_variables/wBDT
    args = parser.parse_args()
    return args

def bdt_features(frame, wACE=False):
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
        If one of the input recos doesnt exist, it predicts np.NaN
        """
            
        features = bdt_features(frame)
        pandasframe = pd.DataFrame(data = features, index = [0])  #dataframe with one-line needs an index
        cleanframe = pandasframe[self.model_list]   #cleans the feature extractor list

        if cleanframe["E_truncated"][0] == np.NaN or cleanframe["E_muex"][0] == np.NaN:
            frame.Put("ACEnergy_Prediction", dataclasses.I3Double(np.NaN))
        else:
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
    
def single_tray(source_dir, filename, dest_dir, model_path, addPred = True, addTruth = True, expDNN = True):
    """Builds the Icetray for one file
    """
    
    tray = I3Tray()
    tray.Add("I3Reader","source", filename = os.path.join(source_dir,filename)) 

    if addTruth:
        tray.AddModule(TrueMuonEnergy, "addingCombiEnergTruth")
    if expDNN:
        tray.AddModule(Exponator, "addingDNNExp")
    if addPred:
        tray.AddModule(ACEPredictor, "addingCombiEnergy", model_path =  model_path)
    
    tray.Add("I3Writer", Filename = os.path.join(dest_dir,filename))
    tray.Execute()
    tray.Finish()
    
def batch_tray(filenamelist, dest_dir, model_path, batch_name, addPred = True, addTruth = True, expDNN = True):
    """Builds the Icetray for a batch of files
    """

    tray = I3Tray()
    tray.Add("I3Reader","source", filenamelist = filenamelist) 

    if addTruth:
        tray.AddModule(TrueMuonEnergy, "addingCombiEnergTruth")
    if expDNN:
        tray.AddModule(Exponator, "addingDNNExp")
    if addPred:
        tray.AddModule(ACEPredictor, "addingCombiEnergy", model_path =  model_path)
    
    tray.Add("I3Writer", Filename = os.path.join(dest_dir,batch_name))
    tray.Execute()
    tray.Finish()
        
    
if __name__ == '__main__':
    
    print("predictor started")
    
    args = parse_arguments()    
    model_path = os.path.join(full_path, "trained_models", args.model)
    source_dir = args.source 
    dest_dir = args.dest

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    filenamelist = []
    if args.loadfiles is not None:
        filenamelist = args.loadfiles
    else:
        for filename in os.listdir(source_dir):
            if filename.endswith(".i3.zst"):
                filenamelist.append(os.path.join(filename))
                
    if args.batching:
        print("i3 files will be batched")

                        
    pred_bool  = args.no_prediction
    truth_bool = args.no_truth
    DNN_bool   = args.no_expDNN
    
    #check there is actually something to be done.
    if not any([pred_bool,truth_bool,DNN_bool]):
        print("All key-writing options were turned off. Exiting.")
        sys.exit()

    print("files are being saved at {}".format(dest_dir))
        
    #handle processing cases.
    #case 1) supply dir and batching turned on.
    if args.loadfiles is None and args.batching:
        nfiles = args.n_i3batch

        file_chunks = [filenamelist[x:x+nfiles] for x in range(0, len(filenamelist), nfiles)]
        print("File chunks look like this:")
        print(file_chunks[0])
        for chunk_idx, chunk in enumerate(file_chunks):
            print("Working on chunk No. {} --- {}".format(str(chunk_idx), str(chunk)))

            filepaths = ([os.path.join(source_dir, file) for file in chunk])

            outname = "chunk_{:04d}_nfiles_{}.i3.zst".format(chunk_idx, len(filepaths))
            #outname = "chunk_"+str(chunk_idx)+"_nfiles_"+str(len(filepaths))+".i3.zst"
            print(outname)
            batch_tray(filepaths, dest_dir, model_path, outname, addPred = pred_bool, addTruth = truth_bool, expDNN = DNN_bool)
            
          
    #case 3) supply filenamelist and batch them.
    if args.loadfiles is not None and args.batching:
        outname = args.batchname
        if not outname.endswith(".i3.zst"):
            outname+=".i3.zst"
        elif outname.endswith(".i3"):
            outname+=".zst"
                      
        filepaths = [os.path.join(source_dir, file) for file in filenamelist]
        batch_tray(filepaths, dest_dir, model_path, outname, addPred = pred_bool, addTruth = truth_bool, expDNN = DNN_bool)
    
    #case 2) and 4) supply filenamelist or dir but dont batch them
    if not args.batching:
        for filename in filenamelist:
            single_tray(source_dir, filename, dest_dir, model_path, addPred = pred_bool, addTruth = truth_bool, expDNN = DNN_bool)



"""                 
    if args.loadpath is not None:
        folder = args.loadpath
        print("all files from {} are being read".format(folder))
        nfiles = 10
        filenamelist = []
        for filename in os.listdir(folder):
            if filename.endswith(".i3.zst"):
                filenamelist.append(os.path.join(filename))


        file_chunks = [filenamelist[x:x+nfiles] for x in range(0, len(filenamelist), nfiles)]
        print("File chunks look like this:")
        print(file_chunks[0])
        print("---")
        print(file_chunks[1])
        for chunk_idx, chunk in enumerate(file_chunks):
            print("Working on chunk No. {} --- {}".format(str(chunk_idx), str(chunk)))

            filepaths = ([os.path.join(folder, file) for file in chunk])
            outname = "chunk_"+str(chunk_idx)+"_nfiles_"+str(len(filepaths))+".i3.zst"
            print(outname)

            tray = I3Tray()

            tray.Add("I3Reader","source", filenamelist = filepaths) #SkipKeys

            tray.AddModule(TrueMuonEnergy, "addingCombiEnergTruth")
            tray.AddModule(Exponator, "addingDNNExp")
            tray.AddModule(ACEPredictor, "addingCombiEnergy", model_path =  model_path)

            tray.Add("I3Writer", Filename = os.path.join(save,outname))
            tray.Execute()
            tray.Finish()
    
    if args.loadfiles is not None:
        files = args.loadfiles
        outname = args.batchname #[0] is the head, [1] is tail
        if not outname.endswith(".i3.zst"):
            outname += ".i3.zst"
        elif outname.endswith(".i3"):
            outname += ".zst"
        
        print("files being read are like {}".format(str(files[0])))
        tray = I3Tray()

        tray.Add("I3Reader","source", filenamelist = files) #SkipKeys

        tray.AddModule(TrueMuonEnergy, "addingCombiEnergTruth")
        tray.AddModule(Exponator, "addingDNNExp")
        tray.AddModule(ACEPredictor, "addingCombiEnergy", model_path =  model_path)

        tray.Add("I3Writer", Filename = os.path.join(save,outname))
        tray.Execute()
        tray.Finish()

"""