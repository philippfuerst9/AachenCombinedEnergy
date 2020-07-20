import xgboost as xgb
import numpy as np
import pandas as pd #?
import argparse
from icecube import icetray, dataclasses, dataio
from I3Tray import *
import time

#load trained model

#predictor has to know which feature function was used in the extractor.

# predictor predicts E_true, E_pred and writes them as keys into an i3 file


#predictor could take a required argument "featuremap" which tells him which feature extractor from functions he should use. the argument given ist the feature extractor function name

#for each feature name it should be clear how this feature is processed from an i3 file, i.e.
#["cog_rho"] --> frame["L5_cog_rho"].value
#maybe write a function for each variable and one function for the BDT variables which are all the same?

#get_e_trunc or sth

class CombiEnergyPredictor(icetray.I3ConditionalModule):
    def __init__(self,context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("modelfile", "xgbooster model to be used", "/home/pfuerst/master_thesis/software/BDT_models/trained_models/pshedelta_good_pshe_d3_n5k_N5000.model")

    def Configure(self):
        #This is called before frames start propagation through IceTray
        self._model   = xgb.Booster()
        self._model.load_model(self.GetParameter("modelfile"))


    def Physics(self,frame):
        #this thing gets a feature list config file and needs to figure out how to read the according features
        #from an i3 file. Thats shitty because there is room for bullshit to happen
        
        truth = np.NaN
        prediction = np.NaN
        try:
            #true energy is energy at detector entry
            truth, e_last = sme.EnergyAtEgdeNoMuonGun(frame) 
            label = [np.log10(truth)]
            
            lempty = frame["SplineMPEICCharacteristicsIC"].empty_hits_track_length
            split_geo = math.cos(frame["SPEFit2GeoSplit2"].dir.zenith)
            #features are same as these:
            #https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/projects/finallevel_filter_diffusenumu/trunk/python/level5/segments.py
            #added is n_string_hits, E_truncated, E_muex, E_dnn and random_variable
            #E_dnn prediction is in log10 E already.
            features = {
                "cog_rho"             : math.sqrt(frame["HitStatisticsValuesIC"].cog.x**2. + frame["HitStatisticsValuesIC"].cog.y**2.),
                "cog_z"               : frame["HitStatisticsValuesIC"].cog.z,
                "lseparation"         : frame["SplineMPEICCharacteristicsIC"].track_hits_separation_length,
                "nch"                 : frame["HitMultiplicityValuesIC"].n_hit_doms,
                "bayes_llh_diff"      : frame["SPEFit2BayesianICFitParams"].logl-frame["SPEFit2ICFitParams"].logl,
                "cos_zenith"          : math.cos(frame["SplineMPEIC"].dir.zenith),
                "rlogl"               : frame["SplineMPEICFitParams"].rlogl,
                "ldir_c"              : frame["SplineMPEICDirectHitsICC"].dir_track_length,
                "ndir_c"              : frame["SplineMPEICDirectHitsICC"].n_dir_doms,
                "sigma_paraboloid"    : math.sqrt(frame["MPEFitParaboloidFitParams"].pbfErr1**2. + frame["MPEFitParaboloidFitParams"].pbfErr2**2.)/math.sqrt(2.) / I3Units.degree,
                "sdir_e"              : frame["SplineMPEICDirectHitsICE"].dir_track_hit_distribution_smoothness,
                "n_string_hits"       : frame["HitMultiplicityValuesIC"].n_hit_strings,
                "E_truncated"         : np.log10(frame["SplineMPEICTruncatedEnergySPICEMie_AllDOMS_Muon"].energy),
                "E_muex"              : np.log10(frame["SplineMPEICMuEXDifferential"].energy),
                "E_dnn"               : frame["TUM_dnn_energy"]["mu_E_on_entry"],
                "random_variable"     : np.random.random()*10
                }

            pandasframe = pd.DataFrame(data = features, index = [0])
            datamatrix  = xgb.DMatrix(data = pandasframe, label =label)
            prediction  = self._model.predict(datamatrix)
            #.predict has a list output (with just one float32 in it which is not accepted as I3Double)
            #it also predicts log10 energy but true energy should be in keys
            prediction= 10**(prediction[0].astype(np.float64))

            frame.Put("ACEnergyCombi_Truth",      dataclasses.I3Double(truth)   )
            frame.Put("ACEnergyCombi_Prediction", dataclasses.I3Double(prediction))
            self.PushFrame(frame)
        except:
            self.PushFrame(frame)      
            frame.Put("ACEnergyCombi_Truth",      dataclasses.I3Double(truth)   )
            frame.Put("ACEnergyCombi_Prediction", dataclasses.I3Double(prediction))
            
    def DAQ(self,frame):
        #This runs on Q-Frames
        self.PushFrame(frame)

    def Finish(self):
        #Here we can perform cleanup work (closing file handles etc.)
        pass


#if __name__ == '__main__':
#    #print help(SimpleModule)
#    testsave = "/data/user/pfuerst/test_with_BDT.i3"
#    tray = I3Tray()
#    tray.AddModule('I3Reader','reader', Filename = files[0])
                   #FilenameList = [os.path.expanduser('~/i3/data/IC86-2011_nugen_numu/Level3_nugen_numu_IC86.2011.010039.000000.i3')])
                    #use either filename and a name
                    #or filenamelist and a list of filenames. otherwise wont work
#    tray.AddModule(CombiEnergyPredictor,"CombiEnergyPredictor")

#    tray.Add("I3Writer",Filename=os.path.expanduser(testsave))

#    tray.Execute()
#    tray.Finish()
    
    ##include the special hdf5 writer here that is necessary to run the NNMFit.
    
    ## then change pass2 variables Truncated_energy --> New Energy
    ## in datasets.cfg
    
    ##muss wissen wo die neuen hdfs sind
    ##muss wissen welche keys darin sind
    ##muss wissen welchen key er statt truncated energy nehmen soll (das ist hardcoded)


































































