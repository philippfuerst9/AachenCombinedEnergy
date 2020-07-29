#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/icetray-start
#METAPROJECT /home/pfuerst/i3_software_py3/combo/build

from icecube import dataio, dataclasses, icetray, common_variables, paraboloid
from icecube.icetray import I3Units
import numpy as np
import math
import os
import imp
import extractor as ex
sme = imp.load_source('joerans_module', '/home/pfuerst/master_thesis/software/Segmented_Muon_Energy_jstettner.py')

#this is how it should be done if new features are added to the Pandas DataFrame later

savepath = "/data/user/pfuerst/Reco_Analysis/Simulated_Energies_Lists/feature_dataframes/"
standard_pandas_name = 'features_dataframe_11029_11060_11070_withNaN.pkl'

standard_pandas_frame = pd.read_pickle(savepath+standard_pandas_name)

new_feature_name = "random_variable_2"
new_feature_list = []

for path in ex.pathlist:
    print("--- folder {} ---".format(path))
    for filename in os.listdir(path):
        if filename.endswith(".i3.zst"):
            print("processing file {}".format(filename))
            with dataio.I3File(os.path.expanduser(path+filename)) as f:
                for currentframe in f:        
                    if str(currentframe.Stop) == "Physics":
                        
                        new_feature = np.random.random()
                        new_feature_list.append(new_feature)
                        

new_feature_dict = {}
new_feature_dict[new_feature_name] = new_feature_list

standard_pandas_frame.update(new_feature_dict)


#savepath = "/data/user/pfuerst/Reco_Analysis/Simulated_Energies_Lists/feature_dataframes/"
#standard_pandas_frame.to_pickle(savepath+'features_dataframe_11029_11060_11070_withNaN.pkl')







