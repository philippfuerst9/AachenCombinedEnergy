import pickle
import pandas as pd
import os
"""
short script to combine pandas dataframes into one big one for training.
This is here for completeness sake and as an example, not intended to be run.
"""

#build 2019

path21002 = "/data/user/pfuerst/Reco_Analysis/Diffuse_sim_2019_nugen_northern_tracks/21002/wACE/pickled/"
path21124 = "/data/user/pfuerst/Reco_Analysis/Diffuse_sim_2019_nugen_northern_tracks/21124/wACE/pickled/"
dataframes = []
pathlist = [path21002, path21124]
for path in pathlist:
    print(path)
    for pickled in os.listdir(path):
        print(pickled)
        with open(path+pickled, "rb") as file:
            currentframe = pickle.load(file)
            dataframes.append(currentframe)

full2019 = pd.concat(dataframes)
full2019.to_pickle("/data/user/pfuerst/Reco_Analysis/Simulated_Energies_Lists/feature_dataframes/2019_frames/full_2019.pickle")

#build 2012
"""
path = "/data/user/pfuerst/Reco_Analysis/Simulated_Energies_Lists/feature_dataframes/2012_frames/"

frames = []
for pickled in os.listdir(path):
    currentframe = pickle.load(open(path+pickled, "rb"))
    frames.append(currentframe)
full2012 = pd.concat(frames)
full2012.to_pickle("/data/user/pfuerst/Reco_Analysis/Simulated_Energies_Lists/feature_dataframes/2012_frames/full2012.pickle")

frames = []
for pickled in os.listdir(path):
    if "_11070" in pickled:
        currentframe = pickle.load(open(path+pickled, "rb"))
        frames.append(currentframe)
new11070 = pd.concat(frames)
new11070.to_pickle("/data/user/pfuerst/Reco_Analysis/Simulated_Energies_Lists/feature_dataframes/2012_frames/full11070.pickle")

frames = []
for pickled in os.listdir(path):
    if "_11069" in pickled:
        currentframe = pickle.load(open(path+pickled, "rb"))
        frames.append(currentframe)

new11069 = pd.concat(frames)
new11069.to_pickle("/data/user/pfuerst/Reco_Analysis/Simulated_Energies_Lists/feature_dataframes/2012_frames/full11069.pickle")

frames = []
for pickled in os.listdir(path):
    if "_11029" in pickled:
        currentframe = pickle.load(open(path+pickled, "rb"))
        frames.append(currentframe)
new11029 = pd.concat(frames)
new11029.to_pickle("/data/user/pfuerst/Reco_Analysis/Simulated_Energies_Lists/feature_dataframes/2012_frames/full11029.pickle")
"""