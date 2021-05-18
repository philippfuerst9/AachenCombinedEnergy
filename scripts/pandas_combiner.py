#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/icetray-start
#METAPROJECT /data/user/pfuerst/software/icecube/i3_software_py3/combo/build

import pickle
import pandas as pd
import os
import argparse
"""
short script to combine pandas dataframes into one big one for training after batching by extractor.
"/data/user/pfuerst/Reco_Analysis/Diffuse_sim_2019_nugen_northern_tracks/21124/wACE_wDNNexp/pickled/"
"/data/user/pfuerst/Reco_Analysis/Simulated_Energies_Lists/feature_dataframes/2019_frames/21124_wXY.pickle"
"""
def parse_arguments():
    """argument parser to specify configuration"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pathlist", type = str, required = True,
        nargs="+",
        help= "paths to pickled pandas dataframes")
    
    parser.add_argument(
        "--save", required = True,
        help="/path/to/combined/file.pickle")

    args = parser.parse_args()
    return args

args = parse_arguments()
pathlist = args.pathlist
save = args.save
if not type(pathlist)==list:
    pathlist = [pathlist]
if not os.path.exists(os.path.dirname(save)):
    os.makedirs(os.path.dirname(save))

dataframes = []
for path in pathlist:
    print(path)
    for pickled in sorted(os.listdir(path)):
        if pickled.endswith(".pickle"):
            print(pickled)
            with open(os.path.join(path,pickled), "rb") as file:
                currentframe = pickle.load(file)
                dataframes.append(currentframe)
                del currentframe
    
full = pd.concat(dataframes)
full.to_pickle(args.save)
#build 2019
"""
path21002 = "/data/user/pfuerst/Reco_Analysis/Diffuse_sim_2019_nugen_northern_tracks/21002/wACE_wDNNexp/pickled/"
path21124 = "/data/user/pfuerst/Reco_Analysis/Diffuse_sim_2019_nugen_northern_tracks/21124/wACE_wDNNexp/pickled/"

full_2019 = False
if full_2019:
    dataframes = []
    pathlist = [path21002, path21124]
    for path in pathlist:
        print(path)
        for pickled in os.listdir(path):
            print(pickled)
            with open(path+pickled, "rb") as file:
                currentframe = pickle.load(file)
                dataframes.append(currentframe)
                del currentframe

    full2019 = pd.concat(dataframes)
    full2019.to_pickle("/data/user/pfuerst/Reco_Analysis/Simulated_Energies_Lists/feature_dataframes/2019_frames/full_2019_wXY.pickle")

#build 2019 single sets
dataframes = []
print(path21002)
for pickled in os.listdir(path21002):
    print(pickled)
    with open(path21002+pickled, "rb") as file:
        currentframe = pickle.load(file)
        dataframes.append(currentframe)
        del currentframe
full21002 = pd.concat(dataframes)
full21002.to_pickle("/data/user/pfuerst/Reco_Analysis/Simulated_Energies_Lists/feature_dataframes/2019_frames/21002_wXY.pickle")

dataframes = []
print(path21124)
for pickled in os.listdir(path21124):
    print(pickled)
    with open(path21124+pickled, "rb") as file:
        currentframe = pickle.load(file)
        dataframes.append(currentframe)
        del currentframe
full21124 = pd.concat(dataframes)
full21124.to_pickle("/data/user/pfuerst/Reco_Analysis/Simulated_Energies_Lists/feature_dataframes/2019_frames/21124_wXY.pickle")

"""
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