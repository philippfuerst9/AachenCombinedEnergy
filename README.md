# CombiEnergy "ACE"

IceCube tool to combine different energy reconstructions for diffuse NuMu analysis and topological variables into a new Energy Reconstruction. 
To achieve this, an xgboost BDT is trained to predict the muon energy at detector entry (like truncated energy). 


# Run it on an IceCube Cobalt server

Git clone this repository to a place where you want it. The /analysis folder contains plotting scripts for my master's thesis (Philipp Fürst) and is not part of the module. The major scripts are in /scripts. Each of them contains hardcoded paths which need to be set up to your environmet,
this includes path to directories for plots etc. 

On Cobalt you can load your icetray environment and then load a virtual environment where everything is installed with `source /home/pfuerst/venvs/venv_py3-v4.1.1/bin/activate` or you can install all the necessary packages yourself (python 3.7, a compatible icetray build and an xgboost 1.1.1 installation is necessary, along with some additional packages (pandas, yaml, etc.). 

# 1) Run extractor.py

To  run `extractor.py` you need to create a list of all paths to your i3 files used for training the energy reco BDT. This is done with `/config/builders/make_i3_pathlist.py` which creates a .yaml file containing a list of paths. Alternatively, just supply one path to a directory containing i3-files or supply a single file.
`extractor.py` then creates a pandas dataframe containing the keys necessary for training the BDT. New keys can be added to this frame via `add_feature.py` or by changing the function `feature_extractor.py`. One can also extract from batches of i3-files to pickles and then run `pandas_combiner.py` to combine these dataframes. This is the fastest method to extract data from large amounts of i3-files. 

# 2) Run trainer.py

Once the extractor is finished, you can train a model on the pandas dataframe with `trainer.py`. It needs the feature configuration you want. Create your own with `/config/builders/make_featurelist.py`. 
The trainer can be customized beyond the input feature config file (read through argparse arguments).
This script will automatically cut some 20% of the training data for validation during training and use the supplied percentage of testing data to automatically create some analysis plots for a quick-low statistics investigation of BDT performance.
The prediction on the testing data set is also directly saved as a pickle file. Check hardcoded paths to plots and pickle file dirs.

# 3) Run predictor.py

To predict energies on new i3 files with your trained model, run `predictor.py`. To build a dagman to submit to condor, run `condor/submit_predictor.py`. It will create a dagman, submit script and some log folders at the specified places. Care to change any hardcoed paths.

# ACE

To add the ACE energy estimator to your files, simply run predictor.py with the default trained model. Check that your i3-files already contain the necessary features.

# Condor

In `/condor` are some example scripts to submit the three scripts to the cluster, including wrapper scripts to load icetray and the virtual environment. 
