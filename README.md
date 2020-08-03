# CombiEnergy

IceCube tool to combine different energy reconstructions for diffuse NuMu analysis and topological variables into a new Energy Reconstruction. 
To achieve this, an xgboost BDT is trained to predict the muon energy at detector entry (like truncated energy). 


# Run it on an IceCube Cobalt server

Git clone this repository to a place where you want it. The major scripts are in /scripts. Each of them contains hardcoded paths which need to be set up to your environmet,
this includes path to directories for plots etc. 
To run the tool, python 3.7, a compatible icetray build and an xgboost 1.1.1 installation is necessary (in python, `import xgboost; xgboost.__version__` ), along with some additional packages (pandas, yaml).

# 1) Run extractor.py

To  run `extractor.py` you need to create a list of all paths to your i3 files used for training the energy reco BDT. This is done with `/config/builders/make_i3_pathlist.py` which creates a .yaml file containing a list of paths.
`extractor.py` then creates a big pandas dataframe containing the keys necessary for training the BDT. New keys can be added to this frame via add_feature.py or by changing the function extractor.py

# 2) Run trainer.py

Once the extractor is finished, you can train a model on the pandas dataframe with `trainer.py`. It needs the feature configuration you want. Create your own with `/config/builders/make_featurelist.py`. 
The trainer can be customized beyond the input feature config file (read through argparse arguments).
This script will automatically cut 20% the training data for validation during training and use the supplied percentage of testing data to create some analysis plots.
The prediction on the testing data set is also directly saved as a pickle file. Check hardcoded paths to plots and pickle file dirs.

# 3) Run predictor.py

To predict energies on new i3 files with your trained model, run `predictor.py`. Load your trained model. predictor.py has to different options, either supply it
with an entire directory containing i3 files which are then batched into new i3 files (e.g. 10 old files into one new file), or supply it with a list of filenames (/path/to/filename).
Then, all supplied files will be batched into one new file. This is convenient for submitting jobs to the condor cluster.

# Condor

In `/condor` are some example scripts to submit the three scripts to the condor cluster. Remove hardcoded paths for executables and output directories.

#personal to do list

NNMFIT: 
`python NNMFit/analysis/run_fit.py /data/user/pfuerst/NNMFit/NNMFit/resources/configs/main_SPL.cfg --analysis_config /data/user/pfuerst/NNMFit/NNMFit/resources/configs/analysis_configs/asimov_SPL.yaml -o my_second_fit.pickle`