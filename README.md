# CombiEnergy

IceCube tool to combine different energy reconstructions for diffuse NuMu analysis and topological variables into a new Energy Reconstruction. 
To achieve this, an xgboost BDT is trained to predict the muon energy at detector entry (like truncated energy). 


# Run it on an IceCube Cobalt server

Git clone this repository to a place where you want it. In /config/make_main.py edit the path dependencies to point at your own paths and execute it. 
To run the tool, python 3.7, a compatible icetray build and an xgboost 1.1.1 installation is necessary (in python, `import xgboost; xgboost.__version__` ), along with some additional packages (pandas, yaml).

To create a list of all paths to your i3 files used for training the energy reco BDT use something like /config/builders/make_i3_pathlist.py which creates a .yaml file containing a list of paths.

First, run `extractor.py` on your i3 files to create a big pandas dataframe containing the keys necessary for training the BDT. New keys can be added to this frame via add_feature.py
Once the extractor is finished, you can train a model on the pandas dataframe with `trainer.py`. It will automatically cut a fraction of the data for validation and testing
and make a prediction on the testing data set which is once again saved as a pickle file. 

To predict energies on new i3 files with your trained model, run `predictor.py`

#personal to do list

1) change all hardcoded paths to depend only on main.cfg, add several "plots" paths to main.cfg one for BDT validation and one for analysis plots vs. topo vars.

2) build predictor.py to create either i3 files in the same format as the given ones (collect i3 file name) or to build a hdf5 file exactly as the NNMFit needs. This hdf5
file builder should be independent, i.e. takes (variable list) and writes it to hdf5 so that e.g. E_dnn can also be processed in this way.

3) add some custom features like DNN classifier output and remove nch,ndirc, nstringhits from training.

4) find joerans data sample for mis reconstructed muon events that look like tracks but arent and jlet the BDT predict their entry energies to see what happens.

#run nnmfit

`python NNMFit/analysis/run_fit.py /data/user/pfuerst/NNMFit/NNMFit/resources/configs/main_SPL.cfg --analysis_config /data/user/pfuerst/NNMFit/NNMFit/resources/configs/analysis_configs/asimov_SPL.yaml -o my_second_fit.pickle`