# CombiEnergy

IceCube tool to combine different energy reconstructions for diffuse NuMu analysis and topological variables into a new Energy Reconstruction. 
To achieve this, an xgboost BDT is trained to predict the muon energy at detector entry (like truncated energy). 


# Run it

To run the tool, python 3.7 a compatible icetray build and xgboost installation is necessary, along with some additional packages (pandas, yaml)
To create a list of all paths to your i3 files used for training the energy reco BDT use something like /config/builders/make_i3_pathlist.py which creates a .yaml file containing a list of paths.



