# CombiEnergy

IceCube tool to combine different energy reconstructions for diffuse NuMu analysis and topological variables into a new Energy Reconstruction. 
To achieve this, an xgboost BDT is trained to predict the muon energy at detector entry (like truncated energy). 


# Run it

To run the tool, python 3.7 a compatible icetray build and xgboost installation is necessary, along with some additional packages (pandas, yaml)



