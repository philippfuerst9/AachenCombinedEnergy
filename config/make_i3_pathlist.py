import yaml
#whatever files need to be read, buid a .yaml file containing list of paths to them and supply it to extractor.

#this is messed up because processed DNN energies are all over the place. Its hardcoded because its done just once.
#all BDT's can be trained from this
pathlist = []  
path11069    = "/data/user/pfuerst/Reco_Analysis/Diffuse_sim_2012_nugen_11069_i3files/full/withDNN/"
folders11069 = ["01000-01999", "02000-02999", "03000-03999", "04000-04999"]

for folder in folders11069:
    pathlist.append(path11069+folder+"/")
                      
                      
path11070    = "/data/ana/Diffuse/IC2010-2014_NuMu/IC86-2012/datasets/finallevel/sim/2012/neutrino-generator/11070/wDNN/01000-01999/"
pathlist.append(path11070)
path11029 = "/data/ana/Diffuse/IC2010-2014_NuMu/IC86-2012/datasets/finallevel/sim/2012/neutrino-generator/11029/wDNN/"
folders11029 = ["02000-02999","03000-03999",  "04000-04999",  "05000-05999"]                      
for folder in folders11029:
    pathlist.append(path11029+folder+"/")

name = 'files/i3_pathlist.yaml'
with open(name, 'w') as file:
    yaml.dump(pathlist, file)
print("file saved at {}".format(name))


#new coherent version from TUM Hans (11029, 11069, 11070 processed in one go)
pathlist2 = []
main_path = "/data/ana/Diffuse/IC2010-2014_NuMu/IC86-2012/datasets/finallevel/sim/2012/neutrino-generator/wPS_variables/"
folders   = ["11029/","11069/","11070/"]

for folder in folders:
    pathlist2.append(main_path+folder)
    
    
name2 = 'files/i3_pathlist_v2.yaml'
with open(name2, 'w') as file2:
    yaml.dump(pathlist2, file2)
print("file saved at {}".format(name2))



#custom new pathlist: