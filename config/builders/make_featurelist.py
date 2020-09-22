import yaml
#standard full features
list_of_features = ["cog_rho","cog_z","lseparation","nch","bayes_llh_diff",
                    "cos_zenith","rlogl","ldir_c",
                    "ndir_c","sigma_paraboloid","sdir_e","n_string_hits",
                    "E_truncated","E_muex","E_dnn","random_variable"]

name = '../files/standard_feat.yaml'
with open(name, 'w') as file:
    yaml.dump(list_of_features, file)
print("file saved at {}".format(name))

#no energies
list_of_features_no_E = ["cog_rho","cog_z","lseparation","nch","bayes_llh_diff",
                    "cos_zenith","rlogl","ldir_c",
                    "ndir_c","sigma_paraboloid","sdir_e","n_string_hits", "random_variable"]



name_no_E = '../files/no_E_feat.yaml'
with open(name_no_E, 'w') as file:
    yaml.dump(list_of_features_no_E, file)
print("file saved at {}".format(name_no_E))



#no photon information
list_of_features_no_light = ["cog_rho","cog_z","lseparation","bayes_llh_diff",
                    "cos_zenith","rlogl","ldir_c",
                    "sigma_paraboloid","sdir_e",
                    "E_truncated","E_muex","E_dnn","random_variable"]

name_no_light = '../files/no_light_feat.yaml'
with open(name_no_light, 'w') as file:
    yaml.dump(list_of_features_no_light, file)
print("file saved at {}".format(name_no_light))

#standard L5+energies
L5_E = ["cog_rho","cog_z","lseparation","nch","bayes_llh_diff",
                    "cos_zenith","rlogl","ldir_c",
                    "ndir_c","sigma_paraboloid","sdir_e",
                    "E_truncated","E_muex","E_dnn","random_variable"]

name_L5_E = '../files/L5_E.yaml'
with open(name_L5_E, 'w') as file:
    yaml.dump(L5_E, file)
print("file saved at {}".format(name_L5_E))

#energies +L5 -sigma paraboloid as it has some failed fits and unphysical information.
L5_E_nopar = ["cog_rho","cog_z","lseparation","nch","bayes_llh_diff",
                    "cos_zenith","rlogl","ldir_c",
                    "ndir_c","sdir_e",
                    "E_truncated","E_muex","E_dnn","random_variable"]

name_L5_E_no_sigmapar = '../files/L5_E_no_sigmapar.yaml'
with open(name_L5_E_no_sigmapar, 'w') as file:
    yaml.dump(L5_E_nopar, file)
print("file saved at {}".format(name_L5_E_no_sigmapar))

#Index(['E_dnn', 'E_entry', 'E_exit', 'E_muex', 'E_truncated',
#       'MCPrimaryCosZen', 'MCPrimaryEnergy', 'MCPrimaryType', 'NEvent',
#       'OneWeight', 'TIntProbW', 'bayes_llh_diff', 'cog_rho', 'cog_x', 'cog_y',
#       'cog_z', 'cos_zenith', 'ldir_c', 'lseparation', 'mean_r', 'mean_x',
#       'mean_y', 'mean_z', 'nch', 'ndir_c', 'random_variable', 'rlogl',
#       'sdir_e', 'sigma_paraboloid'],
#      dtype='object')

#Trunc+MuEx+DNN +Topo - sigmapar -cog_rho +cog_x + cog_y +geometric xyz
full_info = ["cog_x","cog_y","cog_z","mean_x","mean_y","mean_z","lseparation","nch","bayes_llh_diff",
                    "cos_zenith","rlogl","ldir_c",
                    "ndir_c","sdir_e",
                    "E_truncated","E_muex","E_dnn","random_variable"]

name_full_info = '../files/full_info.yaml'
with open(name_full_info, 'w') as file:
    yaml.dump(full_info, file)
print("file saved at {}".format(name_full_info))

#Trunc+MuEx+DNN +Topo - sigmapar -cog_rho +cog_x +cog_y
full_info_no_geo = ["cog_x","cog_y","cog_z","lseparation","nch","bayes_llh_diff",
                    "cos_zenith","rlogl","ldir_c",
                    "ndir_c","sdir_e",
                    "E_truncated","E_muex","E_dnn","random_variable"]

name_full_info_no_geo = '../files/full_info_no_geo.yaml'
with open(name_full_info_no_geo, 'w') as file:
    yaml.dump(full_info_no_geo, file)
print("file saved at {}".format(name_full_info_no_geo))

#DNN + Topo -sigmapar -cog_rho +cog_x +cog_y +geometric xyz
DNN_full_topology = ["cog_x","cog_y","cog_z","mean_x","mean_y","mean_z","lseparation","nch","bayes_llh_diff",
                    "cos_zenith","rlogl","ldir_c",
                    "ndir_c","sdir_e","E_dnn","random_variable"]

name_DNN_full_topology = '../files/DNN_full_info.yaml'
with open(name_DNN_full_topology, 'w') as file:
    yaml.dump(DNN_full_topology, file)
print("file saved at {}".format(name_DNN_full_topology))

#No Energies + Topo +geometric
only_topology = ["cog_x","cog_y","cog_z","mean_x","mean_y","mean_z","lseparation","nch","bayes_llh_diff",
                    "cos_zenith","rlogl","ldir_c",
                    "ndir_c","sdir_e","random_variable"]

name_only_topology = '../files/topology_only.yaml'
with open(name_only_topology, 'w') as file:
    yaml.dump(only_topology, file)
print("file saved at {}".format(name_only_topology))



# -----  any other custom config. If you want to add new features use add_features.py ----- #

#custom_list = []
#custom_name = None

#with open(custom_name, 'w') as file:
#    yaml.dump(custom_list, file)