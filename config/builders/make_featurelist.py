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


#any other custom config. If you want to add new features use add_features.py

custom_list = []
custom_name = None

#with open(custom_name, 'w') as file:
#    yaml.dump(custom_list, file)