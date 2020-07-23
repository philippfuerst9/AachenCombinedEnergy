import configparser
config = configparser.ConfigParser()

#change this to your directory. combienergy should point to the folder with the git repository
config['directories'] = {
    'combienergy': '/home/pfuerst/master_thesis/software/combienergy/',
    'plots'      : '/home/pfuerst/master_thesis/plots/combienergy_plots/'}

with open('../files/main.cfg', 'w') as configfile:
    config.write(configfile)
    
print("main config saved at ../files/main.cfg")