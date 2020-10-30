"""
Builds a dagman job to predict BDT results on all i3 files in given folder, 
adding 2 new keys to them and saving them in a new folder.
Execute this with py3 env loaded on submit node to automatically build paths in submit scratch.
"""

import os
from dag_script import DagScript
import argparse
import time
import glob
import shutil

def parse_arguments():
    """argument parser to specify configuration"""
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i",
                        type=str, required = True,
                        help="/path/to/folder containing .i3.zst files")
    parser.add_argument("-o",
                        type=str, required = True,
                        help="/path/to/folder where i3 files with prefix 'wACE_' will be saved.")

    parser.add_argument("--error_dir", 
                        type=str, required = True,
                        help="place in your /data/user/ where .err and .out files are saved.")
    
    parser.add_argument("--dag_dir", 
                        type=str, required = True,
                        help="place on submit scratch where dagman and .log files are saved.")
    
    parser.add_argument("--model",
                        type=str,
                        default = "PICKLE_pshedelta_winner_N5000_L5_E_no_sigmapar_0.1.model",
                        help="trained xgboost model used for prediction.")   
    
    parser.add_argument("--no_prediction",
                       action="store_false",
                       help="flag to not write prediction keys")
    
    parser.add_argument("--no_truth",
                       action="store_false",
                       help="flag to not write truth keys")
    
    args = parser.parse_args()
    return args

#variables
args = parse_arguments() 
model = args.model
user = os.environ['USER']
date_str = time.strftime("predictor_%d_%m_%Y_%H_%M_%S", time.gmtime())
err_dir = os.path.join("/data/user/",user,args.error_dir, date_str, "logs")
dag_dir = os.path.join("/scratch",user,args.dag_dir, date_str)
log_dir = os.path.join(dag_dir, "logs")
submit_script = "/home/pfuerst/master_thesis/software/combienergy/condor/predict/predict.submit"
dag = DagScript(submitScript= "predict.submit")
infiles = sorted(glob.glob(os.path.join(args.i,"*.i3.zst")))

#make the dirs
if not os.path.exists(err_dir):
    os.makedirs(err_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(args.o):
    print("building paths...")
    os.makedirs(args.o)
#copy the submit script to /scratch
shutil.copy(submit_script,os.path.join(dag_dir,"predict.submit"))
#build the dagman



for infile in infiles:
    new_name = "wACE_"+os.path.split(infile)[1]
    outfile = os.path.join(args.o,new_name)
    jobname="JOB_"+str(os.path.split(infile)[1][-13:-7])
    params="--model {} -i {} -o {}".format(model, infile, outfile)

    if not args.no_prediction:
        params+=" --no_prediction"
    if not args.no_truth:
        params+=" --no_truth"
    submit_params = 'NAME="{}" ERR_DIR="{}" LOG_DIR="{}" PARAMS="{}"'.format(jobname, err_dir, log_dir, params)
                           
    dag.add(name=jobname, params = submit_params)
                    
dag.write(os.path.join(dag_dir,"dagman_predict.dag"))
print("*** dagman is at:")
print(dag_dir)
print("***")
print("copy this folder to your /scratch on the submit node if its not there already.")
print("Data will be at:")
print(args.o)