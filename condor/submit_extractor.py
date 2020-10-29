import os
from dag_script import DagScript
import argparse
import shutil
import glob

def parse_arguments():
    """argument parser to specify configuration"""
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset",
                        type=str, required = True,
                        help="Dataset to do extraction.")
    """
    parser.add_argument("--datapath", type = str,
                        required = False, 
                        default = "/data/ana/Diffuse/AachenUpgoingTracks/sim/simprod_NuMu2019"
                        help = "standard path/to/datasets.")
    
    parser.add_argument("--additional_keys", 
                        required = False, 
                        nargs = "+",
                        type = str,
                        help = "Additional keys to be extracted.")
    
    parser.add_argument("--out",
                        type = str,
                        required = False,
                        default = "/data/user/pfuerst/Reco_Analysis/systematic_sets_2019 ",
                        help="destination path.")
    parser.add_argument("--name",
                        type = str,
                        required = False,
                        help="name additional to the standard scheme.")
    """
    args = parser.parse_args()
    return args

args = parse_arguments()
submit_script = "/home/pfuerst/master_thesis/software/combienergy/condor/extract/extract_prediction.submit"
dag = DagScript(submitScript = ("extract_prediction.submit"))
source = os.path.join("/data/ana/Diffuse/AachenUpgoingTracks/sim/simprod_NuMu2019",args.dataset,"wDNN_wMunich_wACE")

outfolder = os.path.join("/data/user/pfuerst/Reco_Analysis/systematic_sets_2019",args.dataset)
if not os.path.exists(outfolder):
    os.makedirs(outfolder)

submit_folder = os.path.join("/scratch/pfuerst/combienergy/extractors", args.dataset)
log_folder = os.path.join(submit_folder, "logs")
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

err_out_folder = os.path.join("/data/user/pfuerst/condor/logs/extractors/", args.dataset, "logs")
if not os.path.exists(err_out_folder):
    os.makedirs(err_out_folder)
    
nfiles = 50
jobnr = 0
file_chunks_list = []
#filenamelist = []
#for filename in os.listdir(source):
#    if filename.endswith(".i3.zst"):
#        filenamelist.append(os.path.join(source,filename))
filenamelist = sorted(glob.glob(os.path.join(source, "*.i3.zst")))
file_chunks = [filenamelist[x:x+nfiles] for x in range(0, len(filenamelist), nfiles)]
file_chunks_list+=file_chunks

for chunk in file_chunks_list:
    loadfilestr = ""
    for file in chunk:
        loadfilestr+=(str(file)+" ")
    jobnr+=1
    batchname = "batch_{}_{:03d}_{}".format(args.dataset,jobnr, nfiles)
    dag.add(name = "JOB_"+str(jobnr), params = 'FILENAMELIST="%s" WRITE="%s" BATCHNAME="%s" SET="%s"'%(loadfilestr, os.path.join(outfolder,batchname), batchname, os.path.join(args.dataset, "logs")))
dag.write(os.path.join(submit_folder,"submit_dag_{}.dag".format(args.dataset)))
shutil.copy(submit_script, submit_folder)
