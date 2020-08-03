import os
from dag_script import DagScript

dag = DagScript(submitScript= "predict.submit")


folder = "/data/ana/Diffuse/AachenUpgoingTracks/sim/simprod_NuMu2019/21124/wPS_variables/wBDT/"
out    = "/data/user/pfuerst/Reco_Analysis/Diffuse_sim_2019_nugen_northern_tracks/21124/wACE/"
nfiles = 10
filenamelist = []
for filename in os.listdir(folder):
    if filename.endswith(".i3.zst"):
        filenamelist.append(os.path.join(folder, filename))
    
file_chunks = [filenamelist[x:x+nfiles] for x in range(0, len(filenamelist), nfiles)]


jobnr = 0
for chunk in file_chunks:
    loadfilestr = ""
    for file in chunk:
        loadfilestr+=(str(file)+" ")
    jobnr +=1
    batchname = "batch_"+str(jobnr)+"_n_"+str(len(chunk))
    dag.add(name= "JOB_"+str(jobnr),params = 'SAVEPATH="%s" LOADFILES="%s" BATCHNAME="%s"'%(out, loadfilestr, batchname))

dag.write("/home/pfuerst/master_thesis/software/combienergy/condor/predict_21124.dag")