import os
from dag_script import DagScript

#Dagman for 21124 data set
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
    batchname = "batch_{:03d}_n_{}".format(jobnr, len(chunk))
    dag.add(name= "JOB_"+str(jobnr),params = 'SAVEPATH="%s" LOADFILES="%s" BATCHNAME="%s"'%(out, loadfilestr, batchname))

dag.write("/home/pfuerst/master_thesis/software/combienergy/condor/predict_21124.dag")

#Dagman for 21002 data set
dag = DagScript(submitScript= "predict2.submit")
folder = "/data/ana/Diffuse/AachenUpgoingTracks/sim/simprod_NuMu2019/21002/wPS_variables/wBDT"
out    = "/data/user/pfuerst/Reco_Analysis/Diffuse_sim_2019_nugen_northern_tracks/21002/wACE/"
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
    batchname = "batch_{:03d}_n_{}".format(jobnr, len(chunk))
    dag.add(name= "JOB_"+str(jobnr),params = 'SAVEPATH="%s" LOADFILES="%s" BATCHNAME="%s"'%(out, loadfilestr, batchname))

dag.write("/home/pfuerst/master_thesis/software/combienergy/condor/predict_21002.dag")


#dagman to extract all 2012 data to pandas dataframes
dag = DagScript(submitScript = ("/scratch/pfuerst/combienergy/condor/extractors/dagman/extract.submit"))
source = "/data/ana/Diffuse/IC2010-2014_NuMu/IC86-2012/datasets/finallevel/sim/2012/neutrino-generator/wPS_variables/"
folders = ["11029", "11069", "11070"]
outfolder = "/data/user/pfuerst/Reco_Analysis/Simulated_Energies_Lists/feature_dataframes/2012_frames/"
nfiles = 20
jobnr = 0
file_chunks_list = []
for folder in folders:
    filenamelist = []
    for filename in os.listdir(os.path.join(source,folder)):
        if filename.endswith(".i3.zst"):
            filenamelist.append(os.path.join(source,folder,filename))
    file_chunks = [filenamelist[x:x+nfiles] for x in range(0, len(filenamelist), nfiles)]
    file_chunks_list+=file_chunks

for chunk in file_chunks_list:
    loadfilestr = ""
    for file in chunk:
        loadfilestr+=(str(file)+" ")
    jobnr+=1
    #batchname = "batch_"+str(jobnr)+"_"+chunk[0][107:112]
    batchname = "batch_{:03d}_{}".format(jobnr, chunk[0][107:112])
    dag.add(name = "JOB_"+str(jobnr), params = 'FILENAMELIST="%s" WRITE="%s" BATCHNAME="%s"'%(loadfilestr, outfolder+batchname, batchname))

dag.write("/home/pfuerst/master_thesis/software/combienergy/condor/extract2012.dag")

#dagman to extract 2019 21124 data to pandas dataframes
dag = DagScript(submitScript = ("/scratch/pfuerst/combienergy/condor/extractors/07-09-2020_extract2019_wXYcoords/extract_prediction.submit"))
source = "/data/user/pfuerst/Reco_Analysis/Diffuse_sim_2019_nugen_northern_tracks/21124/wACE_wDNNexp/"
outfolder = "/data/user/pfuerst/Reco_Analysis/Diffuse_sim_2019_nugen_northern_tracks/21124/wACE_wDNNexp/pickled/"
nfiles = 20
jobnr = 0
file_chunks_list = []

filenamelist = []
for filename in os.listdir(source):
    if filename.endswith(".i3.zst"):
        filenamelist.append(os.path.join(source,filename))
file_chunks = [filenamelist[x:x+nfiles] for x in range(0, len(filenamelist), nfiles)]
file_chunks_list+=file_chunks

for chunk in file_chunks_list:
    loadfilestr = ""
    for file in chunk:
        loadfilestr+=(str(file)+" ")
    jobnr+=1
    batchname = "batch_{:03d}_{}".format(jobnr, nfiles)
    dag.add(name = "JOB_"+str(jobnr), params = 'FILENAMELIST="%s" WRITE="%s" BATCHNAME="%s"'%(loadfilestr, outfolder+batchname, batchname))
dag.write("/home/pfuerst/master_thesis/software/combienergy/condor/extract21124.dag")


#dagman to extract 2019 21002 data to pandas dataframes
dag = DagScript(submitScript = ("/scratch/pfuerst/combienergy/condor/extractors/07-09-2020_extract2019_wXYcoords/extract_prediction.submit"))
source = "/data/user/pfuerst/Reco_Analysis/Diffuse_sim_2019_nugen_northern_tracks/21002/wACE_wDNNexp/"
outfolder = "/data/user/pfuerst/Reco_Analysis/Diffuse_sim_2019_nugen_northern_tracks/21002/wACE_wDNNexp/pickled/"
nfiles = 20
jobnr = 0
file_chunks_list = []

filenamelist = []
for filename in os.listdir(source):
    if filename.endswith(".i3.zst"):
        filenamelist.append(os.path.join(source,filename))
file_chunks = [filenamelist[x:x+nfiles] for x in range(0, len(filenamelist), nfiles)]
file_chunks_list+=file_chunks

for chunk in file_chunks_list:
    loadfilestr = ""
    for file in chunk:
        loadfilestr+=(str(file)+" ")
    jobnr+=1
    batchname = "batch_{:03d}_{}".format(jobnr, 21002)
    dag.add(name = "JOB_"+str(jobnr), params = 'FILENAMELIST="%s" WRITE="%s" BATCHNAME="%s"'%(loadfilestr, outfolder+batchname, batchname))
dag.write("/home/pfuerst/master_thesis/software/combienergy/condor/extract21002.dag")