#!/bin/bash
set -e
echo "ICETRAY_WRAPPER: loading icetray..."
/cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/icetray-env combo/V01-00-02 /home/pfuerst/master_thesis/software/combienergy/condor/predict/predictor_wrapper.sh $@
echo "ICETRAY_WRAPPER: ending..."