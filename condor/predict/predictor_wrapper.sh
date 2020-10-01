#!/bin/bash
set -e
echo "PREDICTOR_WRAPPER: entering venv..."
source /home/pfuerst/venvs/venv_py3-v4.1.1/bin/activate 
echo "PREDICTOR_WRAPPER: starting training..."
python /home/pfuerst/master_thesis/software/combienergy/scripts/predictor.py $@
echo "PREDICTOR_WRAPPER: finished training."