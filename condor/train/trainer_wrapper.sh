#!/bin/bash
set -e
echo "TRAINER_WRAPPER: entering venv..."
source /home/pfuerst/venvs/venv_py3-v4.1.1/bin/activate 
python -c "import numpy; print(numpy)"
python -c "import sklearn; print(sklearn)"
echo "TRAINER_WRAPPER: starting training..."
python /home/pfuerst/master_thesis/software/combienergy/scripts/trainer.py $@
echo "TRAINER_WRAPPER: finished training."

