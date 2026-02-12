#!/usr/bin/env bash

#BSUB -G XXX
#BSUB -q "normal"
#BSUB -n 1
#BSUB -M 200000
#BSUB -R "select[mem>200000] rusage[mem=200000]"
#BSUB -o "logs/output%J.log"
#BSUB -e "logs/error%J.log"


module load ISG/conda
conda activate sp_env
export PYTHONNOUSERSITE=1
python run_AKOYA.py conf_AKOYA.yaml
