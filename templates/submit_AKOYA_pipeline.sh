#!/usr/bin/env bash

#BSUB -G team283
#BSUB -q "hugemem"
#BSUB -n 1
#BSUB -M 400000
#BSUB -R "select[mem>400000] rusage[mem=400000]"
#BSUB -o "logs/output%J.log"
#BSUB -e "logs/error%J.log"


module load ISG/conda
conda activate ./sp_env
export PYTHONNOUSERSITE=1
python bin/run_AKOYA.py conf_files/conf_BK22-SKI-27-FO-1-S34-D1.yaml
