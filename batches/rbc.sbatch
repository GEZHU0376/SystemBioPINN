#!/bin/bash
#SBATCH -p a6000-gcondo --gres=gpu:1
# Request 1 CPU core
#SBATCH -n 1
#SBATCH -t 60:00:00
#SBATCH --mem=40G
#SBATCH -o job_test%j.out
#SBATCH -e job_test%j.err

module load python/3.7.4 cuda/11.1.1
source /users/zzhang99/apps/amp_pytorch/bin/activate

cd ..
# Run program
python3 -u example_rbc.py
