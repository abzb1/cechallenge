#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --ntasks-per-node=1    # Cores per node
#SBATCH --partition=hgx        # Partition Name (gpu, hgx)
##
#SBATCH --job-name=ohs-jupyter
#SBATCH -o ./log/SLURM.%N.%j.out         # STDOUT
#SBATCH -e ./log/SLURM.%N.%j.err         # STDERR
##
#SBATCH --gres=gpu:hgx:8

hostname
date

echo 'start jupyter lab!'

ssh ohs@172.16.10.36 -R 45393:localhost:45393 -fN "while sleep 100; do; done"&

/home1/ohs/anaconda3/envs/cechallenge/bin/python3 -m jupyter lab --ip=0.0.0.0 --port 45393

echo 'done!'