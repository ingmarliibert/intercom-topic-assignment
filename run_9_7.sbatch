#!/bin/bash
#SBATCH -J myjob10
#SBATCH -N 4
#SBATCH --ntasks-per-node=8
#SBATCH --mem=8192
#SBATCH -t 10:00:00


cd $HOME/mlp/
python ./train_model_w_args.py 9 0.7

