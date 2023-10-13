#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=8G                        # memory per node
#SBATCH --time=00-04:30         # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Cats_3

source $HOME/projects/rrg-lplevass/mjybarth/supermage_env/bin/activate
python $HOME/scratch/train_score_script.py \
    --batchsize=64\
    --epochs=100\
