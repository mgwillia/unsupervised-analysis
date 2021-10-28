#!/bin/bash

#SBATCH --job-name=feat_inet
#SBATCH --output=outfiles/feat_inet.out.%j
#SBATCH --error=outfiles/feat_inet.out.%j
#SBATCH --time=72:00:00
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module load cuda/11.0.3

DATASET="imagenet"
#BACKBONES=("btwins_r50_1000" "dcv2_r50_800" "moco_r50_800" "simclr_r50_800" "simsiam_r50_100" "supervised_r50" "swav_r50_800")
BACKBONES=("simclr_r50_100" "simclr_r50_200" "simclr_r50_400" "simclr_r50_800" "simclr_r50_1000")

srun bash -c "hostname;"
for backbone in ${BACKBONES[@]}; do
    srun bash -c "python get_features.py --backbone $backbone --dataset $DATASET;"
done
