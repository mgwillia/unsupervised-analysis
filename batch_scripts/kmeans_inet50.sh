#!/bin/bash

#SBATCH --job-name=kmeans_i50
#SBATCH --output=outfiles/kmeans_i50.out.%j
#SBATCH --error=outfiles/kmeans_i50.out.%j
#SBATCH --time=72:00:00
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G

DATASET="imagenet_50"
BACKBONES=("btwins_r50_1000" "dcv2_r50_800" "moco_r50_800" "simclr_r50_200" "simsiam_r50_100" "supervised_r50" "swav_r50_800")

srun bash -c "hostname;"
for backbone in ${BACKBONES[@]}; do
    srun bash -c "python kmeans.py --backbone $backbone --dataset $DATASET --num-clusters 50;"
done
