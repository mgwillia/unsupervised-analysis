#!/bin/bash

#SBATCH --job-name=kmeans_inet_3k
#SBATCH --output=outfiles/kmeans_inet_3k.out.%j
#SBATCH --error=outfiles/kmeans_inet_3k.out.%j
#SBATCH --time=72:00:00
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G

DATASET="imagenet3k"
BACKBONES=("swav_r50_800")

srun bash -c "hostname;"
for backbone in ${BACKBONES[@]}; do
    srun bash -c "python experiments/kmeans.py --backbone $backbone --dataset $DATASET --num-clusters 3000;"
done
