#!/bin/bash

#SBATCH --job-name=kmeans_flwrs
#SBATCH --output=outfiles/kmeans_flwrs.out.%j
#SBATCH --error=outfiles/kmeans_flwrs.out.%j
#SBATCH --time=72:00:00
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

module load cuda/11.0.3

DATASET="flowers"
BACKBONES_DIR="/vulcanscratch/mgwillia/unsupervised-classification/backbones"
BACKBONES=("btwins_r50_300" "btwins_r50_1000" "dcv2_r50_400" "dcv2_r50_400_special" "dcv2_r50_800" "jigsaw_r50_100" "moco_r50_200" "pirl_r50_200" "simclr_r50_200" "simclr_r50x2_1000" "supervised_r50" "swav_r50_200" "swav_r50_800" "swav_r50x2_400")

srun bash -c "hostname;"
for backbone in ${BACKBONES[@]}; do
    srun bash -c "python kmeans.py --backbone $BACKBONES_DIR/$backbone --dataset $DATASET --num-clusters 102;"
done
