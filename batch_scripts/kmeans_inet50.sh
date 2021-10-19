#!/bin/bash

#SBATCH --job-name=kmeans_i50
#SBATCH --output=outfiles/kmeans_i50.out.%j
#SBATCH --error=outfiles/kmeans_i50.out.%j
#SBATCH --time=72:00:00
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

module load cuda/11.0.3

DATASET="imagenet_50"
BACKBONES_DIR="/vulcanscratch/mgwillia/unsupervised-classification/backbones"
#BACKBONES=("btwins_r50_300" "btwins_r50_1000" "dcv2_r50_400" "dcv2_r50_400_special" "dcv2_r50_800" "jigsaw_r50_100" "moco_r50_200" "pirl_r50_200" "simclr_r50_200" "simclr_r50x2_1000" "supervised_r50" "swav_r50_200" "swav_r50_800" "swav_r50x2_400")
BACKBONES=("mocoscanv1f1k_r50_20" "mocoscanv1f128_r50_20" "moco_r50_820" "mocoscanf128_r50_20" "scan_with_fc" "moco_r50_200" "moco_r50_800" "supervised_r50" "swav_r50_800")

srun bash -c "hostname;"
for backbone in ${BACKBONES[@]}; do
    srun bash -c "python kmeans.py --backbone $BACKBONES_DIR/$backbone --dataset $DATASET --num-clusters 50;"
done
