#!/bin/bash

#SBATCH --job-name=knn_cub
#SBATCH --output=outfiles/knn_cub.out.%j
#SBATCH --error=outfiles/knn_cub.out.%j
#SBATCH --time=36:00:00
#SBATCH --qos=high
#SBATCH --gres=gpu:p6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

module load cuda/11.0.3

DATASET="cub"
BACKBONES=("btwins_r50_1000" "dcv2_r50_800" "moco_r50_800" "simclr_r50_200" "simsiam_r50_100" "supervised_r50" "swav_r50_800")

srun bash -c "hostname;"
for backbone in ${BACKBONES[@]}; do
    srun bash -c "python experiments/knn_classifier.py --backbone $backbone --dataset $DATASET --temperature 0.1 --num-neighbors 10 --normalize;"
done