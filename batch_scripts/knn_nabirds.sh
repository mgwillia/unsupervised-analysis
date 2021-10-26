#!/bin/bash

#SBATCH --job-name=knn_nabirds
#SBATCH --output=outfiles/knn_nabirds.out.%j
#SBATCH --error=outfiles/knn_nabirds.out.%j
#SBATCH --time=36:00:00
#SBATCH --qos=high
#SBATCH --gres=gpu:p6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module load cuda/11.0.3

DATASET="nabirds"
BACKBONES=("btwins_r50_1000" "dcv2_r50_800" "moco_r50_800" "simclr_r50_200" "simsiam_r50_100" "supervised_r50" "swav_r50_800")
NUM_NEIGHBORS=(5 10 15 20 25 30 35 40 45 50)

srun bash -c "hostname;"
for backbone in ${BACKBONES[@]}; do
    for i in ${NUM_NEIGHBORS[@]}; do
        srun bash -c "python experiments/knn_classifier.py --backbone $backbone --dataset $DATASET --temperature 0.1 --num-neighbors $i --normalize;"
    done
done
