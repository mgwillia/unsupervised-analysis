#!/bin/bash

#SBATCH --job-name=compare_cub
#SBATCH --output=outfiles/compare_cub.out.%j
#SBATCH --error=outfiles/compare_cub.out.%j
#SBATCH --time=36:00:00
#SBATCH --qos=high
#SBATCH --gres=gpu:gtx1080ti:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module load cuda/11.0.3

DATASET="cub"
BACKBONES=("btwins_r50_1000" "dcv2_r50_800" "moco_r50_800" "simclr_r50_200" "simsiam_r50_100" "supervised_r50" "swav_r50_800")

srun bash -c "hostname;"
for backbone in ${BACKBONES[@]}; do
    srun bash -c "python compare_on_transforms.py --backbone $backbone --dataset $DATASET --transform blur;"
    srun bash -c "python compare_on_transforms.py --backbone $backbone --dataset $DATASET --transform color;"
done
