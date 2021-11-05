#!/bin/bash

#SBATCH --job-name=feat_inet
#SBATCH --output=outfiles/feat_inet.out.%j
#SBATCH --error=outfiles/feat_inet.out.%j
#SBATCH --time=36:00:00
#SBATCH --qos=high
#SBATCH --gres=gpu:gtx1080ti:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

module load cuda/11.0.3

DATASET="imagenet"
#BACKBONES=("btwins_r50_1000" "dcv2_r50_800" "moco_r50_800" "simclr_r50_800" "simsiam_r50_100" "supervised_r50" "swav_r50_800")
BACKBONES=("simclr_r50_100" "simclr_r50_200" "simclr_r50_400" "simclr_r50_800" "simclr_r50_1000")

srun bash -c "hostname;"
for backbone in ${BACKBONES[@]}; do
    srun bash -c "python get_features.py --backbone $backbone --dataset $DATASET;"
done
