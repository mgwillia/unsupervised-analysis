#!/bin/bash

#SBATCH --job-name=cka_feat_cub
#SBATCH --output=outfiles/cka_feat_cub.out.%j
#SBATCH --error=outfiles/cka_feat_cub.out.%j
#SBATCH --time=36:00:00
#SBATCH --qos=high
#SBATCH --gres=gpu:gtx1080ti:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

module load cuda/11.0.3

DATASET="cub"
BACKBONES=("btwins_r50_1000" "dcv2_r50_800" "moco_r50_800" "simclr_r50_800" "simsiam_r50_100" "supervised_r50" "swav_r50_800" "simclr_r50_100" "simclr_r50_200" "simclr_r50_400" "simclr_r50_1000")

if [ ! -d /scratch0/mgwillia/CUB_200_2011 ]; then
    srun bash -c "echo 'cub not found on scratch!'"
    srun bash -c "mkdir -p /scratch0/mgwillia"
    srun bash -c "./msrsync -p 16 /vulcanscratch/mgwillia/CUB_200_2011 /scratch0/mgwillia/"
fi

srun bash -c "hostname;"
for backbone in ${BACKBONES[@]}; do
    srun bash -c "python get_cka_features.py --backbone $backbone --dataset $DATASET;"
done
