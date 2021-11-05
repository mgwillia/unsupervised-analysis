#!/bin/bash

#SBATCH --job-name=cka_feat_inet_ablation
#SBATCH --output=outfiles/cka_feat_inet_ablation.out.%j
#SBATCH --error=outfiles/cka_feat_inet_ablation.out.%j
#SBATCH --time=36:00:00
#SBATCH --qos=high
#SBATCH --gres=gpu:gtx1080ti:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

module load cuda/11.0.3

DATASET="imagenet"
BACKBONES=("dcv2_r50_400" "dcv2_r50_400_special" "swav_r50_400" "swav_r50_400_lesscrop" "swav_r50_400_smallbatch")

if [ ! -d /scratch0/mgwillia/imagenet ]; then
    srun bash -c "echo 'imagenet not found on scratch!'"
    srun bash -c "mkdir -p /scratch0/mgwillia"
    srun bash -c "./msrsync -p 16 /vulcanscratch/mgwillia/imagenet /scratch0/mgwillia/"
fi

srun bash -c "hostname;"
for backbone in ${BACKBONES[@]}; do
    srun bash -c "python get_cka_features.py --backbone $backbone --dataset $DATASET;"
done
