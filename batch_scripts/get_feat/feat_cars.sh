#!/bin/bash

#SBATCH --job-name=feat_cars
#SBATCH --output=outfiles/feat_cars.out.%j
#SBATCH --error=outfiles/feat_cars.out.%j
#SBATCH --time=72:00:00
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

module load cuda/11.0.3

DATASET="cars"
#BACKBONES=("btwins_r50_1000" "dcv2_r50_800" "moco_r50_800" "simclr_r50_800" "simsiam_r50_100" "supervised_r50" "swav_r50_800")
BACKBONES=("simclr_r50_100" "simclr_r50_200" "simclr_r50_400" "simclr_r50_800" "simclr_r50_1000")

if [ ! -d /scratch0/mgwillia/StanfordCars ]; then
    srun bash -c "echo 'cars not found on scratch!'"
    srun bash -c "mkdir -p /scratch0/mgwillia"
    srun bash -c "./msrsync -p 16 /vulcanscratch/mgwillia/StanfordCars /scratch0/mgwillia/"
fi

srun bash -c "hostname;"
for backbone in ${BACKBONES[@]}; do
    srun bash -c "python get_features.py --backbone $backbone --dataset $DATASET;"
done
