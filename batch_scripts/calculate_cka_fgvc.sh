#!/bin/bash

#SBATCH --job-name=calc_cka_fgvc
#SBATCH --output=outfiles/calc_cka_fgvc.out.%j
#SBATCH --error=outfiles/calc_cka_fgvc.out.%j
#SBATCH --time=36:00:00
#SBATCH --qos=medium
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module load cuda/11.0.3

#DATASETS=("aircraft" "cars" "cub" "dogs" "flowers" "nabirds")
DATASETS=("cub")
BACKBONES=("btwins_r50_1000" "dcv2_r50_800" "moco_r50_800" "simclr_r50_800" "simsiam_r50_100" "swav_r50_800" "simclr_r50_100" "simclr_r50_200" "simclr_r50_400" "simclr_r50_1000")

srun bash -c "hostname;"
for dataset in ${DATASETS[@]}; do
    for backbone in ${BACKBONES[@]}; do
        srun bash -c "python experiments/calculate_cka.py --backbone-a supervised_r50 --backbone-b $backbone --dataset $dataset;"
    done
done