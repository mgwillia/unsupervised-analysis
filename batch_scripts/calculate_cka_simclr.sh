#!/bin/bash

#SBATCH --job-name=calc_cka_simclr
#SBATCH --output=outfiles/calc_cka_simclr.out.%j
#SBATCH --error=outfiles/calc_cka_simclr.out.%j
#SBATCH --time=36:00:00
#SBATCH --qos=high
#SBATCH --gres=gpu:p6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module load cuda/11.0.3

DATASETS=("imagenet")
BACKBONES=("supervised_r50" "simclr_r50_100" "simclr_r50_200" "simclr_r50_400" "simclr_r50_800" "simclr_r50_1000")

srun bash -c "hostname;"
for dataset in ${DATASETS[@]}; do
    for backbone_a in ${BACKBONES[@]}; do
        for backbone_b in ${BACKBONES[@]}; do
            srun bash -c "python experiments/calculate_cka.py --backbone-a $backbone_a --backbone-b $backbone_b --dataset $dataset;"
        done
    done
done
