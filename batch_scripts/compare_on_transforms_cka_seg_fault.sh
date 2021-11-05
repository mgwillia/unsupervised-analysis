#!/bin/bash

#SBATCH --job-name=compare_transforms_cka_seg_fault
#SBATCH --output=outfiles/compare_transforms_cka_seg_fault.out.%j
#SBATCH --error=outfiles/compare_transforms_cka_seg_fault.out.%j
#SBATCH --time=36:00:00
#SBATCH --qos=high
#SBATCH --gres=gpu:gtx1080ti:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

module load cuda/11.0.3

DATASETS=("imagenet")
BACKBONES=("supervised_r50")
TRANSFORMS=("rotate")

srun bash -c "hostname;"
for dataset in ${DATASETS[@]}; do
    for backbone in ${BACKBONES[@]}; do
        for transform in ${TRANSFORMS[@]}; do
            srun bash -c "python compare_on_transforms_cka.py --dataset $dataset --backbone $backbone --transform $transform;"
        done
    done
done
