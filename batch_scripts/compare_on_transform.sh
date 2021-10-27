#!/bin/bash

#SBATCH --job-name=compare_transforms
#SBATCH --output=outfiles/compare_transforms.out.%j
#SBATCH --error=outfiles/compare_transforms.out.%j
#SBATCH --time=36:00:00
#SBATCH --qos=high
#SBATCH --gres=gpu:gtx1080ti:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

module load cuda/11.0.3

DATASETS=("aircraft" "cars" "cub" "dogs" "flowers" "nabirds" "imagenet")
BACKBONES=("btwins_r50_1000" "dcv2_r50_800" "moco_r50_800" "simclr_r50_200" "simsiam_r50_100" "supervised_r50" "swav_r50_800")
TRANSFORMS=("image_jitter" "patch_jitter" "image_blur" "patch_blur" "horizontal_flip" "vertical_flip" "rotate")

srun bash -c "hostname;"
for dataset in ${DATASETS[@]}; do
    for backbone in ${BACKBONES[@]}; do
        for transform in ${TRANSFORMS[@]}; do
            srun bash -c "python compare_on_transforms.py --dataset $dataset --backbone $backbone --transform $transform;"
        done
    done
done
