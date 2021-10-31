#!/bin/bash

#SBATCH --job-name=compare_transforms_cka_inet
#SBATCH --output=outfiles/compare_transforms_cka_inet.out.%j
#SBATCH --error=outfiles/compare_transforms_cka_inet.out.%j
#SBATCH --time=36:00:00
#SBATCH --qos=high
#SBATCH --gres=gpu:gtx1080ti:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

module load cuda/11.0.3

DATASETS=("imagenet")
BACKBONES=("btwins_r50_1000" "dcv2_r50_800" "moco_r50_800" "simclr_r50_800" "simsiam_r50_100" "supervised_r50" "swav_r50_800")
TRANSFORMS=("image_jitter" "image_blur" "horizontal_flip" "vertical_flip" "rotate")

if [ ! -d /scratch0/mgwillia/fgvc-aircraft-2013b ]; then
    srun bash -c "echo 'aircraft not found on scratch!'"
    srun bash -c "mkdir -p /scratch0/mgwillia"
    srun bash -c "./msrsync -p 16 /vulcanscratch/mgwillia/fgvc-aircraft-2013b /scratch0/mgwillia/"
fi

if [ ! -d /scratch0/mgwillia/StanfordCars ]; then
    srun bash -c "echo 'cars not found on scratch!'"
    srun bash -c "mkdir -p /scratch0/mgwillia"
    srun bash -c "./msrsync -p 16 /vulcanscratch/mgwillia/StanfordCars /scratch0/mgwillia/"
fi

if [ ! -d /scratch0/mgwillia/CUB_200_2011 ]; then
    srun bash -c "echo 'cub not found on scratch!'"
    srun bash -c "mkdir -p /scratch0/mgwillia"
    srun bash -c "./msrsync -p 16 /vulcanscratch/mgwillia/CUB_200_2011 /scratch0/mgwillia/"
fi

if [ ! -d /scratch0/mgwillia/StanfordDogs ]; then
    srun bash -c "echo 'dogs not found on scratch!'"
    srun bash -c "mkdir -p /scratch0/mgwillia"
    srun bash -c "./msrsync -p 16 /vulcanscratch/mgwillia/StanfordDogs /scratch0/mgwillia/"
fi

if [ ! -d /scratch0/mgwillia/OxfordFlowers ]; then
    srun bash -c "echo 'flowers not found on scratch!'"
    srun bash -c "mkdir -p /scratch0/mgwillia"
    srun bash -c "./msrsync -p 16 /vulcanscratch/mgwillia/OxfordFlowers /scratch0/mgwillia/"
fi

if [ ! -d /scratch0/mgwillia/nabirds ]; then
    srun bash -c "echo 'nabirds not found on scratch!'"
    srun bash -c "mkdir -p /scratch0/mgwillia"
    srun bash -c "./msrsync -p 16 /vulcanscratch/mgwillia/nabirds /scratch0/mgwillia/"
fi

srun bash -c "hostname;"
for dataset in ${DATASETS[@]}; do
    for backbone in ${BACKBONES[@]}; do
        for transform in ${TRANSFORMS[@]}; do
            srun bash -c "python compare_on_transforms.py --dataset $dataset --backbone $backbone --transform $transform;"
        done
    done
done
