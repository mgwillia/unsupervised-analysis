"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import torch
from datasets.color_jitter import ColorJitterDataset
from models import *
from datasets import *
import torch.nn.functional as F


def get_features_from_dataset(dataset, backbone, dim):
    features = torch.FloatTensor(len(dataset), dim)
    aug_features = torch.FloatTensor(len(dataset), dim)

    dataloader = torch.utils.data.DataLoader(dataset, num_workers=8,
            batch_size=64, pin_memory=True, drop_last=False, shuffle=False)

    print(len(dataset))
    ptr = 0
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            print(f'Processed {ptr} images so far!')

            images = batch['image'].cuda(non_blocking=True)
            aug_images = batch['aug_image'].cuda(non_blocking=True)
            output = backbone(images)
            aug_output = backbone(aug_images)
            
            b = output.size(0)
            
            assert(b + ptr <= len(dataset))
            
            features[ptr:ptr+b].copy_(output.detach())
            aug_features[ptr:ptr+b].copy_(aug_output.detach())
            ptr += b

    norm_features = F.normalize(features, p=2, dim=1)
    norm_aug_features = F.normalize(aug_features, p=2, dim=1)

    return norm_features, norm_aug_features


def main():

    ### SET UP PARSER ###
    parser = argparse.ArgumentParser(description='Get Features')
    parser.add_argument('--backbone', default='', type=str, help='backbone path')
    parser.add_argument('--dataset', default='imagenet', type=str, help='datset name: cub or imagenet')
    parser.add_argument('--transform', default='blur', type=str, help='transform name: blur or color')
    args = parser.parse_args()

    print(f'Getting features for backbone path: {args.backbone}, dataset: {args.dataset}')

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    ### SET UP DATASETS ###
    if args.dataset == 'imagenet':
        train_dataset = ImageNet('/scratch0/mgwillia/imagenet/', split='train', transform=None)
        val_dataset = ImageNet('/scratch0/mgwillia/imagenet/', split='val', transform=None)
    elif args.dataset == 'imagenet_50':
        train_dataset = ImageNetSubset(f'/cfarhomes/mgwillia/scan-adaptation/Unsupervised-Classification/data/imagenet_subsets/{args.dataset}.txt', '/scratch0/mgwillia/imagenet/', split='train', transform=None)
        val_dataset = ImageNetSubset(f'/cfarhomes/mgwillia/scan-adaptation/Unsupervised-Classification/data/imagenet_subsets/{args.dataset}.txt', '/scratch0/mgwillia/imagenet/', split='val', transform=None)
    elif args.dataset == 'imagenet_200':
        train_dataset = ImageNetSubset(f'/cfarhomes/mgwillia/scan-adaptation/Unsupervised-Classification/data/imagenet_subsets/{args.dataset}.txt', '/scratch0/mgwillia/imagenet/', split='train', transform=None)
        val_dataset = ImageNetSubset(f'/cfarhomes/mgwillia/scan-adaptation/Unsupervised-Classification/data/imagenet_subsets/{args.dataset}.txt', '/scratch0/mgwillia/imagenet/', split='val', transform=None)
    elif args.dataset == 'cub':
        train_dataset = CUB('/fs/vulcan-datasets/CUB/CUB_200_2011/', train=True, transform=None)
        val_dataset = CUB('/fs/vulcan-datasets/CUB/CUB_200_2011/', train=False, transform=None)
    elif args.dataset == 'cars':
        train_dataset = ImageNet('/vulcanscratch/mgwillia/StanfordCars/', split='train', transform=None)
        val_dataset = ImageNet('/vulcanscratch/mgwillia/StanfordCars/', split='val', transform=None)
    elif args.dataset == 'dogs':
        train_dataset = StanfordDogs('/scratch0/mgwillia/StanfordDogs/', train=True, transform=None)
        val_dataset = StanfordDogs('/scratch0/mgwillia/StanfordDogs/', train=False, transform=None)
    elif args.dataset == 'flowers':
        train_dataset = OxfordFlowers('/scratch0/mgwillia/OxfordFlowers/', train=True, transform=None)
        val_dataset = OxfordFlowers('/scratch0/mgwillia/OxfordFlowers/', train=False, transform=None)
    else:
        raise ValueError(f'Invalid dataset: {args.dataset}')
    
    if args.transform == 'blur':
        train_dataset = BlurDataset(train_dataset)
        val_dataset = BlurDataset(val_dataset)
    elif args.transform == 'color':
        train_dataset = ColorJitterDataset(train_dataset)
        val_dataset = ColorJitterDataset(val_dataset)
    else:
        raise ValueError(f'Invalid transform: {args.transform}')

    ### SET UP MODEL ###
    model = resnet50x1()
    
    backbone = model['backbone']
    saved_model = torch.load('/vulcanscratch/mgwillia/unsupervised-classification/backbones/' + args.backbone + '.pth.tar', map_location='cpu')
    missing = backbone.load_state_dict(saved_model, strict=False)
    print(missing)
    backbone = torch.nn.DataParallel(backbone)
    backbone.cuda()
    backbone.eval()

    train_features, train_aug_features = get_features_from_dataset(train_dataset, backbone, model['dim'])
    val_features, val_aug_features = get_features_from_dataset(val_dataset, backbone, model['dim'])

    train_embedding_similarities = torch.bmm(torch.unsqueeze(train_features, 1), torch.unsqueeze(train_aug_features, 2))
    val_embedding_similarities = torch.bmm(torch.unsqueeze(val_features, 1), torch.unsqueeze(val_aug_features, 2))

    print(f'Mean train, val similarities: {train_embedding_similarities.mean()}, {val_embedding_similarities.mean()}')

    embedding_similarities = {
        'train_similarities': train_embedding_similarities,
        'val_similarities': val_embedding_similarities
    }

    torch.save(embedding_similarities, '/vulcanscratch/mgwillia/vissl/features/' + '_'.join([args.backbone, args.dataset, args.transform, 'similarities']) + '.pth.tar')


if __name__ == '__main__':
    main()
