"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import torch
from models import *
from datasets import *
from torchvision import transforms


def get_features_from_dataset(dataset, backbone, dim):
    features = torch.FloatTensor(len(dataset), dim)
    targets = torch.LongTensor(len(dataset))

    dataloader = torch.utils.data.DataLoader(dataset, num_workers=2,
            batch_size=64, pin_memory=True,
            drop_last=False, shuffle=False)

    print(len(dataset))
    ptr = 0
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            print(f'Processed {ptr} images so far!')

            images = batch['image'].cuda(non_blocking=True)
            cur_targets = batch['target'].cuda(non_blocking=True)
            output = backbone(images)
            
            b = output.size(0)
            
            assert(b + ptr <= len(dataset))
            
            features[ptr:ptr+b].copy_(output.detach())
            targets[ptr:ptr+b].copy_(cur_targets.detach())
            ptr += b

    return features, targets


def main():

    ### SET UP PARSER ###
    parser = argparse.ArgumentParser(description='Get Features')
    parser.add_argument('--backbone', default='', type=str, help='backbone path')
    parser.add_argument('--dataset', default='imagenet', type=str, help='datset name: cub or imagenet')
    args = parser.parse_args()

    print(f'Getting features for backbone path: {args.backbone}, dataset: {args.dataset}')

    ### SET UP DATASETS ###

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    if args.dataset == 'imagenet':
        train_dataset = ImageNet('/scratch0/mgwillia/imagenet/', split='train', transform=transform)
        val_dataset = ImageNet('/scratch0/mgwillia/imagenet/', split='val', transform=transform)
    elif args.dataset == 'imagenet_50':
        train_dataset = ImageNetSubset(f'/cfarhomes/mgwillia/scan-adaptation/Unsupervised-Classification/data/imagenet_subsets/{args.dataset}.txt', '/scratch0/mgwillia/imagenet/', split='train', transform=transform)
        val_dataset = ImageNetSubset(f'/cfarhomes/mgwillia/scan-adaptation/Unsupervised-Classification/data/imagenet_subsets/{args.dataset}.txt', '/scratch0/mgwillia/imagenet/', split='val', transform=transform)
    elif args.dataset == 'imagenet_200':
        train_dataset = ImageNetSubset(f'/cfarhomes/mgwillia/scan-adaptation/Unsupervised-Classification/data/imagenet_subsets/{args.dataset}.txt', '/scratch0/mgwillia/imagenet/', split='train', transform=transform)
        val_dataset = ImageNetSubset(f'/cfarhomes/mgwillia/scan-adaptation/Unsupervised-Classification/data/imagenet_subsets/{args.dataset}.txt', '/scratch0/mgwillia/imagenet/', split='val', transform=transform)
    elif args.dataset == 'cub':
        train_dataset = ImageNet('/scratch0/mgwillia/CUB_200_2011/', split='train', transform=transform)
        val_dataset = ImageNet('/scratch0/mgwillia/CUB_200_2011/', split='val', transform=transform)
    elif args.dataset == 'cars':
        train_dataset = ImageNet('/scratch0/mgwillia/StanfordCars/', split='train', transform=transform)
        val_dataset = ImageNet('/scratch0/mgwillia/StanfordCars/', split='val', transform=transform)
    elif args.dataset == 'dogs':
        train_dataset = ImageNet('/scratch0/mgwillia/StanfordDogs/', split='train', transform=transform)
        val_dataset = ImageNet('/scratch0/mgwillia/StanfordDogs/', split='val', transform=transform)
    elif args.dataset == 'flowers':
        train_dataset = ImageNet('/scratch0/mgwillia/OxfordFlowers/', split='train', transform=transform)
        val_dataset = ImageNet('/scratch0/mgwillia/OxfordFlowers/', split='val', transform=transform)
    elif args.dataset == 'aircraft':
        train_dataset = ImageNet('/scratch0/mgwillia/fgvc-aircraft-2013b/', split='train', transform=transform)
        val_dataset = ImageNet('/scratch0/mgwillia/fgvc-aircraft-2013b/', split='val', transform=transform)
    elif args.dataset == 'nabirds':
        train_dataset = ImageNet('/scratch0/mgwillia/nabirds/', split='train', transform=transform)
        val_dataset = ImageNet('/scratch0/mgwillia/nabirds/', split='val', transform=transform)
    else:
        raise ValueError(f'Invalid dataset: {args.dataset}')

    ### SET UP MODEL ###
    model = resnet50x1()
    
    backbone = model['backbone']
    saved_model = torch.load('/vulcanscratch/mgwillia/unsupervised-classification/backbones/' + args.backbone + '.pth.tar', map_location='cpu')
    missing = backbone.load_state_dict(saved_model, strict=False)
    print(missing)
    backbone = torch.nn.DataParallel(backbone)
    backbone.cuda()
    backbone.eval()

    train_features, train_targets = get_features_from_dataset(train_dataset, backbone, model['dim'])
    val_features, val_targets = get_features_from_dataset(val_dataset, backbone, model['dim'])

    backbone_name = args.backbone.split('/')[-1]

    features = {
        'train_features': train_features,
        'val_features': val_features
    }
    targets = {
        'train_targets': train_targets,
        'val_targets': val_targets
    }
    torch.save(features, '/vulcanscratch/mgwillia/vissl/features/' + '_'.join([backbone_name, args.dataset, 'features']) + '.pth.tar')
    torch.save(targets, '/vulcanscratch/mgwillia/vissl/features/' + '_'.join([backbone_name, args.dataset, 'targets']) + '.pth.tar')


if __name__ == '__main__':
    main()
