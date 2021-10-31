"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import torch
from models import *
from datasets import *
from torchvision import transforms


def get_features_from_dataset(dataset, model):
    features_map = {
        'conv1': torch.FloatTensor(len(dataset), 9216),
        'res2': torch.FloatTensor(len(dataset), 9216),
        'res3': torch.FloatTensor(len(dataset), 8192),
        'res4': torch.FloatTensor(len(dataset), 9216),
        'res5': torch.FloatTensor(len(dataset), 8192)
    }
    targets = torch.LongTensor(len(dataset))

    dataloader = torch.utils.data.DataLoader(dataset, num_workers=16,
            batch_size=256, pin_memory=True,
            drop_last=False, shuffle=False)

    print(len(dataset))
    ptr = 0
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            print(f'Processed {ptr} images so far!')

            images = batch['image'].cuda(non_blocking=True)
            cur_targets = batch['target'].cuda(non_blocking=True)
            output = model(images)
            
            b = output['conv1'].shape[0]
            
            assert(b + ptr <= len(dataset))
            
            features_map['conv1'][ptr:ptr+b].copy_(output['conv1'].detach())
            features_map['res2'][ptr:ptr+b].copy_(output['res2'].detach())
            features_map['res3'][ptr:ptr+b].copy_(output['res3'].detach())
            features_map['res4'][ptr:ptr+b].copy_(output['res4'].detach())
            features_map['res5'][ptr:ptr+b].copy_(output['res5'].detach())
            targets[ptr:ptr+b].copy_(cur_targets.detach())
            ptr += b

    return features_map, targets


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
        #train_dataset = ImageNet('/scratch0/mgwillia/imagenet/', split='train', transform=transform)
        val_dataset = ImageNet('/scratch0/mgwillia/imagenet/', split='val', transform=transform)
    elif args.dataset == 'imagenet_50':
        #train_dataset = ImageNetSubset(f'/cfarhomes/mgwillia/scan-adaptation/Unsupervised-Classification/data/imagenet_subsets/{args.dataset}.txt', '/scratch0/mgwillia/imagenet/', split='train', transform=transform)
        val_dataset = ImageNetSubset(f'/cfarhomes/mgwillia/scan-adaptation/Unsupervised-Classification/data/imagenet_subsets/{args.dataset}.txt', '/scratch0/mgwillia/imagenet/', split='val', transform=transform)
    elif args.dataset == 'imagenet_200':
        #train_dataset = ImageNetSubset(f'/cfarhomes/mgwillia/scan-adaptation/Unsupervised-Classification/data/imagenet_subsets/{args.dataset}.txt', '/scratch0/mgwillia/imagenet/', split='train', transform=transform)
        val_dataset = ImageNetSubset(f'/cfarhomes/mgwillia/scan-adaptation/Unsupervised-Classification/data/imagenet_subsets/{args.dataset}.txt', '/scratch0/mgwillia/imagenet/', split='val', transform=transform)
    elif args.dataset == 'cub':
        #train_dataset = ImageNet('/scratch0/mgwillia/CUB_200_2011/', split='train', transform=transform)
        val_dataset = ImageNet('/scratch0/mgwillia/CUB_200_2011/', split='val', transform=transform)
    elif args.dataset == 'cars':
        #train_dataset = ImageNet('/scratch0/mgwillia/StanfordCars/', split='train', transform=transform)
        val_dataset = ImageNet('/scratch0/mgwillia/StanfordCars/', split='val', transform=transform)
    elif args.dataset == 'dogs':
        #train_dataset = ImageNet('/scratch0/mgwillia/StanfordDogs/', split='train', transform=transform)
        val_dataset = ImageNet('/scratch0/mgwillia/StanfordDogs/', split='val', transform=transform)
    elif args.dataset == 'flowers':
        #train_dataset = ImageNet('/scratch0/mgwillia/OxfordFlowers/', split='train', transform=transform)
        val_dataset = ImageNet('/scratch0/mgwillia/OxfordFlowers/', split='val', transform=transform)
    elif args.dataset == 'aircraft':
        #train_dataset = ImageNet('/scratch0/mgwillia/fgvc-aircraft-2013b/', split='train', transform=transform)
        val_dataset = ImageNet('/scratch0/mgwillia/fgvc-aircraft-2013b/', split='val', transform=transform)
    elif args.dataset == 'nabirds':
        #train_dataset = ImageNet('/scratch0/mgwillia/nabirds/', split='train', transform=transform)
        val_dataset = ImageNet('/scratch0/mgwillia/nabirds/', split='val', transform=transform)
    else:
        raise ValueError(f'Invalid dataset: {args.dataset}')

    ### SET UP MODEL ###
    model = resnet50_cka()
    
    saved_model = torch.load('/vulcanscratch/mgwillia/unsupervised-classification/backbones/' + args.backbone + '.pth.tar', map_location='cpu')
    missing = model.load_state_dict(saved_model, strict=False)
    print(missing)
    model = torch.nn.DataParallel(model)
    model.cuda()
    model.eval()

    features = get_features_from_dataset(val_dataset, model)
    torch.save(features, '/vulcanscratch/mgwillia/vissl/cka_features/' + '_'.join([args.backbone, args.dataset, 'features']) + '.pth.tar')


if __name__ == '__main__':
    main()
