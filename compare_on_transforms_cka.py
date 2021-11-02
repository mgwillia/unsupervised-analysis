"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import torch
from models import *
from datasets import *
import torch.nn.functional as F


class CudaCKA(object):
    def __init__(self, device):
        self.device = device
    
    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)


def get_features_from_dataset(dataset, backbone, dim):
    features = torch.FloatTensor(len(dataset), dim)
    aug_features = torch.FloatTensor(len(dataset), dim)

    dataloader = torch.utils.data.DataLoader(dataset, num_workers=16,
            batch_size=256, pin_memory=True, drop_last=False, shuffle=False)

    for param in backbone.parameters():
        param.requires_grad = False
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
        train_dataset = ImageNet('/scratch0/mgwillia/CUB_200_2011/', split='train', transform=None)
        val_dataset = ImageNet('/scratch0/mgwillia/CUB_200_2011/', split='val', transform=None)
    elif args.dataset == 'cars':
        train_dataset = ImageNet('/scratch0/mgwillia/StanfordCars/', split='train', transform=None)
        val_dataset = ImageNet('/scratch0/mgwillia/StanfordCars/', split='val', transform=None)
    elif args.dataset == 'dogs':
        train_dataset = ImageNet('/scratch0/mgwillia/StanfordDogs/', split='train', transform=None)
        val_dataset = ImageNet('/scratch0/mgwillia/StanfordDogs/', split='val', transform=None)
    elif args.dataset == 'flowers':
        train_dataset = ImageNet('/scratch0/mgwillia/OxfordFlowers/', split='train', transform=None)
        val_dataset = ImageNet('/scratch0/mgwillia/OxfordFlowers/', split='val', transform=None)
    elif args.dataset == 'aircraft':
        train_dataset = ImageNet('/scratch0/mgwillia/fgvc-aircraft-2013b/', split='train', transform=None)
        val_dataset = ImageNet('/scratch0/mgwillia/fgvc-aircraft-2013b/', split='val', transform=None)
    elif args.dataset == 'nabirds':
        train_dataset = ImageNet('/scratch0/mgwillia/nabirds/', split='train', transform=None)
        val_dataset = ImageNet('/scratch0/mgwillia/nabirds/', split='val', transform=None)
    else:
        raise ValueError(f'Invalid dataset: {args.dataset}')
    
    if args.transform not in ['image_jitter', 'patch_jitter', 'image_blur', 'patch_blur', 'horizontal_flip', 'vertical_flip', 'rotate']:
        raise ValueError(f'Invalid transform: {args.transform}')
    else:
        train_dataset = TransformsDataset(train_dataset, args.transform)
        val_dataset = TransformsDataset(val_dataset, args.transform)

    ### SET UP MODEL ###
    model = resnet50x1()
    
    backbone = model['backbone']
    saved_model = torch.load('/vulcanscratch/mgwillia/unsupervised-classification/backbones/' + args.backbone + '.pth.tar', map_location='cpu')
    missing = backbone.load_state_dict(saved_model, strict=False)
    print(missing)
    backbone = torch.nn.DataParallel(backbone)
    backbone.cuda()
    backbone.eval()

    val_features, val_aug_features = get_features_from_dataset(val_dataset, backbone, model['dim'])
    
    device = torch.device('cuda')
    cuda_cka = CudaCKA(device)

    cur_CKA = cuda_cka.linear_CKA(val_features[::5].to(device), val_aug_features[::5].to(device))

    print(f'Dataset: {args.dataset}, Transform: {args.transform}, Backbone: {args.backbone}, Linear_CKA: {cur_CKA}', flush=True)


if __name__ == '__main__':
    main()
