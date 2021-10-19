import torch
from torch.utils.data import Dataset


class ImageNetFeatures(Dataset):
    def __init__(self, root, backbone_name, split='train'):
        super(ImageNetFeatures, self).__init__()

        if split == 'train':
            self.features = torch.load(f'{root}/{backbone_name}_imagenet_features.pth.tar', map_location='cpu')['train_features']
            self.targets = torch.load(f'{root}/{backbone_name}_imagenet_targets.pth.tar', map_location='cpu')['train_targets']
        elif split == 'val':
            self.features = torch.load(f'{root}/{backbone_name}_imagenet_features.pth.tar', map_location='cpu')['val_features']
            self.targets = torch.load(f'{root}/{backbone_name}_imagenet_targets.pth.tar', map_location='cpu')['val_targets']

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        features = self.features[index]
        target = self.targets[index]

        out = {'features': features, 'target': target, 'index': index}

        return out
