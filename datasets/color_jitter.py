"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


""" 
    NeighborsDataset
    Returns an image with one of its neighbors.
"""
class ColorJitterDataset(Dataset):
    def __init__(self, dataset):
        super(ColorJitterDataset, self).__init__()
        
        self.crop = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)])
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
       
        dataset.transform = None
        self.dataset = dataset
        self.fn_idxs = []
        for _ in range(len(self.dataset)):
            self.fn_idxs.append(torch.randperm(4))
        self.bcs_factors = torch.empty((len(self.dataset), 3)).uniform_(0.6, 1.4)
        self.hue_factors = torch.empty(len(self.dataset)).uniform_(-0.1, 0.1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)

        output['image'] = self.crop(anchor['image'])
        output['aug_image'] = self.crop(anchor['image'])
        
        fn_idx = self.fn_idxs[index]
        b, c, s = self.bcs_factors[index]
        h = self.hue_factors[index]
        for fn_id in fn_idx:
            if fn_id == 0:
                output['aug_image'] = TF.adjust_brightness(output['aug_image'], float(b))
            elif fn_id == 1:
                output['aug_image'] = TF.adjust_contrast(output['aug_image'], float(c))
            elif fn_id == 2:
                output['aug_image'] = TF.adjust_saturation(output['aug_image'], float(s))
            elif fn_id == 3:
                output['aug_image'] = TF.adjust_hue(output['aug_image'], float(h))


        output['image'] = self.to_tensor(output['image'])
        output['aug_image'] = self.to_tensor(output['aug_image'])
        
        return output
