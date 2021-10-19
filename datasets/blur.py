"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import ImageFilter


""" 
    NeighborsDataset
    Returns an image with one of its neighbors.
"""
class BlurDataset(Dataset):
    def __init__(self, dataset):
        super(BlurDataset, self).__init__()
        
        self.crop = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)])
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
       
        dataset.transform = None
        self.dataset = dataset
        self.blur_radius = torch.empty(len(self.dataset)).uniform_(0.1, 2.0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)

        output['image'] = self.crop(anchor['image'])
        output['aug_image'] = self.crop(anchor['image'])

        output['aug_image'] = output['aug_image'].filter(
            ImageFilter.GaussianBlur(
                radius=self.blur_radius[index]
            )
        )

        output['image'] = self.to_tensor(output['image'])
        output['aug_image'] = self.to_tensor(output['aug_image'])
        
        return output
