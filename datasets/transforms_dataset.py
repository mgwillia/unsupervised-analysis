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
    TransformsDataset
    Returns an image with one of its neighbors.
"""
class TransformsDataset(Dataset):
    def __init__(self, dataset, transform_name):
        super(TransformsDataset, self).__init__()
        
        self.crop = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)])
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform_name = transform_name

        self.horizontal_flip = transforms.RandomHorizontalFlip(1.0)
        self.vertical_flip = transforms.RandomVerticalFlip(1.0)
        self.rotate = transforms.RandomRotation((90, 90))
       
        dataset.transform = None
        self.dataset = dataset
        self.blur_radius = torch.empty(len(self.dataset)).uniform_(0.1, 2.0)
        self.fn_idxs = []
        for _ in range(len(self.dataset)):
            self.fn_idxs.append(torch.randperm(4))
        self.bcs_factors = torch.empty((len(self.dataset), 3)).uniform_(0.6, 1.4)
        self.hue_factors = torch.empty(len(self.dataset)).uniform_(-0.1, 0.1)
        self.patch_coords = torch.randint(0, 168, (len(self.dataset), 2))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)

        output['image'] = self.crop(anchor['image'])
        output['aug_image'] = self.crop(anchor['image'])
        
        if self.transform_name == 'image_jitter':
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
        elif self.transform_name == 'image_blur':
            output['aug_image'] = output['aug_image'].filter(
                ImageFilter.GaussianBlur(
                    radius=self.blur_radius[index]
                )
            )
        elif self.transform_name == 'patch_blur':
            print('shape:', output['aug_image'].shape)
            x, y = self.patch_coords[index]
            width, height =  56, 56
            output['aug_image'][x:x+width,y:y+height] = output['aug_image'][x:x+width,y:y+height].filter(
                ImageFilter.GaussianBlur(
                    radius=self.blur_radius[index]
                )
            )
        elif self.transform_name == 'horizontal_flip':
            output['aug_image'] = self.horizontal_flip(output['aug_image'])
        elif self.transform_name == 'vertical_flip':
            output['aug_image'] = self.vertical_flip(output['aug_image'])
        elif self.transform_name == 'rotate':
            output['aug_image'] = self.rotate(output['aug_image'])

        output['image'] = self.to_tensor(output['image'])
        output['aug_image'] = self.to_tensor(output['aug_image'])

        if self.transform_name == 'patch_jitter':
            print('shape:', output['aug_image'].shape)
            x, y = self.patch_coords[index]
            width, height =  56, 56
            fn_idx = self.fn_idxs[index]
            b, c, s = self.bcs_factors[index]
            h = self.hue_factors[index]
            for fn_id in fn_idx:
                if fn_id == 0:
                    output['aug_image'][x:x+width,y:y+height] = TF.adjust_brightness(output['aug_image'][x:x+width,y:y+height], float(b))
                elif fn_id == 1:
                    output['aug_image'][x:x+width,y:y+height] = TF.adjust_contrast(output['aug_image'][x:x+width,y:y+height], float(c))
                elif fn_id == 2:
                    output['aug_image'][x:x+width,y:y+height] = TF.adjust_saturation(output['aug_image'][x:x+width,y:y+height], float(s))
                elif fn_id == 3:
                    output['aug_image'][x:x+width,y:y+height] = TF.adjust_hue(output['aug_image'][x:x+width,y:y+height], float(h))
        
        output['image'] = self.normalize(output['image'])
        output['aug_image'] = self.normalize(output['aug_image'])
        
        return output
