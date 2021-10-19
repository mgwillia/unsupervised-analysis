""" 
    NeighborsDataset
    Returns an image with one of its neighbors.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class NeighborsDataset(Dataset):
    def __init__(self, dataset, indices, num_neighbors=None):
        super(NeighborsDataset, self).__init__()

        self.dataset = dataset
        self.indices = indices
        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors+1]
        print(self.indices.shape, len(self.dataset))
        assert(self.indices.shape[0] == len(self.dataset))


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)
        
        neighbor_index = np.random.choice(self.indices[index], 1)[0]
        neighbor = self.dataset.__getitem__(neighbor_index)

        output['anchor_features'] = anchor['features']
        output['neighbor_features'] = neighbor['features'] 
        output['possible_neighbors'] = torch.from_numpy(self.indices[index])
        output['target'] = anchor['target']
        
        return output
