""" 
Some code from Unsupservised Classifcation, maybe. Lots of modifications.
"""
from torch import nn


class ClusteringModel(nn.Module):
    def __init__(self, embedding_dim, num_clusters, num_heads=1):
        super(ClusteringModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        assert(isinstance(self.num_heads, int))
        assert(self.num_heads > 0)
        self.cluster_head = nn.ModuleList([nn.Linear(self.embedding_dim, num_clusters) for _ in range(self.num_heads)])

    def forward(self, x):
        out = [cluster_head(x) for cluster_head in self.cluster_head]

        return out