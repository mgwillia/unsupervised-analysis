""" 
Some code from Unsupservised Classifcation, maybe. Lots of modifications.
"""
from torch import nn


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim):
        super(ProjectionHead, self).__init__()

        head_layers = [nn.Linear(embedding_dim, embedding_dim, bias=True)]
        head_layers.append(nn.BatchNorm1d(
                        embedding_dim,
                        eps=1e-5,
                        momentum=0.1,
                    ))
        head_layers.append(nn.ReLU(inplace=True))
        head_layers.append(nn.Linear(embedding_dim, projection_dim, bias=True))
        self.projection_head = nn.Sequential(*head_layers)

    def forward(self, features):
        projections = self.projection_head(features)
        return projections