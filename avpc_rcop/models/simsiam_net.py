"""
SimSiam network, i.e., Similarity Metric, Projection and Prediction MLPs.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


def Dist(p, z, version='simplified'):  # negative cosine similarity
    if version == 'original':
        z = z.detach()   # stop gradient
        p = F.normalize(p, dim=1)   # l2-normalize
        z = F.normalize(z, dim=1)   # l2-normalize
        return -(p * z).sum(dim=1).mean()

    elif version == 'simplified':   # same thing, much faster
        return -F.cosine_similarity(p, z.detach(), dim=1).mean()

    else:
        raise Exception


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
            Projection MLP. The projection MLP (in f) has BN ap-
            plied to each fully-connected (fc) layer, including its out- 
            put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
            This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.num_layers = 3

    def forward(self, x):
        if len(x.size()) != 2:
            x = x.view(x.size(0), -1)

        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):   # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
            Prediction MLP. The prediction MLP (h) has BN applied 
            to its hidden fc layers. Its output fc does not have BN
            (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
            The dimension of h’s input and output (z and p) is d = 2048, 
            and h’s hidden layer’s dimension is 512, making h a 
            bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SimSiam(nn.Module):
    def __init__(self, in_dim_proj):
        super().__init__()

        self.projector = projection_MLP(in_dim_proj)
        self.predictor = prediction_MLP()

    def forward(self, x1, x2):
        z1 = self.projector(x1)
        z2 = self.projector(x2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return Dist(p1, z2) / 2 + Dist(p2, z1) / 2
