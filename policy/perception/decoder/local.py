"""
Codes are from https://github.com/autonomousvision/convolutional_occupancy_networks
"""
import torch.nn as nn
import torch.nn.functional as F

from policy.perception.common import sample_plane_feature, PLANE_TYPES
from policy.perception.layers import ResnetBlockFC


class LocalDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(
            self,
            dim=3,
            c_dim=128,
            hidden_size=256,
            n_blocks=5,
            leaky=False,
            sample_mode='bilinear',
            padding=0.1
    ):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, 3)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

    def forward(self, p, c_plane):
        c = 0
        for plane_type in PLANE_TYPES:
            c += sample_plane_feature(p, c_plane[plane_type],
                                      plane=plane_type,
                                      padding=self.padding,
                                      mode=self.sample_mode)
        c = c.transpose(1, 2)

        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            net = net + self.fc_c[i](c)
            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))

        return out
