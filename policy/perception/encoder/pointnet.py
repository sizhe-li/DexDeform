'''
Codes are from https://github.com/autonomousvision/convolutional_occupancy_networks
'''

import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max

from policy.perception.common import coordinate2index, normalize_coordinate, PLANE_TYPES
from policy.perception.encoder.unet import UNet
from policy.perception.encoder.unet3d import UNet3D
from policy.perception.layers import ResnetBlockFC


class LocalPoolPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks for each point.
        Number of input points are fixed.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, scatter_type='max',
                 unet=False, unet_kwargs=None, unet3d=False, unet3d_kwargs=None,
                 plane_resolution=None, grid_resolution=None, padding=0.1, n_blocks=5):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim

        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None

        if unet3d:
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None

        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.padding = padding

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')

    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)  # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane ** 2)
        c = c.permute(0, 2, 1)  # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane)  # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane,
                                      self.reso_plane)  # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        return fea_plane

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            if key == 'grid':
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_grid ** 3)
            else:
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_plane ** 2)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def forward(self, p):
        # acquire the index for each point
        coord = {}
        index = {}
        for plane_type in PLANE_TYPES:
            coord[plane_type] = normalize_coordinate(p.clone(), plane=plane_type, padding=self.padding)
            index[plane_type] = coordinate2index(coord[plane_type], self.reso_plane)

        net = self.fc_pos(p)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)

        fea = {}
        for plane_type in PLANE_TYPES:
            fea[plane_type] = self.generate_plane_features(p, c, plane=plane_type)

        return fea
