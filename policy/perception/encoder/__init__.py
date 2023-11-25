"""
Codes are from https://github.com/autonomousvision/convolutional_occupancy_networks
"""
from policy.perception.encoder import (
    pointnet, unet
)


encoder_dict = {
    'pointnet_local_pool': pointnet.LocalPoolPointnet,
    'unet': unet.UnetPlan,
}
