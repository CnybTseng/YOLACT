import torch
import torch.nn as nn
from core.layers import Resample

def make_network(in_channels, cfgs, with_last_relu=True):
    def make_layer(cfg):
        nonlocal in_channels
        if isinstance(cfg[0], str):
            raise NotImplementedError
        else:
            out_channels, kernel_size = cfg[:2]
            if kernel_size > 0:
                layer = nn.Conv2d(in_channels, out_channels, kernel_size, **cfg[2])
            else:
                kernel_size = -kernel_size
                if out_channels is None:
                    layer = Resample(scale_factor=kernel_size, mode='bilinear', align_corners=False, **cfg[2])
                else:
                    layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, **cfg[2])
        in_channels = out_channels if out_channels is not None else in_channels
        return [layer, nn.ReLU(inplace=True)]
    
    network = sum([make_layer(cfg) for cfg in cfgs], [])
    if not with_last_relu:
        network = network[:-1]
    return nn.Sequential(*(network)), in_channels