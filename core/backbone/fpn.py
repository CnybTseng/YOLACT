import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_channels, cfg):
        super(FPN, self).__init__()
        self.interpolation_mode         = cfg.fpn.interpolation_mode
        self.num_downsamples            = cfg.fpn.num_downsamples
        self.downsample_layer_with_relu = cfg.fpn.downsample_layer_with_relu
        self.pred_layer_with_relu       = cfg.fpn.pred_layer_with_relu        
        
        self.adapt_layers = nn.ModuleList([
            nn.Conv2d(c, cfg.fpn.num_features, kernel_size=1) for c in reversed(in_channels)])
        
        self.predt_layers = nn.ModuleList([
            nn.Conv2d(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=1) for _ in in_channels])
        
        self.downsample_layers = nn.ModuleList([
            nn.Conv2d(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, stride=2, padding=1)
            for _ in range(self.num_downsamples)]) if cfg.fpn.use_conv_downsample else None      
    
    def forward(self, xs):
        y = torch.zeros(1, dtype=xs[0].dtype, device=xs[0].device)        
        i = len(xs)
        ys = [y] * i
        for adapt_layer, predt_layer in zip(self.adapt_layers, self.predt_layers):
            i -= 1
            if i < len(xs) - 1:
                _, _, h, w = xs[i].size()
                y = F.interpolate(y, size=(h,w), mode=self.interpolation_mode, align_corners=False)
            y = y + adapt_layer(xs[i])
            ys[i] = predt_layer(y)
            if self.pred_layer_with_relu:
                F.relu(ys[i], inplace=True)
        
        if self.downsample_layers is not None:
            for downsample_layer in self.downsample_layers:
                ys.append(downsample_layer(ys[-1]))
        else:
            raise NotImplementedError
        
        if self.downsample_layer_with_relu:
            for i in range(len(xs), len(ys)):
                F.relu(y[i], inplace=True)
        
        return ys