import torch
import torch.nn.functional as F

class BasicTransform(object):
    def __init__(self, cfg):
        self.mean = torch.tensor(cfg.imagenet.mean, dtype=torch.float32)[None, :, None, None]
        self.std  = torch.tensor(cfg.imagenet.std, dtype=torch.float32)[None, :, None, None]
        self.size = cfg.backbone.insize
    
    def __call__(self, im):
        self.mean = self.mean.to(im.device)
        self.std  = self.std.to(im.device)
        im = im.permute(0, 3, 1, 2).contiguous()
        im = F.interpolate(im, size=(self.size, self.size), mode='bilinear', align_corners=False)
        im = (im - self.mean) / self.std
        im = im[:, (2,1,0), :, :].contiguous()
        return im