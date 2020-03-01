import torch.nn as nn
import torch.nn.functional as F

class Resample(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Resample, self).__init__()
        self.args   = args
        self.kwargs = kwargs
    
    def forward(self, x):
        return F.interpolate(x, *self.args, **self.kwargs)