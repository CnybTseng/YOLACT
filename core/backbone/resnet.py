import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, neck_channels, stride=1, dilation=1, downsample=None, norm=nn.BatchNorm2d, dcn_usable=False):
        super(Bottleneck, self).__init__()        
        self.conv1  = [nn.Conv2d(in_channels, neck_channels, kernel_size=1, dilation=dilation, bias=False)]
        self.conv1 += [norm(num_features=neck_channels)]
        self.conv1 += [nn.ReLU(inplace=True)]
        self.conv1  = nn.Sequential(*self.conv1)        
        if dcn_usable:
            raise NotImplementedError
        else:
            self.conv2 = [nn.Conv2d(neck_channels, neck_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)]        
        self.conv2 += [norm(num_features=neck_channels)]
        self.conv2 += [nn.ReLU(inplace=True)]
        self.conv2  = nn.Sequential(*self.conv2)        
        self.conv3  = [nn.Conv2d(neck_channels, neck_channels * self.expansion, kernel_size=1, dilation=dilation, bias=False)]
        self.conv3 += [norm(num_features=neck_channels * self.expansion)]
        self.conv3  = nn.Sequential(*self.conv3)
        self.shortcut = downsample if downsample is not None else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        z = self.shortcut(x)
        return self.relu(y + z)

class ResNet(nn.Module):
    def __init__(self, num_blocks, dcn_usable=[0,0,0,0], dcn_interval=1, atrous_layers=[], block=Bottleneck, norm=nn.BatchNorm2d):
        super(ResNet, self).__init__()
        self.dcn_interval = dcn_interval
        self.atrous_layers = atrous_layers
        self.block = block
        self.norm = norm
        self.in_channels = 64
        self.dilation = 1
        self.channels = []

        self.conv1  = [nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)]
        self.conv1 += [norm(num_features=64)]
        self.conv1 += [nn.ReLU(inplace=True)]
        self.conv1 += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        self.conv1  = nn.Sequential(*self.conv1)
        
        self.conv2_x = self._stack_blocks(neck_channels=64,  num_blocks=num_blocks[0], dcn_usable=dcn_usable[0])
        self.conv3_x = self._stack_blocks(neck_channels=128, num_blocks=num_blocks[1], stride=2, dcn_usable=dcn_usable[1])
        self.conv4_x = self._stack_blocks(neck_channels=256, num_blocks=num_blocks[2], stride=2, dcn_usable=dcn_usable[2])
        self.conv5_x = self._stack_blocks(neck_channels=512, num_blocks=num_blocks[3], stride=2, dcn_usable=dcn_usable[3])
    
    def _stack_blocks(self, neck_channels, num_blocks, stride=1, dcn_usable=0):
        downsample = None
        if stride != 1 or (self.in_channels != neck_channels * self.block.expansion):
            if self.atrous_layers != []:
                raise NotImplementedError
            out_channels = neck_channels * self.block.expansion
            downsample  = [nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, dilation=self.dilation, bias=False)]
            downsample += [nn.BatchNorm2d(num_features=out_channels)]
            downsample  = nn.Sequential(*downsample)
       
        modules = [self.block(self.in_channels, neck_channels, stride, self.dilation, downsample, self.norm, dcn_usable)]
        self.in_channels = neck_channels * self.block.expansion
        self.channels.append(self.in_channels)
        for i in range(1, num_blocks):
            usable = dcn_usable and (i % self.dcn_interval == 0)
            modules += [self.block(self.in_channels, neck_channels, norm=self.norm, dcn_usable=usable)]
        
        return nn.Sequential(*modules)
    
    def forward(self, x):
        outputs = []
        y = self.conv1(x)
        y = self.conv2_x(y)
        outputs.append(y)
        y = self.conv3_x(y)
        outputs.append(y)
        y = self.conv4_x(y)
        outputs.append(y)
        y = self.conv5_x(y)
        outputs.append(y)
        return tuple(outputs)