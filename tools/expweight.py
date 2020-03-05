import os
import sys
import torch
import argparse
sys.path.append('.')
from core import Yolact
from core.config import cfg

def parse_args():
    parser = argparse.ArgumentParser(description='Export weights of trained model from D. Bolya')
    parser.add_argument('--config-file', type=str, default='', help='path to the configuration file')
    parser.add_argument('--trained-model', type=str, default='', help='path to the trained model')
    parser.add_argument('--exported-model', type=str, default='', help='path to the exported model')
    parser.add_argument('opts', default=None, help='modify configuration using the command line', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if os.path.isfile(args.config_file):
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return args

def main(args):
    net = Yolact(cfg)
    
    # For backward compatability
    state_dict = torch.load(args.trained_model, map_location=torch.device('cpu'))
    for key in list(state_dict.keys()):
        if key.startswith('backbone.layer') and not key.startswith('backbone.layers'):
            del state_dict[key]
    
        if key.startswith('fpn.downsample_layers.'):
            if cfg.fpn.usable and int(key.split('.')[2]) >= cfg.fpn.num_downsamples:
                del state_dict[key]
    
    # Move the conv1 and bn1 to the most front of the weight list
    names = []
    weights = []
    conv1_names = []
    conv1_weights = []
    for key, value in state_dict.items():
        if not 'num_batches_tracked' in key:
            if key.startswith('backbone.conv1') or key.startswith('backbone.bn1'):
                conv1_names.insert(0, key)
                conv1_weights.insert(0, value)                
            else:
                names.append(key)
                weights.append(value)                
    
    for name, weight in zip(conv1_names, conv1_weights):
        names.insert(0, name)
        weights.insert(0, weight)
    
    # Copy weights to the model
    i = 0
    for name, module in net.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            assert module.weight.size() == weights[i].size()
            module.weight.data = weights[i].data.clone()
            i += 1
            if module.bias is not None:
                assert module.bias.size() == weights[i].size()
                module.bias.data = weights[i].data.clone()
                i += 1
        elif isinstance(module, torch.nn.BatchNorm2d):
            assert module.weight.size() == weights[i].size()
            module.weight.data = weights[i].data.clone()
            i += 1
            assert module.bias.size() == weights[i].size()
            module.bias.data = weights[i].data.clone()
            i += 1
            assert module.running_mean.size() == weights[i].size()
            module.running_mean.data = weights[i].data.clone()
            i += 1
            assert module.running_var.size() == weights[i].size()
            module.running_var.data = weights[i].data.clone()
            i += 1
    
    torch.save(net.state_dict(), args.exported_model)

if __name__ == '__main__':
    args = parse_args()
    main(args)