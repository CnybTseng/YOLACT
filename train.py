import os
import torch
import argparse
from core.config import cfg
from core import Yolact

def parse_args():
    parser = argparse.ArgumentParser(description='YOLACT training script')
    parser.add_argument('--config-file', type=str, default='', help='path to the configuration file')
    parser.add_argument('opts', default=None, help='modify configuration using the command line', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if os.path.isfile(args.config_file):
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

def main():
    net = Yolact(cfg)
    x = torch.rand(1, 3, 550, 550)
    y = net(x)
    print(net)
    for k, v in y.items():
        print(f"{k}: {v.size()}")

if __name__ == '__main__':
    parse_args()
    main()