import os
import cv2
import torch
import argparse
from core import Yolact
from core.config import cfg
from core.layers import MaskProducer
from core.layers import YolactDecoder
from core.utils.transforms import BasicTransform
from tests.display import Drawer

def parse_args():
    parser = argparse.ArgumentParser(description='YOLACT training script')
    parser.add_argument('--config-file', type=str, default='', help='path to the configuration file')
    parser.add_argument('--model', type=str, default='', help='path to the model')
    parser.add_argument('--image', type=str, default='', help='path to the image or folder of images')
    parser.add_argument('--score-thresh', type=float, default=0.15, help='detection score threshold')
    parser.add_argument('--alpha', type=float, default=0.45, help='alpha')
    parser.add_argument('opts', default=None, help='modify configuration using the command line', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if os.path.isfile(args.config_file):
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return args

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Yolact(cfg).to(device)
    net.load_state_dict(torch.load(args.model))
    net.eval()
    
    if os.path.isfile(args.image):
        im1 = cv2.imread(args.image)
        im1 = torch.from_numpy(im1).to(device).float()
        im2 = im1.unsqueeze(dim=0)
        x  = BasicTransform(cfg)(im2)
    else:
        raise ValueError
    
    detout = None
    decoder = YolactDecoder(cfg)
    with torch.no_grad():
        output = net(x)
        detout = decoder(output)
    
    masker = MaskProducer(score_thresh=args.score_thresh)
    detout = masker(detout, im2.size(2), im2.size(1))
    
    drawer = Drawer(cfg, args.alpha)
    im3 = drawer(detout[0], im1)
    cv2.imwrite(os.path.join("tests", "data", "mask.png"), im3)

if __name__ == '__main__':
    args = parse_args()
    main(args)