import os
import sys
import torch
import argparse
sys.path.append('.')
from core.config import cfg
from core import Yolact

def parse_args():
    parser = argparse.ArgumentParser(description="Validate the correctness of the model's outputs")
    parser.add_argument('--config-file', type=str, default='', help='path to the configuration file')
    parser.add_argument('--model', type=str, default='', help='path to the model')
    parser.add_argument('--baseline-path', type=str, default='', help='path to the baseline')
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
    
    input = torch.load(os.path.join(args.baseline_path, 'input.pt'), map_location=device)
    with torch.no_grad():
        output = net(input)
    
    # Baseline data are coming from D. Bolya's implementation
    bbox_baseline = torch.load(os.path.join(args.baseline_path, 'loc.pt'), map_location=device)
    clas_baseline = torch.load(os.path.join(args.baseline_path, 'conf.pt'), map_location=device)
    mask_baseline = torch.load(os.path.join(args.baseline_path, 'mask.pt'), map_location=device)
    prir_baseline = torch.load(os.path.join(args.baseline_path, 'priors.pt'), map_location=device)
    prot_baseline = torch.load(os.path.join(args.baseline_path, 'proto.pt'), map_location=device)
    
    print(f"bbox output is correct?{torch.equal(bbox_baseline, output['bbox'])}")
    print(f"clas output is correct?{torch.equal(clas_baseline, output['clas'])}")
    print(f"mask output is correct?{torch.equal(mask_baseline, output['mask'])}")
    print(f"prir output is correct?{torch.equal(prir_baseline, output['prir'])}")
    print(f"prot output is correct?{torch.equal(prot_baseline, output['prot'])}")

if __name__ == '__main__':
    args = parse_args()
    main(args)