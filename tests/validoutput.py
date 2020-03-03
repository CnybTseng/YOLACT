import os
import sys
import torch
import argparse
sys.path.append('.')
from core import Yolact
from core.config import cfg
from core.layers import YolactDecoder

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
    
    detout = None
    decoder = YolactDecoder(cfg)
    input = torch.load(os.path.join(args.baseline_path, 'input.pt'), map_location=device)
    with torch.no_grad():
        output = net(input)
        detout = decoder(output)
    
    # Baseline data are coming from D. Bolya's implementation output
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
    
    bbox_baseline = torch.load('tests/data/boxes.pt', map_location=device)    
    mask_baseline = torch.load('tests/data/masks.pt', map_location=device)
    cidx_baseline = torch.load('tests/data/classes.pt', map_location=device)
    clas_baseline = torch.load('tests/data/scores.pt', map_location=device)
    
    print(f"bbox output is correct?{torch.equal(bbox_baseline, detout[0]['bbox'])}")   
    print(f"mask output is correct?{torch.equal(mask_baseline, detout[0]['mask'])}")
    print(f"cidx output is correct?{torch.equal(cidx_baseline, detout[0]['cidx'])}")
    print(f"clas output is correct?{torch.equal(clas_baseline, detout[0]['clas'])}")
    
    delta = torch.abs(bbox_baseline - detout[0]['bbox'])
    print(f"bbox difference: {delta}")
    
    for do in detout:
        for key, value in do.items():
            print(f"{key} {value}")

if __name__ == '__main__':
    args = parse_args()
    main(args)