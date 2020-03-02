import torch
import torch.nn as nn
from math import sqrt
from core.backbone import FPN
from itertools import product
import torch.nn.functional as F
from core.backbone import build_backbone
from core.utils.functions import make_network
from torchvision.models.resnet import Bottleneck

class PredictHead(nn.Module):
    def __init__(self, in_channels, out_channels, cfg, parent=None, index=0):
        super(PredictHead, self).__init__()
        self.num_classes    = len(cfg.dataset.class_names) + 1
        self.mask_type      = cfg.mask.type
        self.mask_dim       = cfg.mask.dim
        self.aspect_ratios  = cfg.mask.aspect_ratios[index]
        self.scales         = cfg.mask.scales[index]
        self.usa            = cfg.mask.use_square_anchor
        self.imgsize        = cfg.backbone.insize
        self.num_anchors    = sum(len(ar) * len(self.scales) for ar in self.aspect_ratios)        
        self.parent         = [parent]
        self.index          = index
        self.num_heads      = len(cfg.backbone.selected_layers) + cfg.fpn.num_downsamples * (cfg.fpn.usable==True)
        self.priorboxes     = None
        
        if cfg.mask.prototype.split_by_head and cfg.mask.type == 'linearcombine':
            self.mask_dim //= self.num_heads
        if cfg.mask.prototype.as_features:
            in_channels += self.mask_dim
        
        if parent is not None:
            return
        
        self.extra_head = None
        if cfg.mask.extra_head_arch is None:
            out_channels = in_channels
        else:
            self.extra_head, out_channels = make_network(in_channels, cfg.mask.extra_head_arch)
        
        self.use_dssd_predict_module = cfg.mask.use_dssd_predict_module
        if cfg.mask.use_dssd_predict_module:
            self.block = Bottleneck(out_channels, out_channels//4)
            self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True) # maybe it's wrong
            self.norm = nn.BatchNorm2d(num_features=out_channels)
        
        self.bbox_layer = nn.Conv2d(out_channels,                4 * self.num_anchors, **cfg.mask.head_layer_params[0])
        self.clas_layer = nn.Conv2d(out_channels, self.num_classes * self.num_anchors, **cfg.mask.head_layer_params[0])
        self.mask_layer = nn.Conv2d(out_channels,    self.mask_dim * self.num_anchors, **cfg.mask.head_layer_params[0])
        
        self.use_mask_scoring = cfg.mask.use_mask_scoring
        if cfg.mask.use_mask_scoring:
            self.score_layer = nn.Conv2d(out_channels, self.num_anchors, **cfg.mask.head_layer_params)
        
        self.use_instance_coeff = cfg.mask.use_instance_coeff
        if cfg.mask.use_instance_coeff:
            raise NotImplementedError
        
        self.bbox_extra = self._make_extra_layer(cfg.mask.extra_layers[0], out_channels)
        self.clas_extra = self._make_extra_layer(cfg.mask.extra_layers[1], out_channels)
        self.mask_extra = self._make_extra_layer(cfg.mask.extra_layers[2], out_channels)
        
        self.gate_layer = None
        if cfg.mask.type == 'linearcombine' and cfg.mask.prototype.coeff_gate:
            self.gate_layer = nn.Conv2d(out_channels, self.num_anchors*self.mask_dim, kernel_size=3, padding=1)
    
    def _make_extra_layer(self, num_layers, io_channels):
        if num_layers == 0:
            return lambda x : x
        else:
            layer  = [nn.Conv2d(io_channels, io_channels, kernel_size=3, padding=1)]
            layer += [nn.ReLU(inplace=True)]
            return nn.Sequential(*sum([layer] * num_layers, []))
    
    def _make_prior_box(self, insize):        
        gh, gw = insize
        priorboxes = []
        for y, x in product(range(gh), range(gw)):
            bx = (x + .5) / gw
            by = (y + .5) / gh
            for aspect_ratio in self.aspect_ratios:
                for scale in self.scales:
                    for ar in aspect_ratio:
                        ar = sqrt(ar)
                        bw = scale * ar / self.imgsize
                        bh = scale / ar / self.imgsize if not self.usa else bw
                        priorboxes += [bx, by, bw, bh]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        priorboxes = torch.tensor(priorboxes, device=device).view(-1, 4).detach()
        return priorboxes.requires_grad_(False)
    
    def forward(self, x):
        if self.priorboxes is None:
            self.priorboxes = self._make_prior_box((x.size(2), x.size(3)))

        predictor = self if self.parent[0] is None else self.parent[0]
        if predictor.extra_head is not None:
            x = predictor.extra_head(x)
        
        if predictor.use_dssd_predict_module:
            y = predictor.block(x)
            z = predictor.conv(x)
            z = predictor.norm(x)
            x = y + F.relu(z)
        
        bbox_x = predictor.bbox_extra(x)
        clas_x = predictor.clas_extra(x)
        mask_x = predictor.mask_extra(x)
        bbox   = predictor.bbox_layer(bbox_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        clas   = predictor.clas_layer(clas_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
        mask   = predictor.mask_layer(mask_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
        
        if predictor.use_mask_scoring:
            raise NotImplementedError
        if predictor.use_instance_coeff:
            raise NotImplementedError
        
        if predictor.mask_type == 'linearcombine':
            mask = torch.tanh(mask)
            if predictor.gate_layer is not None:
                gate = predictor.gate_layer(mask).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
                mask = torch.sigmoid(gate) * mask
        else:
            raise NotImplementedError
        
        return {'bbox':bbox, 'clas':clas, 'mask':mask, 'prir':self.priorboxes}

class Yolact(nn.Module):
    def __init__(self, cfg):
        super(Yolact, self).__init__()
        self.backbone = build_backbone(cfg)
        
        if cfg.mask.type == 'linearcombine':
            if cfg.mask.prototype.use_grid:
                raise NotImplementedError
            else:
                self.num_grids = 0
            self.protosource = cfg.mask.prototype.source
            if self.protosource is None:
                in_channels = 3
            elif cfg.fpn is not None:
                in_channels = cfg.fpn.num_features
            else:
                in_channels = self.backbone.channels[self.protosource]
            in_channels += self.num_grids
            cfg.defrost()
            self.protonet, cfg.mask.dim = make_network(in_channels, cfg.mask.prototype.arch, with_last_relu=False)
            if cfg.mask.prototype.bias:
                cfg.mask.dim += 1
            cfg.freeze()
        else:
            raise NotImplementedError
        
        if cfg.mask.use_maskiou:
            raise NotImplementedError

        self.num_heads = len(cfg.backbone.selected_layers)
        self.backbone_selected_layers = cfg.backbone.selected_layers
        in_channels = [self.backbone.channels[i] for i in cfg.backbone.selected_layers]
        
        self.fpn = None
        if cfg.fpn.usable:
            self.fpn = FPN(in_channels, cfg)
            self.num_heads += cfg.fpn.num_downsamples
            in_channels = [cfg.fpn.num_features] * self.num_heads
        
        self.predict_heads = nn.ModuleList()
        for head_idx, pyramid_level in enumerate(range(self.num_heads)):
            parent = self.predict_heads[0] if cfg.mask.share_predict_head and head_idx > 0 else None
            head = PredictHead(in_channels[pyramid_level], in_channels[pyramid_level], cfg, parent, head_idx)
            self.predict_heads.append(head)
        
        if cfg.mask.use_class_existence_loss:
            raise NotImplementedError
        if cfg.mask.use_semantic_segmentation_loss:
            self.semseg = nn.Conv2d(in_channels[0], out_channels=len(cfg.dataset.class_names), kernel_size=1)
    
    def forward(self, x):
        bbout = self.backbone(x)        
        if self.fpn is not None:
            fpnin  = [bbout[i] for i in self.backbone_selected_layers]
            bbout = self.fpn(fpnin)
        
        protoin = x if self.protosource is None else bbout[self.protosource]
        if self.num_grids > 0:
            raise NotImplementedError
        protout = self.protonet(protoin)
        protout = F.relu(protout, inplace=True)
        protout = protout.permute(0, 2, 3, 1).contiguous()
        
        predout = {'bbox':[], 'clas':[], 'mask':[], 'prir':[]}
        for head_idx, predictor in enumerate(self.predict_heads):
            po = predictor(bbout[head_idx])
            for key, value in po.items():
                predout[key].append(value)
        for key, value in predout.items():
            predout[key] = torch.cat(tensors=value, dim=-2)
        predout['clas'] = F.softmax(input=predout['clas'], dim=-1)
        predout['prot'] = protout
        
        return predout