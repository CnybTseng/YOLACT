import torch
import torch.nn.functional as F

class MaskProducer(object):
    def __init__(self, interpolation_mode='bilinear', crop_mask=True, score_thresh=0):
        self.interpolation_mode = interpolation_mode
        self.crop_mask = crop_mask
        self.score_thresh = score_thresh
    
    def __call__(self, detout, imw, imh):
        for det in detout:
            keeps = det['clas'] > self.score_thresh
            for k, v in det.items():
                if k != 'prot':
                    det[k] = det[k][keeps]
        
            det['mask'] = self._mkmask(det, imw, imh)
            x1, y1, x2, y2 = self._absbox(det['bbox'], imw, imh)
            det['bbox'] = torch.stack([x1, y1, x2, y2], dim=1).long()
        return detout
    
    def _mkmask(self, detout, imw, imh):
        bbox = detout['bbox']
        mask = detout['mask']
        prot = detout['prot']
        mask = prot @ mask.t()
        mask = torch.sigmoid(mask)
        if self.crop_mask:
            mask = self._crop(mask, bbox)
        mask = mask.permute(2, 0, 1).contiguous().unsqueeze(dim=0)
        mask = F.interpolate(mask, size=(imh, imw), mode=self.interpolation_mode, align_corners=False).squeeze(dim=0)
        mask.gt_(0.5)
        return mask
    
    def _crop(self, mask, bbox, padding=1):
        h, w, c = mask.size()
        x1, y1, x2, y2 = self._absbox(bbox, w, h, padding)
        rows = torch.arange(w, device=mask.device, dtype=x1.dtype).view(1, -1, 1).expand(h, w, c)
        cols = torch.arange(h, device=mask.device, dtype=x1.dtype).view(-1, 1, 1).expand(h, w, c)
        left   = rows >= x1.view(1, 1, -1)
        right  = rows <  x2.view(1, 1, -1)
        top    = cols >= y1.view(1, 1, -1)
        bottom = cols <  y2.view(1, 1, -1)
        sel = left * right * top * bottom
        return mask * sel.float()

    def _absbox(self, bbox, w, h, padding=0):
        x1, x2 = bbox[:, 0] * w, bbox[:, 2] * w
        y1, y2 = bbox[:, 1] * h, bbox[:, 3] * h
        x1, x2 = self._calibrate_range(x1, x2, 0, w, padding)
        y1, y2 = self._calibrate_range(y1, y2, 0, h, padding)
        return x1, y1, x2, y2
        
    def _calibrate_range(self, x1, x2, min, max, padding=0):
        _x1 = torch.min(x1, x2) - padding
        _x2 = torch.max(x1, x2) + padding
        _x1 = torch.clamp(_x1, min=min)
        _x2 = torch.clamp(_x2, max=max)
        return _x1, _x2