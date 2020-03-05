import cv2
import torch
import numpy as np

class Drawer(object):
    def __init__(self, cfg, alpha=0.45):
        self.class_names = cfg.dataset.class_names
        self.alpha = alpha
        self.fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL
        self.fontScale = 1
        self.thickness = 1
    
    def __call__(self, detout, im):
        bbox = detout['bbox']
        mask = detout['mask']
        cate = detout['cate']
        clas = detout['clas']
        
        im = im / 255.0
        num_dets = mask.size(0)
        colors = torch.cat([self._randcolor(mask.device).view(1, 1, 1, 3) for _ in range(num_dets)], dim=0)
        mask = mask[:, :, :, None]
        obj = self.alpha * colors * mask.repeat(1, 1, 1, 3)
        bkg_mask = 1 - self.alpha * mask
        for i in range(num_dets):
            im = bkg_mask[i] * im + obj[i]

        im = (255 * im).byte().cpu().numpy()
        bbox = bbox.cpu().numpy().tolist()
        colors = (255 * colors).byte().squeeze().cpu().numpy().tolist()
        for i in range(num_dets):
            x1, y1, x2, y2 = bbox[i]
            text = self.class_names[cate[i]]
            
            [[tw, th], baseline] = cv2.getTextSize(text, self.fontFace, self.fontScale, self.thickness)
            tx = np.clip(x1, 0, im.shape[1] - tw - 1)
            ty = np.clip(y1 - baseline, th, im.shape[0] - baseline - 1)
            
            cv2.rectangle(im, (x1, y1), (x2, y2), colors[i])
            cv2.rectangle(im, (tx, ty - th), (tx + tw - 1,ty + baseline), colors[i], self.thickness)
            cv2.putText(im, text, (tx, ty), self.fontFace, self.fontScale, colors[i], self.thickness)
        
        return im
    
    def _randcolor(self, device):
        return torch.rand(3, device=device)