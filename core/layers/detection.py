import torch

class YolactDecoder(object):
    def __init__(self, cfg):
        self.topk        = cfg.decoder.topk
        self.cnfd_thresh = cfg.decoder.cnfd_thresh
        self.nms_thresh  = cfg.decoder.nms_thresh
        self.imgsize     = cfg.backbone.insize
        self.variances   = (.1, .2)
    
    def __call__(self, predout):
        bbox = predout['bbox']
        clas = predout['clas']
        mask = predout['mask']
        prir = predout['prir']
        prot = predout['prot']
        
        detout = []
        batch_size = bbox.size(0)
        clas = clas.transpose(2, 1).contiguous()
        for batch in range(batch_size):
            max_score, _ = torch.max(input=clas[batch, 1:, :], dim=0)
            keeps = max_score > self.cnfd_thresh
            if torch.nonzero(keeps).size(0) == 0:
                continue
            decoded_bbox = self._decode(bbox[batch, keeps, :], prir[keeps, :])
            _bbox, _mask, _cate, _clas = self._nms(decoded_bbox, clas[batch, 1:, keeps], mask[batch, keeps, :])
            detout.append({'bbox':_bbox, 'mask':_mask, 'cate':_cate, 'clas':_clas, 'prot':prot[batch]})

        return detout

    def _decode(self, bbox, priorboxes):
        xy_cntr = priorboxes[:, :2] + priorboxes[:, 2:] * bbox[:, :2] * self.variances[0]
        wh_half = priorboxes[:, 2:] * torch.exp(bbox[:, 2:] * self.variances[1]) / 2
        x1y1    = xy_cntr - wh_half
        x2y2    = xy_cntr + wh_half
        return torch.cat(tensors=[x1y1, x2y2], dim=1)
    
    def _nms(self, bbox, clas, mask):
        num_classes = clas.size(0)
        _bbox = bbox * self.imgsize
        indices = torch.arange(bbox.size(0), device=bbox.device)
        nms_cate = []
        nms_inds = []
        nms_clas = []
        for cid in range(num_classes):
            keeps = clas[cid, :] > self.cnfd_thresh
            if torch.nonzero(keeps).size(0) == 0:
                continue
            keep_inds = indices[keeps]
            keep_bbox = _bbox[keeps, :]
            keep_clas = clas[cid, keeps]
            keeps = self._nms_per_class(keep_bbox, keep_clas)
            nms_cate.append(torch.LongTensor([cid] * keeps.size(0)).to(bbox.device))
            nms_inds.append(keep_inds[keeps])
            nms_clas.append(keep_clas[keeps])
        
        nms_cate = torch.cat(tensors=nms_cate, dim=0)
        nms_inds = torch.cat(tensors=nms_inds, dim=0)
        nms_clas = torch.cat(tensors=nms_clas, dim=0)
        
        nms_bbox = bbox[nms_inds, :]
        nms_mask = mask[nms_inds, :]
        topk = self.topk if nms_clas.size(0) >= self.topk else nms_clas.size(0)
        nms_clas, indices = torch.topk(nms_clas, topk, dim=0)        
        nms_bbox = nms_bbox[indices, :]
        nms_mask = nms_mask[indices, :]
        nms_cate = nms_cate[indices]
              
        return nms_bbox, nms_mask, nms_cate, nms_clas
            
    def _nms_per_class(self, bbox, clas):
        num_dets = bbox.size(0)
        keeps = torch.ByteTensor(num_dets).fill_(1)
        areas = (bbox[:, 2] - bbox[:, 0] + 1) * (bbox[:, 3] - bbox[:, 1] + 1)
        indices = torch.argsort(clas, dim=0, descending=True)
        for _i in range(num_dets):
            i = indices[_i]
            if keeps[i] == 0:
                continue
            for _j in range(_i + 1, num_dets):
                j = indices[_j]
                if keeps[j] == 0:
                    continue
                minx = max(bbox[i, 0], bbox[j, 0])
                maxx = min(bbox[i, 2], bbox[j, 2])
                miny = max(bbox[i, 1], bbox[j, 1])
                maxy = min(bbox[i, 3], bbox[j, 3])
                w = max(0, maxx - minx + 1)
                h = max(0, maxy - miny + 1)
                inter = w * h
                iou = inter / (areas[i] + areas[j] - inter)
                if iou >= self.nms_thresh:
                    keeps[j] = 0
        return torch.nonzero(keeps).view(-1).contiguous()