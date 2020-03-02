from yacs.config import CfgNode as CN

_C = CN()

# --------------------------------------------------------------------------------
# Copyright
# --------------------------------------------------------------------------------

_C.copyright = CN()
_C.copyright.author = "Zeng Zhiwei"

# --------------------------------------------------------------------------------
# ImageNet
# --------------------------------------------------------------------------------

_C.imagenet = CN()
_C.imagenet.mean = (103.94, 116.78, 123.68)
_C.imagenet.std  = ( 57.38,  57.12,  58.40)

# --------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------

PASCAL_CLASSES = (
"aeroplane", "bicycle", "bird", "boat", "bottle",
"bus", "car", "cat", "chair", "cow",
"diningtable", "dog", "horse", "motorbike", "person",
"pottedplant", "sheep", "sofa", "train", "tvmonitor")

_C.dataset = CN()
_C.dataset.name = 'Pascal SBD'
_C.dataset.class_names  = PASCAL_CLASSES
_C.dataset.train_images = '/path/to/training/images'
_C.dataset.train_annots = '/path/to/training/annotations'
_C.dataset.valid_images = '/path/to/validation/images'
_C.dataset.valid_annots = '/path/to/validation/annotations'

# --------------------------------------------------------------------------------
# Dataloader
# --------------------------------------------------------------------------------

_C.dataloader = CN()
_C.dataloader.num_workers = 4

# --------------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------------

_C.solver = CN()
_C.solver.max_iter = 10000
_C.solver.warmup = 1000
_C.solver.base_lr = 1e-3
_C.solver.lr_gamma = 0.1
_C.solver.milestones = ()
_C.solver.momentum = 0.9
_C.solver.weight_decay = 1e-5
_C.solver.batch_size = 8

# --------------------------------------------------------------------------------
# Backbone
# --------------------------------------------------------------------------------

_C.backbone = CN()
_C.backbone.name = 'ResNet101'
_C.backbone.insize = 550
_C.backbone.args = ([3,4,23,3],)
_C.backbone.selected_layers = (1,2,3)

# --------------------------------------------------------------------------------
# FPN
# --------------------------------------------------------------------------------

_C.fpn = CN()
_C.fpn.usable = True
_C.fpn.num_features = 256
_C.fpn.interpolation_mode = 'bilinear'
_C.fpn.num_downsamples = 2
_C.fpn.use_conv_downsample = True
_C.fpn.downsample_layer_with_relu = False
_C.fpn.pred_layer_with_relu = True

# --------------------------------------------------------------------------------
# Mask
# --------------------------------------------------------------------------------

_C.mask = CN()
_C.mask.type = 'linearcombine'
_C.mask.dim = 0
_C.mask.extra_head_arch = [(256,3,{'padding':1})]
_C.mask.extra_layers = [0,0,0]
_C.mask.use_dssd_predict_module = False
_C.mask.head_layer_params = [{'kernel_size':3,'padding':1}]
_C.mask.share_predict_head = True
_C.mask.aspect_ratios = [[[1,0.5,2]]] * 5
_C.mask.scales = [[24],[48],[96],[192],[384]]
_C.mask.use_square_anchor = True
_C.mask.prototype = CN()
_C.mask.prototype.use_grid = False
_C.mask.prototype.bias = False
_C.mask.prototype.source = 0
_C.mask.prototype.arch = [(256,3,{'padding':1})] * 3 + [(None,-2,{}), (256,3,{'padding':1})] + [(32,1,{})]
_C.mask.prototype.split_by_head = False
_C.mask.prototype.as_features = False
_C.mask.prototype.coeff_gate = False
_C.mask.prototype.coeff_activation = 'tanh'
_C.mask.prototype.proto_activation = 'relu'
_C.mask.use_maskiou = False
_C.mask.use_mask_scoring = False
_C.mask.use_instance_coeff = False
_C.mask.use_class_existence_loss = False
_C.mask.use_semantic_segmentation_loss = True