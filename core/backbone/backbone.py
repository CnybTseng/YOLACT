from core.backbone import ResNet
from core.utils.registry import Registry

BACKBONES = Registry()

@BACKBONES.register("ResNet50")
@BACKBONES.register("ResNet101")
def build_resnet_backbone(cfg):
    return ResNet(*cfg.backbone.args)

@BACKBONES.register("DarkNet53")
def build_darknet_backbone(cfg):
    print("DarkNet53 hasn't been implemented yet, using ResNet instead")
    return ResNet(*cfg.backbone.args)

def build_backbone(cfg):
    return BACKBONES[cfg.backbone.name](cfg)