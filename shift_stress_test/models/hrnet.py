
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

MODEL_NAME = "hrnet"
FAMILY = "dilated_multiscale_cnn"
SOURCE = {"type": "library", "package": "timm", "class": "timm hrnet_w32 (features_only, raw branch outputs)", "version": timm.__version__, "variant": "HRNet-W32 backbone + HRNetV2 segmentation head (upsample 4 branches to 1/4, concat 480 ch, 1x1 conv-BN-ReLU, 1x1 classifier, x4 bilinear)"}


class HRNetSeg(nn.Module):

    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model("hrnet_w32", pretrained=pretrained, features_only=True, feature_location="", out_indices=(1, 2, 3, 4))
        ch = sum(self.backbone.feature_info.channels())
        self.head = nn.Sequential(nn.Conv2d(ch, ch, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True), nn.Conv2d(ch, num_classes, 1))

    def forward(self, x):
        feats = self.backbone(x)
        size = feats[0].shape[-2:]
        y = torch.cat([feats[0]] + [F.interpolate(f, size=size, mode="bilinear", align_corners=False) for f in feats[1:]], dim=1)
        return F.interpolate(self.head(y), size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_model(num_classes=2, img_size=224, pretrained=False):
    return HRNetSeg(num_classes, pretrained)
