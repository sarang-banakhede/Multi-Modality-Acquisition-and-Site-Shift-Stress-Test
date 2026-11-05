
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import SegformerConfig, SegformerForSemanticSegmentation

MODEL_NAME = "segformer"
FAMILY = "pure_transformer"
SOURCE = {"type": "library", "package": "transformers", "class": "SegformerForSemanticSegmentation", "version": transformers.__version__, "variant": "MiT-b2 configuration, random init, logits upsampled from 1/4 to input size"}


class SegFormer(nn.Module):

    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        cfg = SegformerConfig(num_channels=3, num_encoder_blocks=4, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], hidden_sizes=[64, 128, 320, 512], patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2], num_attention_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], hidden_act="gelu", hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0, classifier_dropout_prob=0.1, drop_path_rate=0.1, decoder_hidden_size=768, num_labels=num_classes)
        self.net = SegformerForSemanticSegmentation(cfg)
        if pretrained:
            enc = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b2", config=cfg, ignore_mismatched_sizes=True)
            self.net.load_state_dict(enc.state_dict())

    def forward(self, x):
        logits = self.net(pixel_values=x).logits
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_model(num_classes=2, img_size=224, pretrained=False):
    return SegFormer(num_classes, pretrained)
