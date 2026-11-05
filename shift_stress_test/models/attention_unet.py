
import monai
from monai.networks.nets import AttentionUnet

MODEL_NAME = "attention_unet"
FAMILY = "classic_cnn"
SOURCE = {"type": "library", "package": "monai", "class": "monai.networks.nets.AttentionUnet", "version": monai.__version__, "variant": "channels (64,128,256,512,1024), strides (2,2,2,2), BatchNorm"}


def build_model(num_classes=2, img_size=224):
    return AttentionUnet(spatial_dims=2, in_channels=3, out_channels=num_classes, channels=(64, 128, 256, 512, 1024), strides=(2, 2, 2, 2))
