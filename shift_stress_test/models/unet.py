
import contextlib
import io
import monai
from monai.networks.nets import BasicUNet

MODEL_NAME = "unet"
FAMILY = "classic_cnn"
SOURCE = {"type": "library", "package": "monai", "class": "monai.networks.nets.BasicUNet", "version": monai.__version__, "variant": "features (64,128,256,512,1024,64), BatchNorm, ReLU, transposed-conv upsampling"}


def build_model(num_classes=2, img_size=224):
    with contextlib.redirect_stdout(io.StringIO()):
        return BasicUNet(spatial_dims=2, in_channels=3, out_channels=num_classes, features=(64, 128, 256, 512, 1024, 64), act="relu", norm="batch", upsample="deconv")
