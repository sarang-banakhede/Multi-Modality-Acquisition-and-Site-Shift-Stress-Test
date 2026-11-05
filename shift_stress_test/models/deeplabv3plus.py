
import segmentation_models_pytorch as smp

MODEL_NAME = "deeplabv3plus"
FAMILY = "dilated_multiscale_cnn"
SOURCE = {"type": "library", "package": "segmentation_models_pytorch", "class": "smp.DeepLabV3Plus", "version": smp.__version__, "variant": "ResNet-50 encoder, encoder_weights=None, output stride 16"}


def build_model(num_classes=2, img_size=224, pretrained=False):
    return smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=num_classes)
