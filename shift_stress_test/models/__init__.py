import importlib

MODEL_NAMES = ["unet", "attention_unet", "deeplabv3plus", "hrnet", "transunet", "hiformer", "swinunet", "segformer"]
PRETRAINABLE = ["deeplabv3plus", "hrnet", "segformer"]
SUFFIX = "_pt"


def split_name(name):
    return (name[: -len(SUFFIX)], True) if name.endswith(SUFFIX) else (name, False)


def _module(base):
    if base not in MODEL_NAMES:
        raise ValueError(f"unknown model {base}")
    return importlib.import_module(f"shift_stress_test.models.{base}")


def build_model(name, num_classes=2, img_size=224):
    base, pretrained = split_name(name)
    if pretrained:
        if base not in PRETRAINABLE:
            raise ValueError(f"{base} has no pretrained variant")
        return _module(base).build_model(num_classes, img_size, pretrained=True)
    return _module(base).build_model(num_classes, img_size)


def model_info(name):
    base, pretrained = split_name(name)
    m = _module(base)
    return {"family": m.FAMILY, "source": m.SOURCE, "pretrained": pretrained}
