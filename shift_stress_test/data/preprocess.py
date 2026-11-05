import numpy as np
from PIL import Image


def load_native(path, gray):
    with Image.open(path) as im:
        im = im.convert("L") if gray else im.convert("RGB")
        return np.asarray(im, dtype=np.uint8)


def load_mask(path, expected_hw):
    with Image.open(path) as mk:
        mk = mk.convert("L")
        if (mk.height, mk.width) != tuple(expected_hw):
            raise ValueError(f"image/mask size mismatch: {path}")
        return (np.asarray(mk) > 127).astype(np.uint8)


def resize_image(arr, size):
    out = np.asarray(Image.fromarray(arr).resize((size, size), Image.BILINEAR), dtype=np.uint8)
    return out[..., None] if out.ndim == 2 else out


def resize_mask(mask, size):
    return (np.asarray(Image.fromarray(mask * 255).resize((size, size), Image.NEAREST)) > 127).astype(np.uint8)


def luminance(img):
    img = img.astype(np.float32)
    if img.shape[-1] == 1:
        return img[..., 0]
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
