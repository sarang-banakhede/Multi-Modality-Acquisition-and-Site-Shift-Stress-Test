import io

import cv2
import numpy as np
from PIL import Image

from .base import Degradation, short_side, to_float


class GaussianBlur(Degradation):
    name = "G1"
    unit = "sigma / short side"

    def apply(self, img, state, s):
        sigma = s * self.params["max"] * short_side(img)
        return cv2.GaussianBlur(img, (0, 0), max(sigma, 1e-3))

    def physical(self, s):
        return s * self.params["max"]


class GaussianNoise(Degradation):
    name = "G2"
    unit = "noise std / 255"

    def prepare(self, img, rng):
        return rng.standard_normal(img.shape).astype(np.float32)

    def apply(self, img, state, s):
        return to_float(img) + state * (s * self.params["max"] * 255.0)

    def physical(self, s):
        return s * self.params["max"]


class JpegCompression(Degradation):
    name = "G3"
    unit = "JPEG quality"

    def quality(self, s):
        hi, lo = self.params["max_quality"], self.params["min_quality"]
        return int(round(hi - s * (hi - lo)))

    def apply(self, img, state, s):
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=self.quality(s), subsampling=2 if img.ndim == 3 else 0)
        return np.asarray(Image.open(io.BytesIO(buf.getvalue())))

    def physical(self, s):
        return float(self.quality(s))


REGISTRY = {"G1": GaussianBlur, "G2": GaussianNoise, "G3": JpegCompression}
