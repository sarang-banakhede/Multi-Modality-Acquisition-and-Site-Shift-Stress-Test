import cv2
import numpy as np

from .base import Degradation
from .endoscopy import motion_kernel


def to_transmission(img, d):
    return 10.0 ** (-(1.0 - img.astype(np.float32) / 255.0) * d)


def to_display(trans, d):
    return 255.0 * (1.0 + np.log10(np.maximum(trans, 1e-6)) / d)


class LowDose(Degradation):
    name = "X1"
    unit = "dose fraction lost"

    def prepare(self, img, rng):
        return rng.standard_normal(img.shape).astype(np.float32), rng.standard_normal(img.shape).astype(np.float32)

    def apply(self, img, state, s):
        d, n0, electronic = self.params["density"], self.params["n0"], self.params["electronic"]
        lost = s * self.params["max"]
        lam = n0 * (1.0 - lost) * to_transmission(img, d)
        counts = lam + np.sqrt(np.maximum(lam, 0.0)) * state[0] + electronic * state[1]
        return to_display(np.maximum(counts, 0.0) / (n0 * (1.0 - lost)), d)

    def physical(self, s):
        return s * self.params["max"]


class BreathingMotion(Degradation):
    name = "X2"
    unit = "motion length / height"

    def prepare(self, img, rng):
        ang = np.deg2rad(90.0 + rng.uniform(-15.0, 15.0))
        t = np.linspace(-0.5, 0.5, 33)
        return np.stack([t * np.cos(ang), t * np.sin(ang)], axis=1)

    def apply(self, img, state, s):
        length = s * self.params["max"] * img.shape[0]
        if length < 2:
            return img
        blurred = cv2.filter2D(img.astype(np.float32), -1, motion_kernel(state, length), borderType=cv2.BORDER_REFLECT)
        w = np.linspace(0.0, 1.0, img.shape[0], dtype=np.float32)[:, None] ** self.params["ramp"]
        return img.astype(np.float32) * (1 - w) + blurred * w

    def physical(self, s):
        return s * self.params["max"]


class Scatter(Degradation):
    name = "X3"
    unit = "scatter-to-primary ratio"

    def apply(self, img, state, s):
        d = self.params["density"]
        trans = to_transmission(img, d)
        sigma = self.params["sigma"] * img.shape[1]
        mixed = trans + s * self.params["max"] * cv2.GaussianBlur(trans, (0, 0), sigma)
        lo, hi = trans.min(), trans.max()
        mixed = (mixed - mixed.min()) / max(mixed.max() - mixed.min(), 1e-9) * (hi - lo) + lo
        return to_display(mixed, d)

    def physical(self, s):
        return s * self.params["max"]


REGISTRY = {"X1": LowDose, "X2": BreathingMotion, "X3": Scatter}
