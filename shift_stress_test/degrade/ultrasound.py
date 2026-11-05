import cv2
import numpy as np

from .base import Degradation

RANGE_DB = 60.0


def to_amplitude(img):
    return 10.0 ** ((img.astype(np.float32) / 255.0 - 1.0) * RANGE_DB / 20.0)


def to_display(amp):
    return 255.0 * (1.0 + 20.0 * np.log10(np.maximum(amp, 1e-6)) / RANGE_DB)


class SpeckleIncrease(Degradation):
    name = "U1"
    unit = "speckle blend"

    def prepare(self, img, rng):
        a = cv2.GaussianBlur(rng.standard_normal(img.shape).astype(np.float32), (0, 0), self.params["correlation_px"])
        b = cv2.GaussianBlur(rng.standard_normal(img.shape).astype(np.float32), (0, 0), self.params["correlation_px"])
        r = np.hypot(a, b)
        return r / r.mean()

    def apply(self, img, state, s):
        blend = s * self.params["max"]
        return to_display(to_amplitude(img) * (1.0 + blend * (state - 1.0)))

    def physical(self, s):
        return s * self.params["max"]


class GainError(Degradation):
    name = "U2"
    unit = "gain error (dB)"

    def apply(self, img, state, s):
        db = s * self.params["max"]
        depth = np.linspace(0.0, 1.0, img.shape[0], dtype=np.float32)[:, None]
        return img.astype(np.float32) - (255.0 / RANGE_DB) * db * (0.5 + 0.5 * depth)

    def physical(self, s):
        return s * self.params["max"]


class LateralResolutionLoss(Degradation):
    name = "U3"
    unit = "sigma_x / width at the bottom"

    def apply(self, img, state, s):
        h, w = img.shape[:2]
        sigma_bottom = s * self.params["max"] * w
        levels = np.linspace(0.0, sigma_bottom * 1.5, int(self.params["levels"]))
        stack = [img.astype(np.float32)] + [
            cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigmaX=float(sg), sigmaY=float(0.25 * sg)) for sg in levels[1:]]
        depth = np.linspace(0.0, 1.0, h)
        need = sigma_bottom * (0.5 + depth)
        pos = np.clip(need / max(levels[-1], 1e-9), 0, 1) * (len(levels) - 1)
        lo = np.floor(pos).astype(int).clip(0, len(levels) - 2)
        frac = (pos - lo)[:, None]
        stack = np.stack(stack)
        rows = np.arange(h)
        return stack[lo, rows] * (1 - frac) + stack[lo + 1, rows] * frac

    def physical(self, s):
        return s * self.params["max"]


REGISTRY = {"U1": SpeckleIncrease, "U2": GainError, "U3": LateralResolutionLoss}
