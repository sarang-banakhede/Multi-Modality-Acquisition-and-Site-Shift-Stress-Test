import cv2
import numpy as np

from .base import Degradation, from_linear, short_side, to_linear


class UnevenIllumination(Degradation):
    name = "F1"
    unit = "falloff strength"

    def prepare(self, img, rng):
        h, w = img.shape[:2]
        centre = np.array([w / 2, h / 2]) + rng.uniform(-0.3, 0.3, 2) * min(h, w) / 2
        return centre, rng.uniform(*self.params["gamma"])

    def apply(self, img, state, s):
        centre, gamma = state
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.hypot(xx - centre[0], yy - centre[1]) / (0.5 * min(h, w))
        gain = np.clip(1.0 - s * self.params["max"] * np.clip(r, 0, 1.6) ** gamma, 0.0, 1.0)
        return from_linear(to_linear(img) * gain[..., None])

    def physical(self, s):
        return s * self.params["max"]


class CataractHaze(Degradation):
    name = "F2"
    unit = "opacity (1 - t)"

    def prepare(self, img, rng):
        h, w = img.shape[:2]
        factor = 8
        small = cv2.resize(img.astype(np.float32), (max(w // factor, 1), max(h // factor, 1)), interpolation=cv2.INTER_AREA)
        haze = cv2.GaussianBlur(small, (0, 0), self.params["haze_sigma"] * min(h, w) / factor)
        return cv2.resize(haze, (w, h), interpolation=cv2.INTER_LINEAR)

    def apply(self, img, state, s):
        opacity = s * self.params["max"]
        veil = cv2.GaussianBlur(img.astype(np.float32), (0, 0), max(s * self.params["veil_sigma"] * short_side(img), 1e-3))
        return veil * (1.0 - opacity) + state * opacity

    def physical(self, s):
        return s * self.params["max"]


class LensArtefacts(Degradation):
    name = "F3"
    unit = "coverage x opacity"

    def prepare(self, img, rng):
        h, w = img.shape[:2]
        blobs = []
        for _ in range(int(self.params["n_blobs"])):
            r = rng.uniform(*self.params["radius"]) * min(h, w)
            blobs.append((rng.uniform(0.15, 0.85) * w, rng.uniform(0.15, 0.85) * h, r, rng.random() < 0.7, rng.uniform(0.5, 1.0)))
        return blobs

    def apply(self, img, state, s):
        out = img.astype(np.float32).copy()
        h, w = img.shape[:2]
        n = s * len(state)
        for i, (cx, cy, r, dark, op) in enumerate(state[: int(np.ceil(n))]):
            frac = min(1.0, n - i)
            rr = int(2.5 * r) + 2
            x0, x1, y0, y1 = max(int(cx) - rr, 0), min(int(cx) + rr, w), max(int(cy) - rr, 0), min(int(cy) + rr, h)
            if x1 <= x0 or y1 <= y0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            alpha = np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / (r * 0.6) ** 2) * op * frac
            alpha = np.clip(alpha, 0, 0.95)[..., None]
            target = 0.0 if dark else 255.0
            out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - alpha) + target * alpha
        return out

    def physical(self, s):
        return s * self.params["max"]


REGISTRY = {"F1": UnevenIllumination, "F2": CataractHaze, "F3": LensArtefacts}
