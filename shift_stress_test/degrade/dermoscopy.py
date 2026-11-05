import cv2
import numpy as np

from .base import Degradation, from_linear, short_side, to_linear


def _bezier(p0, p1, p2, p3, n=40):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3


class HairOcclusion(Degradation):
    name = "D1"
    unit = "strands drawn"

    def prepare(self, img, rng):
        h, w = img.shape[:2]
        strands = []
        for _ in range(int(self.params["max"])):
            p0 = rng.uniform([0, 0], [w, h])
            ang = rng.uniform(0, 2 * np.pi)
            length = rng.uniform(0.4, 0.9) * max(h, w)
            p3 = p0 + length * np.array([np.cos(ang), np.sin(ang)])
            p1 = p0 + (p3 - p0) * 0.33 + rng.normal(0, 0.08 * length, 2)
            p2 = p0 + (p3 - p0) * 0.66 + rng.normal(0, 0.08 * length, 2)
            width = rng.uniform(*self.params["width"]) * min(h, w)
            colour = rng.uniform([15, 10, 5], [70, 50, 40])
            strands.append((np.round(_bezier(p0, p1, p2, p3)).astype(np.int32), max(width, 1.0), colour))
        return strands

    def apply(self, img, state, s):
        out = img.astype(np.float32).copy()
        h, w = img.shape[:2]
        n = s * len(state)
        for i, (pts, width, colour) in enumerate(state[: int(np.ceil(n))]):
            frac = min(1.0, n - i)
            x0, y0 = np.maximum(pts.min(0) - int(width) - 4, 0)
            x1, y1 = np.minimum(pts.max(0) + int(width) + 5, [w, h])
            if x1 <= x0 or y1 <= y0:
                continue
            mask = np.zeros((y1 - y0, x1 - x0), np.float32)
            cv2.polylines(mask, [pts - [x0, y0]], False, 1.0, thickness=max(1, int(round(width))), lineType=cv2.LINE_AA)
            mask = cv2.GaussianBlur(mask, (0, 0), 0.7) * frac * 0.95
            out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - mask[..., None]) + colour * mask[..., None]
        return out


class DarkCorners(Degradation):
    name = "D2"
    unit = "aperture radius / short side"

    def apply(self, img, state, s):
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dist = np.hypot(yy - h / 2, xx - w / 2)
        half_diag = np.hypot(h, w) / 2
        r_min = self.params["min_radius"] * short_side(img)
        radius = (1 - s) * half_diag + s * r_min
        soft = 0.04 * short_side(img)
        inside = np.clip((radius - dist) / soft, 0, 1)
        fall = 1 - self.params["falloff"] * s * np.clip(dist / radius, 0, 1) ** 2
        gain = (inside * fall)[..., None]
        return from_linear(to_linear(img) * gain)

    def physical(self, s):
        return (1 - s) * 1.0 + s * self.params["min_radius"]


def _kelvin_rgb(t):
    t = t / 100.0
    r = 255 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = 99.4708025861 * np.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255 if t >= 66 else (0 if t <= 19 else 138.5177312231 * np.log(t - 10) - 305.0447927307)
    return np.clip(np.array([r, g, b]), 1, 255) / 255.0


class IlluminationShift(Degradation):
    name = "D3"
    unit = "shift t (0-1)"

    def prepare(self, img, rng):
        return self.params["warm_k"] if rng.random() < 0.5 else self.params["cool_k"]

    def apply(self, img, state, s):
        target = 6500 + s * (state - 6500)
        gains = _kelvin_rgb(target) / _kelvin_rgb(6500)
        gains = gains / gains.max()
        ev = -s * self.params["max_ev"]
        return from_linear(to_linear(img) * gains[None, None] * 2.0 ** ev)

    def physical(self, s):
        return float(s)


REGISTRY = {"D1": HairOcclusion, "D2": DarkCorners, "D3": IlluminationShift}
