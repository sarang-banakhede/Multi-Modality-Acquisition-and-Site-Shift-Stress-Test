import cv2
import numpy as np

from .base import Degradation, from_linear, short_side, to_linear


def trajectory(rng, n=64, anxiety=0.005):
    pos = np.zeros((n, 2))
    vel = rng.normal(0, 1, 2)
    vel = vel / (np.linalg.norm(vel) + 1e-9)
    for i in range(1, n):
        vel = vel + anxiety * n * 0.02 * rng.normal(0, 1, 2) + 0.15 * rng.normal(0, 1, 2)
        vel = vel / (np.linalg.norm(vel) + 1e-9)
        pos[i] = pos[i - 1] + vel / n
    return pos - pos.mean(0)


def motion_kernel(path, length):
    size = int(max(3, round(length))) | 1
    k = np.zeros((size, size), np.float32)
    pts = path / max(np.abs(path).max(), 1e-9) * (size - 1) / 2 + (size - 1) / 2
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(k, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), 1.0, 1, cv2.LINE_AA)
    return k / k.sum()


class MotionBlur(Degradation):
    name = "E1"
    unit = "path length / short side"

    def prepare(self, img, rng):
        return trajectory(rng)

    def apply(self, img, state, s):
        length = s * self.params["max"] * short_side(img)
        if length < 2:
            return img
        return cv2.filter2D(img, -1, motion_kernel(state, length), borderType=cv2.BORDER_REFLECT)

    def physical(self, s):
        return s * self.params["max"]


class SpecularHighlights(Degradation):
    name = "E2"
    unit = "covered area fraction"

    def prepare(self, img, rng):
        h, w = img.shape[:2]
        n = int(self.params["n_blobs"])
        area = self.params["max"] * h * w / n
        blobs = []
        for _ in range(n):
            a = np.sqrt(area / np.pi) * rng.uniform(0.5, 1.6)
            b = a * rng.uniform(0.4, 1.0)
            blobs.append((rng.uniform(0, w), rng.uniform(0, h), a, b, rng.uniform(0, np.pi)))
        return blobs

    def apply(self, img, state, s):
        out = img.astype(np.float32).copy()
        h, w = img.shape[:2]
        n = s * len(state)
        for i, (cx, cy, a, b, th) in enumerate(state[: int(np.ceil(n))]):
            frac = min(1.0, n - i)
            r = int(3 * max(a, b)) + 2
            x0, x1, y0, y1 = max(int(cx) - r, 0), min(int(cx) + r, w), max(int(cy) - r, 0), min(int(cy) + r, h)
            if x1 <= x0 or y1 <= y0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            dx, dy = xx - cx, yy - cy
            u = dx * np.cos(th) + dy * np.sin(th)
            v = -dx * np.sin(th) + dy * np.cos(th)
            alpha = np.clip(np.exp(-0.5 * ((u / a) ** 2 + (v / b) ** 2)) * 1.8, 0, 1) * frac
            out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - alpha[..., None]) + 255.0 * alpha[..., None]
        return out

    def physical(self, s):
        return s * self.params["max"]


class LowLightNoise(Degradation):
    name = "E3"
    unit = "fraction of light lost"

    def prepare(self, img, rng):
        return rng.standard_normal(img.shape).astype(np.float32)

    def apply(self, img, state, s):
        full, read = self.params["full_well"], self.params["read_noise"]
        keep = 1.0 - s * self.params["max"]
        lam = to_linear(img) * keep * full
        counts = lam + np.sqrt(np.maximum(lam, 0.0)) * state + read * state[::-1]
        return from_linear(np.maximum(counts, 0.0) / (keep * full))

    def physical(self, s):
        return s * self.params["max"]


REGISTRY = {"E1": MotionBlur, "E2": SpecularHighlights, "E3": LowLightNoise}
