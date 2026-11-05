import numpy as np

from ..data.preprocess import luminance, resize_image
from .ssim import fov_mask, masked_ssim


def _refine(probe, lo, hi, s_lo, s_hi, target, tol, max_iter):
    best = (abs(s_lo - target), lo) if abs(s_lo - target) < abs(s_hi - target) else (abs(s_hi - target), hi)
    for i in range(max_iter):
        if hi - lo < 1e-5:
            break
        if i % 2 == 0 and s_lo > s_hi:
            mid = lo + (hi - lo) * (s_lo - target) / (s_lo - s_hi)
            mid = min(max(mid, lo + 0.05 * (hi - lo)), hi - 0.05 * (hi - lo))
        else:
            mid = 0.5 * (lo + hi)
        val = probe(mid)
        if abs(val - target) < best[0]:
            best = (abs(val - target), mid)
        if abs(val - target) <= tol:
            break
        if val > target:
            lo, s_lo = mid, val
        else:
            hi, s_hi = mid, val
    return best[1], best[0]


def match_image(deg, native, dataset, image_id, targets, size, mcfg):
    state = deg.prepare(native, deg.rng(dataset, image_id))
    clean = resize_image(native, size)
    lum_c = luminance(clean)
    mask = fov_mask(lum_c)
    cache = {}

    def probe(s):
        key = round(float(s), 7)
        if key not in cache:
            d224 = resize_image(deg(native, state, key), size)
            cache[key] = (masked_ssim(lum_c, luminance(d224), mask), d224)
        return cache[key][0]

    is_jpeg = deg.name == "G3"
    tol = mcfg.jpeg_tolerance if is_jpeg else mcfg.tolerance
    accept = mcfg.jpeg_tolerance if is_jpeg else 2.0 * mcfg.tolerance
    grid = np.geomspace(0.01, 1.0, mcfg.coarse_points)
    curve_s = np.concatenate([[0.0], grid])
    curve_v = np.minimum.accumulate(np.concatenate([[1.0], [probe(s) for s in grid]]))
    n = len(targets)
    out = {"images": np.zeros((n, size, size, clean.shape[-1]), np.uint8), "ssim": np.full(n, np.nan),
           "strength": np.full(n, np.nan), "physical": np.full(n, np.nan), "reachable": np.zeros(n, bool)}
    for k, target in enumerate(targets):
        if curve_v[-1] > target + accept:
            continue
        j = int(np.argmax(curve_v <= target))
        lo, hi = float(curve_s[max(j - 1, 0)]), float(curve_s[j])
        s_lo, s_hi = float(curve_v[max(j - 1, 0)]), float(curve_v[j])
        if j == 0:
            lo, s_lo = 0.0, 1.0
        s, err = _refine(probe, lo, hi, s_lo, s_hi, target, tol, mcfg.max_iterations)
        probe(s)
        if err > accept:
            continue
        key = round(float(s), 7)
        out["images"][k] = cache[key][1]
        out["ssim"][k], out["strength"][k] = cache[key][0], key
        out["physical"][k], out["reachable"][k] = deg.physical(key), True
    return out
