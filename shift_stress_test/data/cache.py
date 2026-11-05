import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import binary_fill_holes

from ..utils.io import load_json, save_json, save_npz
from ..utils.log import get_logger, progress
from .preprocess import load_mask, load_native, resize_image, resize_mask
from .records import internal_ref, list_set


def is_gray(cfg, name):
    return name in cfg.experiment.grayscale


def _load_pair(args):
    img_path, msk_path, gray, size, ring = args
    native = load_native(img_path, gray)
    mask = load_mask(msk_path, native.shape[:2])
    has_hole = bool((binary_fill_holes(mask) != mask).any()) if ring else True
    return resize_image(native, size), resize_mask(mask, size), has_hole


def build_cache(layout, cfg, ref):
    path = layout.cache_file(ref.name, ref.split or "all")
    if path.exists():
        return path
    names, imgs, msks = list_set(layout, ref, cfg.data.get("limit_per_split"))
    gray = is_gray(cfg, ref.name)
    size = cfg.data.img_size
    ring = ref.name in cfg.experiment.get("ring_targets", [])
    jobs = [(i, m, gray, size, ring) for i, m in zip(imgs, msks)]
    workers = max(1, min(cfg.runtime.workers, os.cpu_count() or 1))
    with ThreadPoolExecutor(workers) as ex:
        pairs = list(progress(ex.map(_load_pair, jobs), total=len(jobs), desc=f"cache {ref.name}/{ref.split or 'all'}"))
    save_npz(path, images=np.stack([p[0] for p in pairs]), masks=np.stack([p[1] for p in pairs]), names=np.array(names))
    if ring:
        bad = [n for n, p in zip(names, pairs) if not p[2]]
        save_json({"set": ref.name, "split": ref.split, "masks_without_hole": bad}, path.with_suffix(".ring.json"))
        get_logger().info(f"ring check {ref.name}/{ref.split or 'all'}: {len(bad)} mask(s) without a hole")
    return path


def load_cache(layout, cfg, ref):
    path = build_cache(layout, cfg, ref)
    with np.load(path) as z:
        return {"images": z["images"], "masks": z["masks"], "names": [str(n) for n in z["names"]]}


def ensure_norm(layout, cfg, dataset):
    path = layout.norm_file(dataset)
    if path.exists():
        return load_json(path)
    train = load_cache(layout, cfg, internal_ref(dataset, "train"))["images"]
    x = train.astype(np.float64) / 255.0
    c = x.shape[-1]
    mean, std = x.mean(axis=(0, 1, 2)), x.std(axis=(0, 1, 2))
    if c == 1:
        mean, std = np.repeat(mean, 3), np.repeat(std, 3)
    out = {"dataset": dataset, "mean": mean.tolist(), "std": np.maximum(std, 1e-6).tolist(), "n_train": int(len(train)),
           "img_size": cfg.data.img_size, "computed_on": "train split only"}
    save_json(out, path)
    get_logger().info(f"norm stats {dataset}: mean {np.round(mean, 4).tolist()} std {np.round(std, 4).tolist()}")
    return out
