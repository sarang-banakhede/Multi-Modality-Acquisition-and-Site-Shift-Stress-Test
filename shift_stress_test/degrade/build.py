import multiprocessing as mp
import shutil

import numpy as np
import pandas as pd

from ..config import Config
from ..data.cache import is_gray
from ..data.preprocess import load_native
from ..data.records import image_id, internal_ref, list_set
from ..utils.io import load_json, save_json, save_npz
from ..utils.log import get_logger, progress
from .atlas import make_atlas
from .base import build_degradation, degradations_for
from .match import match_image


def _work(args):
    cfg_dict, dataset, name, paths, gray, targets = args
    cfg = Config(cfg_dict)
    deg = build_degradation(name, cfg)
    parts = [match_image(deg, load_native(p, gray), dataset, image_id(p.name), targets, cfg.data.img_size, cfg.degrade)
             for p in paths]
    return {k: np.stack([p[k] for p in parts], axis=1 if k == "images" else 0) if k == "images"
            else np.stack([p[k] for p in parts], axis=1) for k in parts[0]}


def _targets(cfg, layout):
    path = layout.out / "degrade_config.json"
    n = load_json(path)["n_levels"] if path.exists() else len(cfg.degrade.levels)
    return list(cfg.degrade.levels)[:n]


def build_one(cfg, layout, dataset, split, name, targets, pool):
    log = get_logger()
    final = layout.degraded_file(dataset, split, name)
    if final.exists():
        return
    names, paths, _ = list_set(layout, internal_ref(dataset, split), cfg.data.get("limit_per_split"))
    gray = is_gray(cfg, dataset)
    parts_dir = final.parent / "_parts" / name
    parts_dir.mkdir(parents=True, exist_ok=True)
    size = cfg.degrade.chunk_size
    chunks = [(k, list(range(i, min(i + size, len(paths))))) for k, i in enumerate(range(0, len(paths), size))]
    todo = [c for c in chunks if not (parts_dir / f"chunk_{c[0]:04d}.npz").exists()]
    jobs = [(cfg.to_dict(), dataset, name, [paths[i] for i in idx], gray, targets) for _, idx in todo]
    bar = progress(total=len(paths), desc=f"degrade {dataset}/{split}/{name}")
    bar.update(len(paths) - sum(len(idx) for _, idx in todo))
    for (k, idx), res in zip(todo, pool.imap(_work, jobs)):
        save_npz(parts_dir / f"chunk_{k:04d}.npz", **res)
        bar.update(len(idx))
    bar.close()
    merged = {}
    for k, _ in chunks:
        with np.load(parts_dir / f"chunk_{k:04d}.npz") as z:
            for key in z.files:
                merged.setdefault(key, []).append(z[key])
    arrays = {k: np.concatenate(v, axis=1) for k, v in merged.items()}
    save_npz(final, **arrays, names=np.array(names), levels=np.array(targets))
    shutil.rmtree(parts_dir)
    reach = arrays["reachable"].mean()
    log.info(f"built {dataset}/{split}/{name}: {len(names)} images x {len(targets)} levels, reachable {reach:.1%}")


def reachability_table(layout, dataset, split, names):
    rows = []
    for name in names:
        f = layout.degraded_file(dataset, split, name)
        if not f.exists():
            continue
        with np.load(f) as z:
            reach, ssim, phys, levels = z["reachable"], z["ssim"], z["physical"], z["levels"]
        for k, target in enumerate(levels):
            ok = reach[k]
            rows.append({"dataset": dataset, "split": split, "degradation": name, "level": k + 1, "target_ssim": target,
                         "reachable_share": float(ok.mean()),
                         "achieved_ssim_median": float(np.nanmedian(ssim[k][ok])) if ok.any() else np.nan,
                         "physical_median": float(np.nanmedian(phys[k][ok])) if ok.any() else np.nan,
                         "physical_q25": float(np.nanpercentile(phys[k][ok], 25)) if ok.any() else np.nan,
                         "physical_q75": float(np.nanpercentile(phys[k][ok], 75)) if ok.any() else np.nan})
    return pd.DataFrame(rows)


def decide_levels(cfg, layout, datasets):
    path = layout.out / "degrade_config.json"
    if path.exists():
        return load_json(path)
    levels = list(cfg.degrade.levels)
    share = {}
    for d in datasets:
        for n in degradations_for(d):
            with np.load(layout.degraded_file(d, "val", n)) as z:
                share[f"{d}/{n}"] = z["reachable"].mean(axis=1).tolist()
    out = {"levels": levels, "n_levels": len(levels), "min_reachable": cfg.degrade.min_reachable,
           "reachable_share_validation": share,
           "ssim": {"window": "gaussian sigma 1.5", "mask": "clean luminance > 10, eroded 3 px", "size": cfg.data.img_size}}
    save_json(out, path)
    get_logger().info(f"degradation levels frozen: {levels}")
    return out


def run_degrade(cfg, layout, datasets):
    pool = mp.get_context("fork").Pool(cfg.runtime.workers)
    try:
        full = list(cfg.degrade.levels)
        for dataset in datasets:
            for name in degradations_for(dataset):
                build_one(cfg, layout, dataset, "val", name, full, pool)
        decide_levels(cfg, layout, datasets)
        targets = _targets(cfg, layout)
        for dataset in datasets:
            for name in degradations_for(dataset):
                build_one(cfg, layout, dataset, "test", name, targets, pool)
    finally:
        pool.close()
        pool.join()
    tables = []
    for dataset in datasets:
        for split in ("val", "test"):
            t = reachability_table(layout, dataset, split, degradations_for(dataset))
            tables.append(t)
            if len(t):
                t.to_csv(layout.degrade_report(dataset, split), index=False)
        make_atlas(cfg, layout, dataset)
    return pd.concat(tables, ignore_index=True)
