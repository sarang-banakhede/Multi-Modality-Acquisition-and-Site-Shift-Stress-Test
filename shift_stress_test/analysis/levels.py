import numpy as np
import pandas as pd

from ..degrade import GENERIC, SPECIFIC
from ..utils.io import load_json


def planned_levels(layout, cfg):
    path = layout.out / "degrade_config.json"
    return load_json(path)["levels"] if path.exists() else list(cfg.degrade.levels)


def reach_levels(layout, cfg, dataset, split, degradation):
    levels = planned_levels(layout, cfg)
    f = layout.degraded_file(dataset, split, degradation)
    if f.exists():
        with np.load(f) as z:
            share = z["reachable"][: len(levels)].mean(axis=1)
    else:
        t = pd.read_csv(layout.degrade_report(dataset, split))
        share = t[t["degradation"] == degradation].sort_values("level")["reachable_share"].to_numpy()[: len(levels)]
    ok = share >= cfg.degrade.min_reachable
    k = len(ok) if ok.all() else int(np.argmin(ok))
    return k if k >= cfg.analysis.min_levels else 0, share


def included(layout, cfg, dataset, split):
    out = {}
    for g in GENERIC + SPECIFIC[dataset]:
        k, share = reach_levels(layout, cfg, dataset, split, g)
        out[g] = {"K": k, "share": share.tolist()}
    return out
