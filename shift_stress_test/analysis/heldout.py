import numpy as np
import pandas as pd

from .crossfit import Clusters
from .labelfree import evaluate_cell, iter_cells, pair_index, reference_for
from .meta import benjamini_hochberg, percentile_ci


class SlimCell:
    def __init__(self, cell, primary):
        self.gidx, self.dice, self.image_id = cell.gidx, cell.dice, cell.image_id
        self.feats = {k: v for k, v in cell.feats.items() if primary.startswith("L") and primary.endswith(k)}
        self.agree = cell.agree if primary == "L1" else None


def collect(layout, cfg, models, seeds, T, ext, primary):
    rows, store = [], {"test": [], "external": []}
    runs = [(m, s) for m in models for s in seeds]
    pairs = pair_index(runs)
    ref = reference_for(layout, cfg, models, T.dataset, seeds, pairs)
    m, s = len(models), len(seeds)
    test_index = {n: i for i, n in enumerate(T.names)}
    for kind, degs in (("heldout_specific", T.specific), ("in_family", T.generic)):
        for deg, lv, cell in iter_cells(layout, models, T.dataset, seeds, "test", degs, pairs, T.K, names_index=test_index):
            for r in evaluate_cell(ref, cell, m, s):
                rows.append({"dataset": T.dataset, "kind": kind, "degradation": deg, "level": lv, **r})
            if kind == "heldout_specific":
                store["test"].append((deg, lv, SlimCell(cell, primary)))
    ext_index = {n: i for i, n in enumerate(ext.names)}
    for deg, lv, cell in iter_cells(layout, models, T.dataset, seeds, "external", ["clean"], pairs, None, names_index=ext_index):
        for r in evaluate_cell(ref, cell, m, s):
            rows.append({"dataset": T.dataset, "kind": "external", "degradation": deg, "level": lv, **r})
        store["external"].append((deg, lv, SlimCell(cell, primary)))
    return pd.DataFrame(rows), {"ref": ref, "cells": store, "groups": T.groups, "ext_groups": ext.groups,
                                "n_test": len(T.names), "n_ext": len(ext.names), "m": m, "s": s}


def _components(entry, name, mild, w_test, w_ext, wseed):
    out = {"specific_mild": [], "specific_all": [], "external": []}
    for kind, w in (("test", w_test), ("external", w_ext)):
        for deg, lv, cell in entry["cells"][kind]:
            ww = None if w is None else w[cell.gidx]
            rows = evaluate_cell(entry["ref"], cell, entry["m"], entry["s"], ["B0", name], ww, wseed)
            delta = rows[1]["regret"] - rows[0]["regret"]
            if kind == "external":
                out["external"].append(delta)
            else:
                out["specific_all"].append(delta)
                if lv <= mild:
                    out["specific_mild"].append(delta)
    return {k: float(np.mean(v)) if v else np.nan for k, v in out.items()}


def _combine(c):
    parts = [v for v in (c["specific_mild"], c["external"]) if np.isfinite(v)]
    return float(np.mean(parts)) if parts else np.nan


def h5_bootstrap(store, name, mild, reps, rng):
    def stat(w_fn):
        comps = [_components(e, name, mild, *w_fn(d, e)) for d, e in store.items()]
        mean = {k: float(np.nanmean([c[k] for c in comps])) for k in comps[0]}
        mean["combined"] = float(np.nanmean([_combine(c) for c in comps]))
        return mean

    obs = stat(lambda d, e: (None, None, None))
    clusters = {d: (Clusters(np.arange(e["n_test"]), e["groups"]), Clusters(np.arange(e["n_ext"]), e["ext_groups"]))
                for d, e in store.items()}

    def drawn(d, e):
        wt = np.bincount(clusters[d][0].draw(rng), minlength=e["n_test"]).astype(float)
        we = np.bincount(clusters[d][1].draw(rng), minlength=e["n_ext"]).astype(float)
        return wt, we, rng.multinomial(e["s"], np.full(e["s"], 1.0 / e["s"]), size=e["m"]) / e["s"]

    boots = [stat(drawn) for _ in range(reps)]
    out = {"estimator": name, "reps": reps}
    for key in obs:
        b = np.array([x[key] for x in boots]) if reps else np.array([])
        lo, hi = percentile_ci(b[np.isfinite(b)]) if len(b) and np.isfinite(b).any() else (np.nan, np.nan)
        p = float(min(1.0, 2 * (1 + min((b <= 0).sum(), (b >= 0).sum())) / (len(b) + 1))) if len(b) else np.nan
        out[key] = {"delta_regret": obs[key], "ci_low": lo, "ci_high": hi, "p_bootstrap": p}
    return out


def summarise(cells_df, primary):
    out = cells_df.groupby(["kind", "estimator"])[["regret", "mae", "tau", "top2"]].mean().reset_index()
    out["is_primary"] = out["estimator"] == primary
    p = cells_df.dropna(subset=["tau_p"]).copy()
    if len(p):
        p["tau_p_bh"] = benjamini_hochberg(p["tau_p"].to_numpy())
    return out, p
