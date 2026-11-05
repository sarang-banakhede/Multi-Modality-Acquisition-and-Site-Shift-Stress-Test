import itertools

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm

from ..degrade import GENERIC
from ..metrics import FEATURE_COLUMNS
from .data import probit
from .levels import reach_levels

ATC_VARIANTS = ["band_conf", "conf_fg"]
DOC_VARIANTS = ["msp", "msp_fg"]
ESTIMATORS_DEV = ["B0", "L1"] + [f"L2_{v}" for v in ATC_VARIANTS] + [f"L3_{v}" for v in DOC_VARIANTS]
ESTIMATORS = ESTIMATORS_DEV + ["B1"]


def load_unit(layout, models, dataset, seeds, subset, degradation, ckpt="best", masks=True):
    runs = [(m, s) for m in models for s in seeds]
    dice, feats, packed, ref = [], {k: [] for k in FEATURE_COLUMNS}, [], None
    for m, s in runs:
        df = pd.read_parquet(layout.eval_file(m, dataset, s, ckpt, subset, degradation),
                             columns=["image_id", "level", "dice"] + FEATURE_COLUMNS)
        key = (df["image_id"].to_numpy(), df["level"].to_numpy())
        if ref is None:
            ref = key
        assert (key[0] == ref[0]).all() and (key[1] == ref[1]).all(), "runs disagree on image order"
        dice.append(df["dice"].to_numpy())
        for k in FEATURE_COLUMNS:
            feats[k].append(df[k].to_numpy())
        if masks:
            with np.load(layout.eval_masks(m, dataset, s, ckpt, subset, degradation)) as z:
                packed.append(z["masks"])
    out = {"runs": runs, "image_id": ref[0], "level": ref[1], "dice": np.array(dice),
           "feats": {k: np.array(v) for k, v in feats.items()}}
    if masks:
        out["packed"] = np.stack(packed)
    return out


def pair_index(runs):
    return [(i, j) for i, j in itertools.combinations(range(len(runs)), 2) if runs[i][0] != runs[j][0]]


def pair_agreement(packed, pairs):
    r, n, _ = packed.shape
    ii, jj = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    out = np.zeros((len(pairs), n), np.float32)
    for i in range(n):
        bits = np.unpackbits(packed[:, i], axis=1).astype(np.float32)
        inter = bits @ bits.T
        size = bits.sum(1)
        den = size[:, None] + size[None]
        dice = np.where(den > 0, 2 * inter / np.maximum(den, 1), 1.0)
        out[:, i] = dice[ii, jj]
    return out


class Reference:
    def __init__(self, unit, pairs, stress=None):
        self.D = unit["dice"].mean(1)
        self.stress = stress if stress is not None else self.D.copy()
        self.feats = unit["feats"]
        self.agree = pair_agreement(unit["packed"], pairs).mean(1)
        self.pairs = pairs


class Cell:
    def __init__(self, unit, sel, pairs, with_agree=True):
        self.gidx = unit["gidx"][sel]
        self.dice = unit["dice"][:, sel]
        self.feats = {k: v[:, sel] for k, v in unit["feats"].items()}
        self.agree = pair_agreement(unit["packed"][:, sel], pairs) if with_agree else None
        self.image_id = unit["image_id"][sel]


def _wmean(x, w):
    return x.mean(-1) if w is None else (x * w).sum(-1) / max(w.sum(), 1e-9)


def predict(name, ref, cell, w=None):
    if name == "B0":
        return ref.D.copy(), {}
    if name == "B1":
        return ref.stress.copy(), {}
    if name == "L1":
        a_t = np.clip(_wmean(cell.agree, w), 1e-3, 1 - 1e-3)
        a_v = np.clip(ref.agree, 1e-3, 1 - 1e-3)
        x, y = probit(a_v), probit(a_t)
        b = np.cov(x, y, ddof=0)[0, 1] / (x.var() + 1e-12)
        a = y.mean() - b * x.mean()
        r2 = float(np.corrcoef(x, y)[0, 1] ** 2) if x.std() > 0 and y.std() > 0 else np.nan
        return norm.cdf(a + b * probit(ref.D)), {"r2": r2}
    kind, variant = name.split("_", 1)
    if kind == "L2":
        thr = np.array([np.quantile(ref.feats[variant][r], 1.0 - ref.D[r]) for r in range(len(ref.D))])
        return _wmean((cell.feats[variant] > thr[:, None]).astype(float), w), {}
    gap = ref.feats[variant].mean(1) - _wmean(cell.feats[variant], w)
    return np.clip(ref.D - gap, 0, 1), {}


def model_level(v, n_models, n_seeds, wseed=None):
    v = np.asarray(v).reshape(n_models, n_seeds)
    return v.mean(1) if wseed is None else (v * wseed).sum(1)


def cell_metrics(pred_runs, truth_runs, n_models, n_seeds, wseed=None):
    pm, tm = model_level(pred_runs, n_models, n_seeds, wseed), model_level(truth_runs, n_models, n_seeds, wseed)
    star = int(np.argmax(pm))
    tau, p = kendalltau(pm, tm) if np.std(pm) > 0 and np.std(tm) > 0 else (np.nan, np.nan)
    return {"regret": float(tm.max() - tm[star]), "top2": bool(star in np.argsort(-tm)[:2]), "tau": float(tau),
            "tau_p": float(p), "mae": float(np.abs(pred_runs - truth_runs).mean())}


def evaluate_cell(ref, cell, n_models, n_seeds, names=ESTIMATORS, w=None, wseed=None):
    truth = _wmean(cell.dice, w)
    rows = []
    for name in names:
        pred, extra = predict(name, ref, cell, w)
        met = cell_metrics(pred, truth, n_models, n_seeds, wseed)
        if name == "B1":
            met["mae"] = np.nan
        rows.append({"estimator": name, **met, **extra})
    return rows


def val_levels(layout, cfg, dataset):
    return {g: reach_levels(layout, cfg, dataset, "val", g)[0] for g in GENERIC}


def stress_score(layout, models, dataset, seeds, kmax_of):
    runs = [(m, s) for m in models for s in seeds]
    scores = []
    for deg in GENERIC:
        if kmax_of.get(deg, 0) == 0:
            continue
        unit = load_unit(layout, models, dataset, seeds, "val", deg, masks=False)
        keep = unit["level"] <= kmax_of[deg]
        scores.append(unit["dice"][:, keep].mean(1))
    return np.mean(scores, axis=0) if scores else np.zeros(len(runs))


def reference_for(layout, cfg, models, dataset, seeds, pairs):
    unit = load_unit(layout, models, dataset, seeds, "val", "clean")
    return Reference(unit, pairs, stress_score(layout, models, dataset, seeds, val_levels(layout, cfg, dataset)))


def iter_cells(layout, models, dataset, seeds, subset, degradations, pairs, kmax_of=None, with_agree=True, names_index=None):
    for deg in degradations:
        if kmax_of is not None and kmax_of.get(deg, 0) == 0:
            continue
        unit = load_unit(layout, models, dataset, seeds, subset, deg)
        levels = np.unique(unit["level"])
        index = names_index or {n: i for i, n in enumerate(dict.fromkeys(unit["image_id"]))}
        unit["gidx"] = np.array([index[n] for n in unit["image_id"]])
        for lv in levels:
            if kmax_of is not None and lv > kmax_of[deg]:
                continue
            sel = unit["level"] == lv
            yield deg, int(lv), Cell(unit, sel, pairs, with_agree)
        del unit


def dev_cells(layout, cfg, models, datasets, seeds):
    rows = []
    for d in datasets:
        runs = [(m, s) for m in models for s in seeds]
        pairs = pair_index(runs)
        ref = reference_for(layout, cfg, models, d, seeds, pairs)
        for deg, lv, cell in iter_cells(layout, models, d, seeds, "val", GENERIC, pairs, val_levels(layout, cfg, d)):
            for r in evaluate_cell(ref, cell, len(models), len(seeds), ESTIMATORS_DEV):
                rows.append({"dataset": d, "degradation": deg, "level": lv, **r})
    return pd.DataFrame(rows)


def choose_primary(dev):
    agg = dev.groupby(["dataset", "estimator"])[["regret", "mae"]].mean().groupby("estimator").mean()
    agg["top2"] = dev.groupby("estimator")["top2"].mean()
    picks = {}
    for kind in ("L2", "L3"):
        sub = agg[agg.index.str.startswith(kind)].sort_values(["regret", "mae"])
        picks[kind] = sub.index[0]
    candidates = [picks["L2"], picks["L3"]]
    primary = agg.loc[candidates].sort_values(["regret", "mae"]).index[0]
    return {"L2_variant": picks["L2"], "L3_variant": picks["L3"], "primary": primary,
            "table": agg.reset_index().to_dict(orient="records")}
