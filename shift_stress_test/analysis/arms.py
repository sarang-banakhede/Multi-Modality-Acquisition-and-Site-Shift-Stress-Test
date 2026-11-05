import numpy as np
import pandas as pd

from .crossfit import Clusters
from .data import DatasetTensor, ExternalTensor, probit
from .meta import percentile_ci
from .scores import area


def line_residual(x_s, y_s, x_o, y_o):
    xs, ys = probit(x_s), probit(y_s)
    b = np.cov(xs, ys, ddof=0)[0, 1] / (xs.var() + 1e-12)
    a = ys.mean() - b * xs.mean()
    return probit(y_o) - (a + b * probit(x_o))


def arm_stats(sm_s, sm_p, ext_s, ext_p, keep, idx, ext_idx):
    ds, dp = sm_s[..., idx].mean(-1), sm_p[..., idx].mean(-1)
    es, ep = ext_s[:, ext_idx].mean(-1), ext_p[:, ext_idx].mean(-1)
    er = np.stack([line_residual(ds[:, 0], ds[:, k], dp[:, 0], dp[:, k]) for k in range(1, ds.shape[1])], axis=1)
    er = np.concatenate([np.zeros((len(dp), 1)), er], axis=1)
    return {"clean_dice": dp[:, 0] - ds[keep, 0],
            "auc_loss": (dp[:, 0] - area(dp)) - (ds[keep, 0] - area(ds[keep])),
            "auc_er_vs_scratch_line": area(er), "external_dice": ep - es[keep],
            "er_ext_vs_scratch_line": line_residual(ds[:, 0], es, dp[:, 0], ep)}


def pretrained_arm(layout, cfg, datasets, seeds, scratch, pretrained, levels, rng):
    rows = []
    reps = min(int(cfg.analysis.bootstrap_reps), 500)
    keep = [scratch.index(m[: -len("_pt")]) for m in pretrained]
    for d in datasets:
        ts = DatasetTensor(layout, cfg, d, scratch, seeds, levels)
        pt_seeds = list(cfg.experiment.pretrained_seeds)
        tp = DatasetTensor(layout, cfg, d, pretrained, pt_seeds, levels)
        xs, xp = ExternalTensor(layout, cfg, d, scratch, seeds), ExternalTensor(layout, cfg, d, pretrained, pt_seeds)
        ext_s, ext_p = np.nanmean(xs.dice, axis=1), np.nanmean(xp.dice, axis=1)
        ext_cl = Clusters(np.arange(len(xs.names)), xs.groups)
        for group, degs in (("generic", ts.generic), ("specific", ts.specific)):
            for g in [g for g in degs if g in tp.K]:
                idx = np.nonzero(ts.complete(g) & tp.complete(g))[0]
                if len(idx) < 3:
                    continue
                ids = [0] + ts.cond_ids(g)
                sm_s = np.nanmean(ts.tensor("dice")[:, :, ids], axis=1)[..., idx]
                sm_p = np.nanmean(tp.tensor("dice")[:, :, ids], axis=1)[..., idx]
                cl = Clusters(np.arange(len(idx)), ts.groups[idx])
                point = arm_stats(sm_s, sm_p, ext_s, ext_p, keep, np.arange(len(idx)), np.arange(len(xs.names)))
                boots = [arm_stats(sm_s, sm_p, ext_s, ext_p, keep, cl.draw(rng), ext_cl.draw(rng)) for _ in range(reps)]
                for j, name in enumerate(pretrained):
                    for key, val in point.items():
                        b = [x[key][j] for x in boots]
                        lo, hi = percentile_ci(b) if b else (np.nan, np.nan)
                        rows.append({"dataset": d, "model": name, "group": group, "degradation": g, "metric": key,
                                     "delta": float(val[j]), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)
