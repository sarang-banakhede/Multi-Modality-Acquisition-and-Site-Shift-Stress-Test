import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .confirmatory import make_perms
from .crossfit import seed_weights, standardise
from .meta import perm_pvalue, percentile_ci
from .scores import area


def _ext_arrays(ctx, split, side, w=None, rng=None, ext_idx=None):
    c = ctx.half_clusters(split, side)
    idx = c.draw(rng) if rng is not None else c.idx
    if len(idx) == 0:
        return None
    x = ctx.link(ctx.clean(idx, w))
    y = ctx.link(ctx.ext_dice(ext_idx, w))
    return standardise(x)[None], standardise(y)[None]


def beta_ext(ctxs, cfg, rng, with_tests=True):
    n_perm, n_boot = int(cfg.analysis.permutation_reps), int(cfg.analysis.bootstrap_reps)
    pairs = {d: [a for s in range(len(c.halves)) for side in (0, 1) if (a := _ext_arrays(c, s, side)) is not None]
             for d, c in ctxs.items() if c.ext is not None}
    per_ds = {d: float(np.mean([(zx * zy).mean() for zx, zy in ps])) for d, ps in pairs.items() if ps}
    if not per_ds:
        return {}
    out = {"beta_ext": float(np.mean(list(per_ds.values()))), "per_dataset": per_ds}
    if with_tests:
        stat = np.zeros(n_perm)
        for d, ps in pairs.items():
            if ps:
                perms = make_perms(ctxs[d].M, n_perm, rng)
                stat += np.mean([np.einsum("cpm,cm->pc", zx[:, perms], zy).mean(1) / zx.shape[-1] for zx, zy in ps], axis=0)
        stat /= len(per_ds)
        boots = []
        for _ in range(n_boot):
            vals = []
            for d, c in ctxs.items():
                if not pairs.get(d):
                    continue
                w = seed_weights(rng, c.M, c.S)
                a = _ext_arrays(c, int(rng.integers(len(c.halves))), int(rng.integers(2)), w, rng, c.ext_clusters.draw(rng))
                if a is not None:
                    vals.append((a[0] * a[1]).mean())
            boots.append(np.mean(vals) if vals else np.nan)
        out["ci_low"], out["ci_high"] = percentile_ci(np.array(boots)[np.isfinite(boots)]) if np.isfinite(boots).any() else (np.nan, np.nan)
        out["p_permutation"] = perm_pvalue(out["beta_ext"], stat)
    return out


def _metrics(score, truth):
    star = int(np.argmax(score))
    tau = kendalltau(score, truth)[0] if np.std(score) > 0 and np.std(truth) > 0 else np.nan
    return {"regret": float(truth.max() - truth[star]), "top2": bool(star in np.argsort(-truth)[:2]), "tau": float(tau),
            "selected": int(star)}


def external_selection(T, ext, labelfree_cells, primary):
    rows = []
    d = T.tensor("dice")
    seed_mean = np.nanmean(d, axis=1)
    clean_test = np.nanmean(seed_mean[:, 0], axis=-1)
    auc = []
    for g in T.degs:
        idx = np.nonzero(T.complete(g))[0]
        if len(idx):
            auc.append(area(np.nanmean(seed_mean[:, [0] + T.cond_ids(g)][..., idx], axis=-1)))
    auc_abs = np.mean(auc, axis=0) if auc else np.full(len(T.models), np.nan)
    sources = np.unique(ext.sources)
    ext_seed = np.nanmean(ext.dice, axis=1)
    sub = labelfree_cells[(labelfree_cells["dataset"] == T.dataset) & (labelfree_cells["kind"] == "external")]
    for src in list(sources) + (["all"] if len(sources) > 1 else []):
        sel = np.ones(len(ext.names), bool) if src == "all" else ext.sources == src
        truth = np.nanmean(ext_seed[:, sel], axis=-1)
        scorers = {"B0_clean_val": sub[sub["estimator"] == "B0"], "B1_stress_val": sub[sub["estimator"] == "B1"]}
        for name, row in scorers.items():
            if len(row) and (src == "all" or len(sources) == 1):
                rows.append({"dataset": T.dataset, "source": src, "selector": name, **{k: float(row[k].iloc[0]) for k in ("regret", "tau")},
                             "top2": bool(row["top2"].iloc[0])})
        for name, score in (("clean_test", clean_test), ("synthetic_test_auc_abs", auc_abs)):
            if np.isfinite(score).all():
                rows.append({"dataset": T.dataset, "source": src, "selector": name, **{k: v for k, v in _metrics(score, truth).items() if k != "selected"}})
        prim = sub[sub["estimator"] == primary]
        if len(prim) and (src == "all" or len(sources) == 1):
            rows.append({"dataset": T.dataset, "source": src, "selector": f"primary_{primary}", "regret": float(prim["regret"].iloc[0]),
                         "tau": float(prim["tau"].iloc[0]), "top2": bool(prim["top2"].iloc[0])})
    return pd.DataFrame(rows)


def best_last_agreement(clean_table):
    rows = []
    if clean_table.empty:
        return pd.DataFrame()
    for (d, subset), grp in clean_table.groupby(["dataset", "subset"]):
        piv = grp.pivot_table(index=["model", "seed"], columns="ckpt", values="dice")
        if {"best", "last"} <= set(piv.columns) and len(piv) > 2:
            rows.append({"dataset": d, "subset": subset, "spearman_best_vs_last": float(piv["best"].corr(piv["last"], method="spearman")),
                         "mean_dice_best": float(piv["best"].mean()), "mean_dice_last": float(piv["last"].mean())})
    return pd.DataFrame(rows)

