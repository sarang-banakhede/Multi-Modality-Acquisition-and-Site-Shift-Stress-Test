import numpy as np
import pandas as pd

from .crossfit import (boot_rho, cell_corr, h1_boot, h1_perm, h1_point, mean_rho, pair_rank_perm, rho_pairs_h2,
                       rho_pairs_h4)
from .meta import fisher, perm_pvalue, percentile_ci, pooled_summary


def make_perms(m, n, rng):
    return rng.permuted(np.tile(np.arange(m), (n, 1)), axis=1)


def loo_q2(zx, zy):
    m = zx.shape[-1]
    b = (zx * zy).mean(-1, keepdims=True)
    h = 1.0 / m + zx ** 2 / m
    press = (((zy - b * zx) / (1 - h)) ** 2).sum(-1)
    return 1.0 - press / m


def h1_cells_table(ctxs, pairs):
    rows = []
    for d, ps in pairs.items():
        if not ps:
            continue
        r = np.mean([cell_corr(*p) for p in ps], axis=0)
        for (g, k), v in zip(ctxs[d].cells, r):
            rows.append({"dataset": d, "degradation": g, "level": k, "pearson": float(v)})
    return pd.DataFrame(rows)


def heterogeneity(cells):
    if cells.empty:
        return {}
    by_ds = cells.groupby("dataset")["pearson"].mean()
    by_deg = cells.groupby(["dataset", "degradation"])["pearson"].mean()
    by_level = cells.groupby("level")["pearson"].mean()
    slope = float(np.polyfit(cells["level"], cells["pearson"], 1)[0]) if cells["level"].nunique() > 1 else np.nan
    return {"sd_between_datasets": float(by_ds.std(ddof=1)) if len(by_ds) > 1 else np.nan,
            "sd_between_degradations": float(by_deg.std(ddof=1)) if len(by_deg) > 1 else np.nan,
            "sd_cells": float(cells["pearson"].std(ddof=1)), "slope_per_level": slope,
            "share_cells_positive": float((cells["pearson"] > 0).mean()),
            "by_level": {int(k): float(v) for k, v in by_level.items()}}


def run_h1(ctxs, cfg, rng, with_tests=True):
    a = cfg.analysis
    n_perm, n_boot = int(a.permutation_reps), int(a.bootstrap_reps)
    pairs = {d: h1_point(c) for d, c in ctxs.items()}
    per_ds = {d: float(np.mean([cell_corr(*p).mean() for p in ps])) for d, ps in pairs.items() if ps}
    if not per_ds:
        return {}, pd.DataFrame()
    cells = h1_cells_table(ctxs, pairs)
    out = {"beta": float(np.mean(list(per_ds.values()))), "per_dataset": per_ds,
           "q2": float(np.mean([loo_q2(*p).mean() for ps in pairs.values() for p in ps])),
           "heterogeneity": heterogeneity(cells), "n_cells": int(len(cells))}
    if with_tests:
        perm_stat = np.zeros(n_perm)
        for d, ps in pairs.items():
            if ps:
                perm_stat += h1_perm(ps, make_perms(ctxs[d].M, n_perm, rng))
        perm_stat /= len(per_ds)
        boots = np.mean([h1_boot(c, n_boot, rng) for d, c in ctxs.items() if pairs[d]], axis=0)
        out["ci_low"], out["ci_high"] = percentile_ci(boots) if len(boots) else (np.nan, np.nan)
        out["p_permutation"] = perm_pvalue(out["beta"], perm_stat)
    return out, cells


def run_rho_test(ctxs, cfg, rng, pair_fn, with_tests=True):
    a = cfg.analysis
    n_perm, n_boot = int(a.permutation_reps), int(a.bootstrap_reps)
    rhos, ses, perm_z = {}, {}, np.zeros(n_perm)
    for d, ctx in ctxs.items():
        pairs = pair_fn(ctx)
        if not pairs:
            continue
        rhos[d] = mean_rho(pairs)
        if not with_tests:
            continue
        boots = boot_rho(lambda c, s, r: pair_fn(c, s, r), ctx, n_boot, rng)
        z = fisher(boots[np.isfinite(boots)]) if np.isfinite(boots).any() else np.array([0.0])
        ses[d] = float(max(z.std(ddof=1), 1e-3)) if len(z) > 1 else 0.3
        perm_z += fisher(pair_rank_perm(pairs, make_perms(ctx.M, n_perm, rng)))
    if not rhos:
        return {}
    obs_z = float(np.mean([fisher(r) for r in rhos.values()]))
    out = {"mean_fisher_z": obs_z, "rho_mean": float(np.tanh(obs_z)), "per_dataset": rhos}
    if with_tests:
        perm_z /= len(rhos)
        out.update(pooled_summary([fisher(r) for r in rhos.values()], [ses[d] for d in rhos]))
        out["p_permutation"] = perm_pvalue(obs_z, perm_z)
        out["per_dataset_se_z"] = ses
    return out


def run_h2(ctxs, cfg, rng, with_tests=True):
    return run_rho_test(ctxs, cfg, rng, rho_pairs_h2, with_tests)


def run_h4(ctxs, cfg, rng, with_tests=True):
    return run_rho_test(ctxs, cfg, rng, rho_pairs_h4, with_tests)


def e1_cells(ctxs, cfg, rng, n_perm=1000):
    rows = []
    for d, ctx in ctxs.items():
        pairs = h1_point(ctx, rank=True)
        if not pairs:
            continue
        m = ctx.M
        obs = np.mean([cell_corr(*p) for p in pairs], axis=0)
        perms = make_perms(m, n_perm, rng)
        null = np.zeros((n_perm, len(obs)))
        for zx, zy in pairs:
            null += np.einsum("cpm,cm->pc", zx[:, perms], zy) / m
        null /= len(pairs)
        for ci, (g, k) in enumerate(ctx.cells):
            p = float((1 + (np.abs(null[:, ci]) >= abs(obs[ci])).sum()) / (n_perm + 1))
            rho = float(obs[ci])
            cls = "predictive" if p < 0.05 and rho > 0 else "inverted" if p < 0.05 else "uninformative"
            rows.append({"dataset": d, "degradation": g, "level": k, "spearman": rho, "p_perm": p, "class": cls})
    return rows
