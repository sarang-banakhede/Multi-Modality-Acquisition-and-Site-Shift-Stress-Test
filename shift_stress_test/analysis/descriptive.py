import itertools

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, spearmanr, wilcoxon

from .crossfit import Clusters, DatasetContext, cell_corr, h1_point
from .meta import holm, percentile_ci
from .scores import area, curve_scores


def selection_regret(T, rng, reps):
    rows = []
    seed_mean = np.nanmean(T.tensor("dice"), axis=1)
    for g in T.degs:
        idx = np.nonzero(T.complete(g))[0]
        if len(idx) == 0:
            continue
        ids = [0] + T.cond_ids(g)
        sm = seed_mean[:, ids][..., idx]
        d = sm.mean(-1)
        best_clean = int(np.argmax(d[:, 0]))
        cl = Clusters(np.arange(len(idx)), T.groups[idx])
        draws = [cl.draw(rng) for _ in range(reps)]
        pairs = list(itertools.combinations(range(len(T.models)), 2))
        for k in range(1, T.K[g] + 1):
            flips = np.mean([np.sign(d[i, 0] - d[j, 0]) != np.sign(d[i, k] - d[j, k]) for i, j in pairs])
            stay = np.mean([np.argmax(sm[:, k][:, dr].mean(-1)) == best_clean for dr in draws]) if reps else np.nan
            rows.append({"dataset": T.dataset, "degradation": g, "level": k, "best_clean_model": T.models[best_clean],
                         "best_degraded_model": T.models[int(np.argmax(d[:, k]))],
                         "regret": float(d[:, k].max() - d[best_clean, k]), "rank_flip_share": float(flips),
                         "clean_still_best": bool(np.argmax(d[:, k]) == best_clean), "p_clean_best_stays_best": float(stay)})
    return pd.DataFrame(rows)


def pairwise_models(scores):
    blocks = scores.pivot_table(index=["dataset", "degradation"], columns="model", values="auc_loss").dropna()
    models = list(blocks.columns)
    out = {"n_blocks": len(blocks), "mean_rank": dict(zip(models, rankdata(blocks.to_numpy(), axis=1).mean(0)))}
    if len(models) >= 3 and len(blocks) >= 3:
        stat, p = friedmanchisquare(*[blocks[m].to_numpy() for m in models])
        out["friedman"] = {"chi2": float(stat), "p": float(p)}
    rows = []
    for a, b in itertools.combinations(models, 2):
        diff = blocks[a].to_numpy() - blocks[b].to_numpy()
        p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
        rows.append({"model_a": a, "model_b": b, "median_diff_auc_loss": float(np.median(diff)), "p": float(p)})
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = holm(df["p"].to_numpy())
    return out, df


def metric_sensitivity(T, ext, cfg, rng):
    rows = []
    for metric in ("dice", "iou", "nsd", "hd95"):
        ctx = DatasetContext(T, ext, min(5, int(cfg.analysis.halves)), rng, metric)
        pairs = h1_point(ctx)
        if pairs:
            rows.append({"dataset": T.dataset, "metric": metric,
                         "h1_beta": float(np.mean([cell_corr(*p).mean() for p in pairs]))})
    retention = []
    for g in T.degs:
        idx = np.nonzero(T.complete(g))[0]
        if len(idx) == 0:
            continue
        near = [i + 1 for i, s in enumerate(T.levels[: T.K[g]]) if abs(s - 0.75) < 1e-9]
        for k in sorted(set(near + [T.K[g]])):
            c = T.cond_ids(g)[k - 1]
            for m, model in enumerate(T.models):
                r = {"dataset": T.dataset, "degradation": g, "level": k, "target_ssim": T.levels[k - 1], "model": model}
                for metric in ("dice", "nsd", "hd95"):
                    t = T.tensor(metric)[m][:, :, idx]
                    r[f"{metric}_ratio"] = float(np.nanmean(t[:, c]) / max(float(np.nanmean(t[:, 0])), 1e-9))
                retention.append(r)
    return pd.DataFrame(rows), pd.DataFrame(retention)


def kendall_w_blocks(T):
    ws = []
    for g in T.degs:
        idx = np.nonzero(T.complete(g))[0]
        if len(idx) == 0:
            continue
        t = T.tensor("dice")[:, :, [0] + T.cond_ids(g)][..., idx]
        ranks = rankdata(np.nanmean(t, axis=-1).mean(-1), axis=0)
        m, n = ranks.shape[1], ranks.shape[0]
        s = ((ranks.sum(1) - ranks.sum(1).mean()) ** 2).sum()
        ws.append(12 * s / (m ** 2 * (n ** 3 - n)))
    return ws


def kendall_w_summary(tensors, rng, reps=1000):
    ws = np.array([w for T in tensors.values() for w in kendall_w_blocks(T)])
    if len(ws) == 0:
        return {}
    boots = [ws[rng.integers(0, len(ws), len(ws))].mean() for _ in range(reps)]
    lo, hi = percentile_ci(boots)
    return {"mean_w": float(ws.mean()), "ci_low": lo, "ci_high": hi, "n_blocks": int(len(ws))}


def variance_shares(T):
    ids = np.nonzero(np.all([T.complete(g) for g in T.degs], axis=0))[0]
    if len(ids) < 3 or not T.degs:
        return {}
    loss = np.zeros((len(T.models), len(T.seeds), len(T.degs), len(ids)))
    for gi, g in enumerate(T.degs):
        t = T.tensor("dice")[:, :, [0] + T.cond_ids(g)][..., ids]
        loss[:, :, gi] = (t[:, :, :1] - t[:, :, 1:]).mean(2)
    x = loss
    gm = x.mean()
    m_, s_, g_, n_ = x.shape
    mean_m, mean_g, mean_n = x.mean((1, 2, 3)), x.mean((0, 1, 3)), x.mean((0, 1, 2))
    mean_ms, mean_nm, mean_ng, mean_mg = x.mean((2, 3)), x.mean((1, 2)), x.mean((0, 1)), x.mean((1, 3))
    ss = {"image": m_ * s_ * g_ * ((mean_n - gm) ** 2).sum(), "model": s_ * g_ * n_ * ((mean_m - gm) ** 2).sum(),
          "degradation": m_ * s_ * n_ * ((mean_g - gm) ** 2).sum(),
          "seed_within_model": g_ * n_ * ((mean_ms - mean_m[:, None]) ** 2).sum(),
          "image_x_model": s_ * g_ * ((mean_nm - mean_n[None] - mean_m[:, None] + gm) ** 2).sum(),
          "image_x_degradation": m_ * s_ * ((mean_ng - mean_n[None] - mean_g[:, None] + gm) ** 2).sum(),
          "model_x_degradation": s_ * n_ * ((mean_mg - mean_m[:, None] - mean_g[None] + gm) ** 2).sum()}
    total = ((x - gm) ** 2).sum()
    ss["residual"] = total - sum(ss.values())
    return {k: float(v / total) for k, v in ss.items()}


def one_way_reliability(values):
    m, s = values.shape
    if s < 2 or m < 2:
        return np.nan
    msb = s * values.mean(1).var(ddof=1)
    msw = ((values - values.mean(1, keepdims=True)) ** 2).sum() / (m * (s - 1))
    sigma_m = max((msb - msw) / s, 0.0)
    return float(sigma_m / (sigma_m + msw / s)) if (sigma_m + msw) > 0 else np.nan


def reliability(T, ext, threshold=0.7):
    d = T.tensor("dice")
    clean = np.nanmean(d[:, :, 0], axis=-1)
    row = {"dataset": T.dataset, "rel_clean": one_way_reliability(clean)}
    auc = []
    for g in T.degs:
        idx = np.nonzero(T.complete(g))[0]
        if len(idx):
            auc.append(np.nanmean(d[:, :, T.cond_ids(g)][..., idx], axis=(2, 3)))
    if auc:
        row["rel_degraded"] = one_way_reliability(np.mean(auc, axis=0))
    if ext is not None:
        row["rel_external"] = one_way_reliability(np.nanmean(ext.dice, axis=-1))
    row["clean_dice_spread_models"] = float(np.nanstd(clean.mean(1)))
    row["clean_dice_seed_sd"] = float(np.nanmean(clean.std(1, ddof=1))) if clean.shape[1] > 1 else np.nan
    rels = [v for k, v in row.items() if k.startswith("rel_") and np.isfinite(v)]
    row["low_reliability"] = bool(rels and min(rels) < threshold)
    return row


def mce_agreement(scores):
    rows = []
    for (d, g), grp in scores.groupby(["dataset", "degradation"]):
        if len(grp) > 2:
            rho = spearmanr(grp["auc_loss"], grp["rel_mce"])[0]
            rows.append({"dataset": d, "degradation": g, "spearman_auc_loss_vs_rel_mce": float(rho)})
    return pd.DataFrame(rows)


def subgroup_table(T):
    if T.subgroup is None:
        return pd.DataFrame()
    rows = []
    for sg in np.unique(T.subgroup):
        sel = T.subgroup == sg
        for m, model in enumerate(T.models):
            row = {"dataset": T.dataset, "subgroup": sg, "model": model, "n_images": int(sel.sum()),
                   "clean_dice": float(np.nanmean(T.tensor("dice")[m, :, 0][:, sel]))}
            drops = []
            for g in T.degs:
                idx = np.nonzero(T.complete(g) & sel)[0]
                if len(idx):
                    curve = np.nanmean(T.tensor("dice")[m][:, [0] + T.cond_ids(g)][..., idx], axis=(0, 2))
                    drops.append(curve[0] - area(curve))
            row["auc_loss_mean"] = float(np.mean(drops)) if drops else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def score_summary(scores):
    cols = ["d0", "auc_abs", "auc_loss", "auc_er", "rel_mce"]
    g = scores.assign(group=np.where(scores["degradation"].str.startswith("G"), "generic", "specific"))
    return g.groupby(["dataset", "group", "model", "family"])[cols].mean().reset_index()


__all__ = ["curve_scores"]
