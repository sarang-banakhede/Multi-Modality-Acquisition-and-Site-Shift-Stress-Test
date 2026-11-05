import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .crossfit import Clusters
from .data import load_eval
from .meta import percentile_ci


def mean_limit(ret, ssim, theta):
    thr = 1.0 - theta
    bad = np.nonzero(ret[1:] < thr)[0]
    if len(bad) == 0:
        return np.nan
    j = bad[0] + 1
    r0, r1, s0, s1 = ret[j - 1], ret[j], ssim[j - 1], ssim[j]
    return float(s0 + (r0 - thr) / (r0 - r1) * (s1 - s0))


def pass_limit(share, ssim, q):
    bad = np.nonzero(share[1:] < q)[0]
    if len(bad) == 0:
        return np.nan
    j = bad[0] + 1
    p0, p1, s0, s1 = share[j - 1], share[j], ssim[j - 1], ssim[j]
    return float(s0 + (p0 - q) / (p0 - p1) * (s1 - s0))


def _curve_stats(dm, d_runs, n_idx, deltas):
    ret = dm[n_idx].mean(0)
    ret = ret / max(ret[0], 1e-9)
    shares = [(np.abs(d_runs[:, :, n_idx] - d_runs[:, :1, n_idx]) <= dl).mean(-1).mean(0) for dl in deltas]
    return ret, shares


def envelope_table(T, cfg, rng):
    a = cfg.analysis
    reps = min(int(a.bootstrap_reps), 500)
    ssim_all = T.ssim()
    rows = []
    for g in T.degs:
        idx = np.nonzero(T.complete(g))[0]
        if len(idx) < 2:
            continue
        ids = [0] + T.cond_ids(g)
        s_axis = np.concatenate([[1.0], np.nanmedian(ssim_all[T.cond_ids(g)][:, idx], axis=1)])
        cl = Clusters(idx, T.groups)
        draws = [cl.draw(rng) for _ in range(reps)]
        for metric in ("dice", "nsd"):
            t = T.tensor(metric)[:, :, ids]
            for m, model in enumerate(T.models):
                runs = t[m][:, :, :]
                dm = np.nanmean(runs, axis=0).T
                specs = []
                for theta in a.loss_tolerances[metric]:
                    specs.append(("mean_loss", theta, lambda ret, sh, th=theta: mean_limit(ret, s_axis, th)))
                if metric == "dice":
                    for k, dl in enumerate(a.pass_deltas):
                        specs.append(("pass_share", dl, lambda ret, sh, k=k: pass_limit(sh[k], s_axis, a.primary_pass_share)))
                full_ret, full_sh = _curve_stats(dm, runs, idx, list(a.pass_deltas) if metric == "dice" else [])
                boots = [_curve_stats(dm, runs, d, list(a.pass_deltas) if metric == "dice" else []) for d in draws]
                for kind, tol, fn in specs:
                    val = fn(full_ret, full_sh)
                    bs = np.array([fn(r, sh) for r, sh in boots], float)
                    lo, hi = percentile_ci(bs) if np.isfinite(bs).any() else (np.nan, np.nan)
                    rows.append({"dataset": T.dataset, "degradation": g, "model": model, "family": T.family[model],
                                 "metric": metric, "kind": kind, "tolerance": tol, "limit_ssim": val, "ci_low": lo,
                                 "ci_high": hi, "reached": bool(np.isfinite(val))})
    return pd.DataFrame(rows)


def cvar(x, frac=0.1):
    k = max(1, int(np.ceil(len(x) * frac)))
    return float(np.sort(x)[-k:].mean())


def icc21(a, b):
    x = np.stack([a, b], axis=1)
    n, k = x.shape
    gm = x.mean()
    ms_r = k * ((x.mean(1) - gm) ** 2).sum() / (n - 1)
    ms_c = n * ((x.mean(0) - gm) ** 2).sum() / (k - 1)
    ms_e = ((x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return float((ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n))


def tail_table(T):
    rows = []
    dice, area_p = T.tensor("dice"), T.tensor("pred_area")
    deltas = {k: T.tensor(k) for k in ("fp_area", "fn_area", "n_components", "n_spurious")}
    for g in T.degs:
        idx = np.nonzero(T.complete(g))[0]
        if len(idx) < 3:
            continue
        for k, c in enumerate(T.cond_ids(g), start=1):
            for m, model in enumerate(T.models):
                per_seed = []
                for s in range(len(T.seeds)):
                    d0, dk = dice[m, s, 0, idx], dice[m, s, c, idx]
                    drop = d0 - dk
                    a0, ak = area_p[m, s, 0, idx], area_p[m, s, c, idx]
                    both = (a0 > 0) & (ak > 0)
                    row = {"mean_drop": drop.mean(), "p90_drop": np.percentile(drop, 90), "cvar10_drop": cvar(drop),
                           "share_drop_gt_005": (drop > 0.05).mean(), "share_drop_gt_010": (drop > 0.10).mean(),
                           "p10_degraded_dice": np.percentile(dk, 10),
                           "share_area_change_gt_10pct": (np.abs(ak - a0) / np.maximum(a0, 1) > 0.10).mean()}
                    if both.sum() > 3:
                        la0, lak = np.log(a0[both]), np.log(ak[both])
                        diff = lak - la0
                        row.update({"icc_log_area": icc21(la0, lak), "ba_bias_log_area": diff.mean(),
                                    "ba_loa_log_area": 1.96 * diff.std(ddof=1)})
                    for name, arr in deltas.items():
                        row[f"delta_{name}"] = float((arr[m, s, c, idx] - arr[m, s, 0, idx]).mean())
                    per_seed.append(row)
                avg = pd.DataFrame(per_seed).mean(numeric_only=True).to_dict()
                rows.append({"dataset": T.dataset, "degradation": g, "level": k, "model": model, "family": T.family[model],
                             "n_images": len(idx), **avg})
    return pd.DataFrame(rows)


def envelope_gap(env):
    mean = env[env["kind"] == "mean_loss"][["dataset", "degradation", "model", "metric", "tolerance", "limit_ssim"]]
    mean = mean[mean["metric"] == "dice"].rename(columns={"tolerance": "theta", "limit_ssim": "mean_limit"})
    ps = env[env["kind"] == "pass_share"][["dataset", "degradation", "model", "tolerance", "limit_ssim"]]
    ps = ps.rename(columns={"tolerance": "delta", "limit_ssim": "individual_limit"})
    merged = ps.merge(mean, on=["dataset", "degradation", "model"])
    merged = merged[((merged["delta"] == 0.02) & (merged["theta"] == 0.02)) | ((merged["delta"] == 0.05) & (merged["theta"] == 0.05))]
    merged["gap"] = merged["individual_limit"] - merged["mean_limit"]
    return merged


def confident_failure(layout, cfg, T, models, seeds, rng):
    val = load_eval(layout, models, T.dataset, seeds, "val")
    val = val[val["degradation"] == "clean"]
    mild = cfg.analysis.mild_levels
    reps = min(int(cfg.analysis.bootstrap_reps), 500)
    cl = Clusters(np.arange(len(T.names)), T.groups)
    draws = [np.bincount(cl.draw(rng), minlength=len(T.names)).astype(float) for _ in range(reps)]
    rows = []
    for m, model in enumerate(T.models):
        parts = {flag: {"den": np.zeros(len(T.names)), "fail": np.zeros(len(T.names)), "conf": np.zeros(len(T.names)),
                        "ent": [], "msp": [], "failed": []} for flag in (True, False)}
        for s, seed in enumerate(T.seeds):
            v = val[(val["model"] == model) & (val["seed"] == seed)]
            if v.empty:
                continue
            tau = float(np.percentile(v["entropy"], 95))
            ent, msp, d = T.tensor("entropy")[m, s], T.tensor("msp")[m, s], T.tensor("dice")[m, s]
            miss = T.tensor("detected")[m, s] == 0
            eligible = d[0] >= 0.7
            for g in T.degs:
                for k, c in enumerate(T.cond_ids(g), start=1):
                    ok = eligible & ~np.isnan(d[c])
                    fail = ok & ((d[c] < 0.5) | miss[c])
                    conf = fail & (ent[c] <= tau)
                    for flag in ((True, False) if k <= mild else (False,)):
                        p = parts[flag]
                        p["den"] += ok
                        p["fail"] += fail
                        p["conf"] += conf
                        p["ent"].append(ent[c][ok])
                        p["msp"].append(msp[c][ok])
                        p["failed"].append(fail[ok])
        for flag, p in parts.items():
            if p["den"].sum() == 0:
                continue
            rate = lambda num: float(num.sum() / p["den"].sum())
            boot_f = np.array([(w * p["fail"]).sum() / max((w * p["den"]).sum(), 1) for w in draws])
            boot_c = np.array([(w * p["conf"]).sum() / max((w * p["den"]).sum(), 1) for w in draws])
            ent, msp, failed = (np.concatenate(p[k]) for k in ("ent", "msp", "failed"))
            row = {"dataset": T.dataset, "model": model, "family": T.family[model], "mild": flag,
                   "n_eligible": int(p["den"].sum()), "failure_rate": rate(p["fail"]), "confident_failure_rate": rate(p["conf"]),
                   "confident_share_of_failures": float(p["conf"].sum() / p["fail"].sum()) if p["fail"].sum() else np.nan}
            if reps:
                row["failure_ci"] = percentile_ci(boot_f)
                row["confident_ci"] = percentile_ci(boot_c)
            if 0 < failed.sum() < len(failed):
                row["auroc_entropy"] = float(roc_auc_score(failed, ent))
                row["auroc_low_confidence"] = float(roc_auc_score(failed, -msp))
            rows.append(row)
    return pd.DataFrame(rows)
