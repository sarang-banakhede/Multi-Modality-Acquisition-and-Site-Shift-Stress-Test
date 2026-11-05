import numpy as np
import pandas as pd

from ..config import Config
from .confirmatory import run_h1, run_h2, run_h4
from .crossfit import DatasetContext
from .external import beta_ext

VARIANTS = {"base": {}, "linear_scale": {"link": "linear"}, "mild_only": {"mild": True}, "no_cross_fitting": {"crossfit": False}}


def _row(variant, test, res, key, extra=None):
    if not res:
        return None
    return {"variant": variant, "test": test, "estimate": res.get(key), "ci_low": res.get("ci_low"), "ci_high": res.get("ci_high"),
            "p": res.get("p_permutation"), **(extra or {})}


def run_variants(T, X, cfg, rng, datasets):
    a = cfg.analysis
    sub = cfg.to_dict()
    sub["analysis"] = {**sub["analysis"], "permutation_reps": a.sensitivity_permutations, "bootstrap_reps": a.sensitivity_bootstrap}
    scfg = Config(sub)
    rows = []
    for name, opts in VARIANTS.items():
        ctxs = {d: DatasetContext(T[d], X[d], int(a.halves), rng, "dice", opts.get("link", "probit"),
                                  int(a.mild_levels) if opts.get("mild") else None, opts.get("crossfit", True)) for d in datasets}
        for test, fn, key in (("H1", run_h1, "beta"), ("H2", run_h2, "rho_mean"), ("H4", run_h4, "rho_mean"), ("beta_ext", beta_ext, "beta_ext")):
            res = fn(ctxs, scfg, rng)
            res = res[0] if isinstance(res, tuple) else res
            r = _row(name, test, res, key)
            if r:
                rows.append(r)
    return pd.DataFrame(rows)


def disattenuated(base_results, reliability):
    rel = reliability.set_index("dataset")
    rows = []
    for test, key, rx, ry in (("H1", "per_dataset", "rel_clean", "rel_degraded"), ("H2", "per_dataset", "rel_degraded", "rel_degraded"),
                              ("H4", "per_dataset", "rel_degraded", "rel_external"), ("beta_ext", "per_dataset", "rel_clean", "rel_external")):
        res = base_results.get(test)
        if not res:
            continue
        corrected = []
        for d, r in res[key].items():
            den = np.sqrt(max(rel.loc[d, rx], 1e-3) * max(rel.loc[d, ry], 1e-3))
            corrected.append(float(np.clip(r / den, -1, 1)))
        rows.append({"test": test, "observed_mean": float(np.mean(list(res[key].values()))), "disattenuated_mean": float(np.mean(corrected))})
    return pd.DataFrame(rows)


def per_source_h4(T, X, cfg, rng, datasets):
    rows = []
    for d in datasets:
        for src in np.unique(X[d].sources):
            if (X[d].sources == src).sum() < 3:
                continue
            ctx = DatasetContext(T[d], X[d], int(cfg.analysis.halves), rng, source=src)
            res = run_h4({d: ctx}, cfg, rng, with_tests=False)
            if res:
                rows.append({"dataset": d, "source": src, "n_images": int((X[d].sources == src).sum()), "rho": res["per_dataset"][d]})
    return pd.DataFrame(rows)
