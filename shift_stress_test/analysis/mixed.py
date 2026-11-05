import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.regression.mixed_linear_model import MixedLM

from .data import probit

TERMS = {"model": "0 + C(model)", "dataset": "0 + C(dataset)", "degradation": "0 + C(degradation)",
         "model_dataset": "0 + C(model_dataset)", "model_degradation": "0 + C(model_degradation)",
         "dataset_degradation": "0 + C(dataset_degradation)", "run": "0 + C(run)"}
ARCHITECTURE = ["model", "model_dataset", "model_degradation"]


def h3_frame(tensors):
    rows = []
    for T in tensors:
        d = T.tensor("dice")
        for g in T.generic:
            idx = np.nonzero(T.complete(g))[0]
            if len(idx) == 0:
                continue
            base = probit(np.nanmean(d[:, :, 0][..., idx], axis=-1))
            for k, c in enumerate(T.cond_ids(g), start=1):
                cur = probit(np.nanmean(d[:, :, c][..., idx], axis=-1))
                for m, model in enumerate(T.models):
                    for s, seed in enumerate(T.seeds):
                        rows.append({"delta": cur[m, s] - base[m, s], "knot": k, "family": T.family[model], "model": model,
                                     "dataset": T.dataset, "degradation": g, "model_dataset": f"{model}|{T.dataset}",
                                     "model_degradation": f"{model}|{g}", "dataset_degradation": f"{T.dataset}|{g}",
                                     "run": f"{model}|{T.dataset}|{seed}"})
    df = pd.DataFrame(rows)
    df["all"] = 1
    return df


def _fit(df, terms, reml):
    vc = {t: TERMS[t] for t in terms}
    fixed = "delta ~ knot + C(family)" if df["family"].nunique() > 1 else "delta ~ knot"
    model = MixedLM.from_formula(fixed, groups="all", vc_formula=vc, re_formula="0", data=df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(reml=reml, method="lbfgs", maxiter=200)


def _shares(res, terms):
    var = dict(zip(res.model.exog_vc.names, res.vcomp))
    var["residual"] = float(res.scale)
    total = sum(var.values())
    return {k: float(v / total) for k, v in var.items()}


def _simulate(df, res, terms, rng):
    y = np.asarray(res.model.exog @ res.fe_params[: res.model.exog.shape[1]])
    var = dict(zip(res.model.exog_vc.names, res.vcomp))
    for t in terms:
        codes, uniques = pd.factorize(df[t])
        y = y + rng.normal(0, np.sqrt(max(var.get(t, 0.0), 0.0)), len(uniques))[codes]
    return y + rng.normal(0, np.sqrt(res.scale), len(df))


def run_h3(df, reps, rng):
    full_terms = list(TERMS)
    reduced_terms = [t for t in full_terms if t not in ARCHITECTURE]
    full = _fit(df, full_terms, reml=False)
    red = _fit(df, reduced_terms, reml=False)
    stat = max(0.0, 2 * (full.llf - red.llf))
    out = {"n_rows": len(df), "lrt_stat": float(stat), "shares_ml": _shares(full, full_terms)}
    out["architecture_share"] = float(sum(out["shares_ml"].get(t, 0.0) for t in ARCHITECTURE))
    out["p_asymptotic"] = float(0.5 * chi2.sf(stat, 3) + 0.5 * chi2.sf(stat, 2))
    if reps > 0:
        boot, shares = [], []
        for _ in range(reps):
            sim = df.copy()
            sim["delta"] = _simulate(df, red, reduced_terms, rng)
            try:
                f, r = _fit(sim, full_terms, False), _fit(sim, reduced_terms, False)
                boot.append(max(0.0, 2 * (f.llf - r.llf)))
            except Exception:
                continue
        out["p_bootstrap"] = float((1 + np.sum(np.array(boot) >= stat)) / (len(boot) + 1)) if boot else np.nan
        out["bootstrap_reps_ok"] = len(boot)
        for _ in range(min(reps, 50)):
            sim = df.copy()
            sim["delta"] = _simulate(df, full, full_terms, rng)
            try:
                shares.append(_shares(_fit(sim, full_terms, False), full_terms))
            except Exception:
                continue
        if shares:
            keys = shares[0].keys()
            out["shares_ci"] = {k: [float(np.percentile([s[k] for s in shares], 2.5)),
                                    float(np.percentile([s[k] for s in shares], 97.5))] for k in keys}
    return out
