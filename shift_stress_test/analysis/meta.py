import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm, rankdata


def fisher(r):
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def reml_meta(z, se):
    z, se = np.asarray(z, float), np.asarray(se, float)
    k = len(z)
    if k < 2:
        return {"mu": float(z[0]), "se": float(se[0]), "tau2": 0.0, "i2": 0.0, "k": k}
    v = se ** 2

    def nll(t2):
        w = 1.0 / (v + t2)
        mu = (w * z).sum() / w.sum()
        return 0.5 * (np.log(v + t2).sum() + np.log(w.sum()) + (w * (z - mu) ** 2).sum())

    tau2 = float(minimize_scalar(nll, bounds=(0.0, 10.0), method="bounded").x)
    w = 1.0 / (v + tau2)
    mu = float((w * z).sum() / w.sum())
    w0 = 1.0 / v
    q = float((w0 * (z - (w0 * z).sum() / w0.sum()) ** 2).sum())
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    return {"mu": mu, "se": float(np.sqrt(1.0 / w.sum())), "tau2": tau2, "i2": i2, "k": k}


def pooled_summary(z, se):
    m = reml_meta(z, se)
    half = 1.96 * m["se"]
    p = 2 * (1 - norm.cdf(abs(m["mu"]) / m["se"])) if m["se"] > 0 else np.nan
    return {"rho": float(np.tanh(m["mu"])), "ci_low": float(np.tanh(m["mu"] - half)), "ci_high": float(np.tanh(m["mu"] + half)),
            "z": m["mu"], "se_z": m["se"], "tau2": m["tau2"], "i2": m["i2"], "k": m["k"], "p_wald": float(p)}


def spearman_rows(a, b):
    ra, rb = rankdata(a, axis=-1), rankdata(b, axis=-1)
    ra, rb = ra - ra.mean(-1, keepdims=True), rb - rb.mean(-1, keepdims=True)
    return (ra * rb).sum(-1) / np.sqrt((ra ** 2).sum(-1) * (rb ** 2).sum(-1) + 1e-12)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    out = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def benjamini_hochberg(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    return out


def perm_pvalue(obs, perm):
    return float((1 + np.sum(np.abs(perm) >= abs(obs))) / (len(perm) + 1))


def percentile_ci(x, level=0.95):
    a = (1 - level) / 2
    return float(np.nanpercentile(x, 100 * a)), float(np.nanpercentile(x, 100 * (1 - a)))
