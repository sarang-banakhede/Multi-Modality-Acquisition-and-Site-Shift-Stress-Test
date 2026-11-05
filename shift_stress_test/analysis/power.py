import numpy as np
from scipy.stats import rankdata

from .meta import fisher


def _sim_stat(rng, n_sim, m, rho, rel_x, rel_y):
    k = len(rel_x)
    u = rng.standard_normal((n_sim, k, m))
    v = rng.standard_normal((n_sim, k, m))
    rx, ry = np.asarray(rel_x).reshape(1, k, 1), np.asarray(rel_y).reshape(1, k, 1)
    x = np.sqrt(rx) * u + np.sqrt(1 - rx) * rng.standard_normal(u.shape)
    y = np.sqrt(ry) * (rho * u + np.sqrt(1 - rho ** 2) * v) + np.sqrt(1 - ry) * rng.standard_normal(u.shape)
    a, b = rankdata(x, axis=-1), rankdata(y, axis=-1)
    a, b = a - a.mean(-1, keepdims=True), b - b.mean(-1, keepdims=True)
    return fisher((a * b).sum(-1) / np.sqrt((a ** 2).sum(-1) * (b ** 2).sum(-1))).mean(-1)


def power_check(designs, n_models, rng, n_sim=2000, alpha=0.05, n_tests=5):
    out = {}
    for name, (rel_x, rel_y) in designs.items():
        rx = np.array([r if np.isfinite(r) else 0.8 for r in rel_x], float)
        ry = np.array([r if np.isfinite(r) else 0.8 for r in rel_y], float)
        null = _sim_stat(rng, n_sim, n_models, 0.0, rx, ry)
        thr = float(np.percentile(np.abs(null), 100 * (1 - alpha / n_tests)))
        table = [{"true_rho": round(float(r), 2), "power": float((np.abs(_sim_stat(rng, n_sim, n_models, float(r), rx, ry)) > thr).mean())}
                 for r in np.arange(0.1, 1.0, 0.1)]
        out[name] = {"reliability_x": rx.tolist(), "reliability_y": ry.tolist(), "threshold_z": thr, "table": table,
                     "minimal_detectable_rho_80pct": next((t["true_rho"] for t in table if t["power"] >= 0.8), None)}
    return out
