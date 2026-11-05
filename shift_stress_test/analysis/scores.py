import numpy as np
import pandas as pd

from .data import probit

LINKS = {"probit": probit, "linear": lambda x: np.asarray(x, dtype=float)}


def area(x):
    k = x.shape[-1]
    w = np.ones(k)
    w[0] = w[-1] = 0.5
    return (x * w).sum(-1) / (k - 1)


def er_curves(d, link=probit):
    p = link(d)
    x, y = p[..., :1], p[..., 1:]
    n = d.shape[-2] - 1
    sx = x.sum(-2, keepdims=True) - x
    sy = y.sum(-2, keepdims=True) - y
    sxx = (x * x).sum(-2, keepdims=True) - x * x
    sxy = (x * y).sum(-2, keepdims=True) - x * y
    b = (n * sxy - sx * sy) / (n * sxx - sx ** 2 + 1e-9)
    a = (sy - b * sx) / n
    return np.concatenate([np.zeros_like(x), y - (a + b * x)], axis=-1)


def curves(tensor, cond_ids, img_idx, seed_w=None):
    sel = np.take(tensor[:, :, cond_ids], img_idx, axis=-1)
    per_seed = np.nanmean(sel, axis=-1)
    if seed_w is None:
        return per_seed.mean(axis=1)
    return (per_seed * seed_w[:, :, None]).sum(axis=1)


def curve_scores(d, mild, link=probit):
    au = area(d)
    er = er_curves(d, link)
    out = {"d0": d[..., 0], "auc_abs": au, "auc_loss": d[..., 0] - au, "auc_rel": au / np.maximum(d[..., 0], 1e-6),
           "auc_er": area(er)}
    if mild and d.shape[-1] > mild + 1:
        dm = d[..., :mild + 1]
        am = area(dm)
        out.update({"mild_auc_abs": am, "mild_auc_loss": d[..., 0] - am, "mild_auc_er": area(er[..., :mild + 1])})
    return out


def relative_mce(d, baseline):
    e = 1.0 - d
    loss = (e - e[..., :1])[..., 1:].sum(-1)
    return loss / np.maximum(loss[baseline], 1e-9)


def score_table(T, mild):
    rows = []
    for g in T.degs:
        idx = np.nonzero(T.complete(g))[0]
        if len(idx) == 0:
            continue
        d = curves(T.tensor("dice"), [0] + T.cond_ids(g), idx)
        sc = curve_scores(d, mild)
        baseline = T.models.index("unet") if "unet" in T.models else 0
        mce = relative_mce(d, baseline)
        for m, model in enumerate(T.models):
            rows.append({"dataset": T.dataset, "degradation": g, "model": model, "family": T.family[model],
                         "n_complete": len(idx), "coverage": len(idx) / len(T.names), "rel_mce": mce[m],
                         **{k: float(v[m]) for k, v in sc.items()}})
    return pd.DataFrame(rows)
