import numpy as np
import pandas as pd

from ..utils.io import load_json


def clean_tables(layout, models, datasets, seeds):
    from .data import load_eval

    frames = []
    cols = ["dice", "iou", "precision", "recall", "specificity", "hd95", "assd", "nsd", "area_error", "volume_similarity",
            "detected", "empty_pred"]
    for d in datasets:
        for subset, ckpt in (("test", "best"), ("test", "last"), ("external", "best"), ("external", "last")):
            try:
                df = load_eval(layout, models, d, seeds, subset, ckpt)
            except ValueError:
                continue
            df = df[df["degradation"] == "clean"]
            g = df.groupby(["dataset", "model", "family", "seed"])[cols].mean().reset_index()
            g.insert(3, "ckpt", ckpt)
            g.insert(4, "subset", subset)
            frames.append(g)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    rows = []
    for d in datasets:
        for m in models:
            for s in seeds:
                p = layout.run_dir(m, d, s) / "summary.json"
                if p.exists():
                    sm = load_json(p)
                    rows.append({"dataset": d, "model": m, "seed": s, "best_iteration": sm["best_iteration"],
                                 "best_val_dice": sm["best_val_dice"], "train_time_h": sm["train_time_h"],
                                 "params": sm["params"]})
    return out, pd.DataFrame(rows)


def curve_table(T):
    rows = []
    ssim = T.ssim()
    for g in T.degs:
        idx = np.nonzero(T.complete(g))[0]
        if len(idx) == 0:
            continue
        for k in range(0, T.K[g] + 1):
            c = 0 if k == 0 else T.cidx[(g, k)]
            for m, model in enumerate(T.models):
                row = {"dataset": T.dataset, "degradation": g, "level": k, "model": model, "n_images": len(idx),
                       "achieved_ssim": 1.0 if k == 0 else float(np.nanmedian(ssim[c][idx]))}
                for metric in ("dice", "iou", "nsd", "hd95", "precision", "recall"):
                    row[metric] = float(np.nanmean(T.tensor(metric)[m, :, c][:, idx]))
                rows.append(row)
    return pd.DataFrame(rows)


def model_level_table(T, X):
    from .scores import area, curves, er_curves

    d = T.tensor("dice")
    parts = {"generic": [], "specific": []}
    for g in T.degs:
        idx = np.nonzero(T.complete(g))[0]
        if len(idx) == 0:
            continue
        c = curves(d, [0] + T.cond_ids(g), idx)
        parts["generic" if g in T.generic else "specific"].append((area(c), area(er_curves(c)), c[:, 0] - area(c)))
    clean = np.nanmean(d[:, :, 0], axis=(1, 2))
    ext = np.nanmean(X.dice, axis=(1, 2))
    er_ext = er_curves(np.stack([clean, ext], axis=1))[:, 1]
    rows = []
    for m, model in enumerate(T.models):
        row = {"dataset": T.dataset, "model": model, "family": T.family[model], "clean_dice": float(clean[m]),
               "external_dice": float(ext[m]), "er_ext": float(er_ext[m])}
        for name, items in parts.items():
            if items:
                row[f"auc_abs_{name}"] = float(np.mean([i[0][m] for i in items]))
                row[f"auc_er_{name}"] = float(np.mean([i[1][m] for i in items]))
                row[f"auc_loss_{name}"] = float(np.mean([i[2][m] for i in items]))
        allv = [i for v in parts.values() for i in v]
        row["auc_er_all"] = float(np.mean([i[1][m] for i in allv])) if allv else np.nan
        row["auc_abs_all"] = float(np.mean([i[0][m] for i in allv])) if allv else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
