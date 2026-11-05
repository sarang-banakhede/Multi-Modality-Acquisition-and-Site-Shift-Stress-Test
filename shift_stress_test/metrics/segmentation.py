import math
import multiprocessing as mp

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure, label

STRUCT = generate_binary_structure(2, 1)

METRIC_COLUMNS = ["dice", "iou", "precision", "recall", "specificity", "hd95", "assd", "nsd", "area_error",
                  "volume_similarity", "gt_area", "pred_area", "fp_area", "fn_area", "n_components", "n_spurious",
                  "detected", "empty_pred"]


def _surface(mask):
    return mask & ~binary_erosion(mask, structure=STRUCT, border_value=0)


def image_metrics(pred, gt, nsd_tol=2.0):
    p, g = pred.astype(bool), gt.astype(bool)
    n = p.size
    diag = math.sqrt(2.0) * p.shape[0]
    ps, gs = int(p.sum()), int(g.sum())
    inter = int((p & g).sum())
    fp, fn = ps - inter, gs - inter
    specificity = (n - ps - fn) / max(n - gs, 1)
    n_comp = int(label(p, structure=STRUCT)[1]) if ps else 0
    if ps == 0 and gs == 0:
        return (1.0, 1.0, 1.0, 1.0, specificity, 0.0, 0.0, 1.0, 0.0, 1.0, gs, ps, fp, fn, 0, 0, 1.0, 0.0)
    if ps == 0 or gs == 0:
        area_err = float("nan") if gs == 0 else 1.0
        return (0.0, 0.0, 0.0, 0.0, specificity, diag, diag, 0.0, area_err, 0.0, gs, ps, fp, fn,
                n_comp, n_comp, 0.0, float(ps == 0))
    dice = 2.0 * inter / (ps + gs)
    iou = inter / (ps + gs - inter)
    sp, sg = _surface(p), _surface(g)
    d_pg = distance_transform_edt(~sg)[sp]
    d_gp = distance_transform_edt(~sp)[sg]
    hd95 = float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))
    assd = float((d_pg.sum() + d_gp.sum()) / (d_pg.size + d_gp.size))
    nsd = float(((d_pg <= nsd_tol).sum() + (d_gp <= nsd_tol).sum()) / (d_pg.size + d_gp.size))
    labels, k = label(p, structure=STRUCT)
    overlap = np.bincount(labels[g & p], minlength=k + 1)[1:] if k else np.zeros(0)
    spurious = int((overlap == 0).sum())
    return (dice, iou, inter / ps, inter / gs, specificity, hd95, assd, nsd, abs(ps - gs) / gs,
            1.0 - abs(ps - gs) / (ps + gs), gs, ps, fp, fn, n_comp, spurious, float(iou > 0.1), 0.0)


def _work(args):
    return image_metrics(*args)


class MetricPool:
    def __init__(self, workers):
        self.pool = mp.get_context("fork").Pool(workers) if workers > 1 else None

    def map(self, preds, gts, nsd_tol=2.0):
        jobs = [(p, g, nsd_tol) for p, g in zip(preds, gts)]
        if self.pool is None:
            return [_work(j) for j in jobs]
        return self.pool.map(_work, jobs, chunksize=max(1, len(jobs) // (4 * self.pool._processes)))

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
