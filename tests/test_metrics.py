import math

import numpy as np

from shift_stress_test.metrics import METRIC_COLUMNS, image_metrics

DIAG = math.sqrt(2.0) * 224


def blob(x0, x1, y0, y1):
    m = np.zeros((224, 224), np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def row(pred, gt):
    return dict(zip(METRIC_COLUMNS, image_metrics(pred, gt)))


def test_identical_masks():
    g = blob(50, 150, 60, 160)
    r = row(g, g)
    assert r["dice"] == 1 and r["iou"] == 1 and r["hd95"] == 0 and r["nsd"] == 1 and r["assd"] == 0


def test_both_empty():
    z = np.zeros((224, 224), np.uint8)
    r = row(z, z)
    assert r["dice"] == 1 and r["iou"] == 1 and r["nsd"] == 1 and r["hd95"] == 0


def test_one_empty():
    z, g = np.zeros((224, 224), np.uint8), blob(50, 150, 60, 160)
    for r in (row(z, g), row(g, z)):
        assert r["dice"] == 0 and r["iou"] == 0 and r["nsd"] == 0
        assert math.isclose(r["hd95"], DIAG, rel_tol=1e-6)


def test_known_overlap():
    g, p = blob(0, 10, 0, 10), blob(5, 15, 0, 10)
    r = row(p, g)
    assert math.isclose(r["dice"], 0.5) and math.isclose(r["iou"], 1 / 3)
    assert r["fp_area"] == 50 and r["fn_area"] == 50 and r["n_components"] == 1


def test_spurious_components():
    g = blob(10, 40, 10, 40)
    p = g.copy()
    p[150:160, 150:160] = 1
    r = row(p, g)
    assert r["n_components"] == 2 and r["n_spurious"] == 1
