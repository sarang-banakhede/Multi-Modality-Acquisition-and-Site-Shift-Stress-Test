import numpy as np

from shift_stress_test.analysis.labelfree import Cell, Reference, pair_agreement, pair_index, predict


def fake_unit(n_models=3, n_seeds=2, n=20, seed=0):
    rng = np.random.default_rng(seed)
    runs = [(f"m{i}", s) for i in range(n_models) for s in range(n_seeds)]
    r = len(runs)
    base = (rng.random((n, 28, 28)) > 0.5).astype(np.uint8)
    packed = np.stack([np.packbits(np.stack([base[i] if rng.random() > 0.2 else 1 - base[i] for i in range(n)]).reshape(n, -1), axis=1)
                       for _ in range(r)])
    return {"runs": runs, "image_id": np.arange(n), "level": np.zeros(n, int), "gidx": np.arange(n),
            "dice": rng.random((r, n)), "feats": {k: rng.random((r, n)) for k in ("conf_fg", "msp", "msp_fg", "entropy", "band_conf")},
            "packed": packed}


def test_identical_masks_have_agreement_one():
    packed = np.repeat(np.packbits(np.ones((1, 5, 64), np.uint8), axis=2), 4, axis=0)
    pairs = [(0, 1), (2, 3)]
    assert np.allclose(pair_agreement(packed, pairs), 1.0)


def test_estimators_return_validation_dice_when_target_equals_validation():
    unit = fake_unit()
    pairs = pair_index(unit["runs"])
    ref = Reference(unit, pairs)
    cell = Cell(unit, np.ones(len(unit["image_id"]), bool), pairs)
    for name in ("B0", "L2_band_conf", "L2_conf_fg", "L3_msp", "L3_msp_fg"):
        pred, _ = predict(name, ref, cell)
        assert np.allclose(pred, ref.D, atol=0.06), name
    pred, extra = predict("L1", ref, cell)
    assert np.allclose(pred, ref.D, atol=1e-6) and extra["r2"] > 0.99


def test_agreement_line_is_recovered():
    from scipy.stats import norm

    from shift_stress_test.analysis.data import probit

    rng = np.random.default_rng(0)
    r = 12
    a_v = rng.uniform(0.6, 0.95, r)
    a_t = norm.cdf(0.5 + 0.8 * probit(a_v))
    d_v = rng.uniform(0.5, 0.9, r)

    class Ref:
        D, agree, stress, feats = d_v, a_v, d_v, {}

    class CellX:
        agree = np.repeat(a_t[:, None], 5, axis=1)

    pred, extra = predict("L1", Ref, CellX)
    assert np.allclose(pred, norm.cdf(0.5 + 0.8 * probit(d_v)), atol=1e-6) and extra["r2"] > 0.999
