import numpy as np

from shift_stress_test.analysis.limits import mean_limit, pass_limit
from shift_stress_test.analysis.meta import benjamini_hochberg, holm, reml_meta
from shift_stress_test.analysis.scores import area, curve_scores, er_curves


def test_area_of_constant_curve():
    assert np.isclose(area(np.full(9, 0.8)), 0.8)


def test_flat_curve_has_zero_loss():
    d = np.tile(np.linspace(0.6, 0.9, 8)[:, None], (1, 9))
    s = curve_scores(d, mild=2)
    assert np.allclose(s["auc_loss"], 0) and np.allclose(s["auc_rel"], 1)


def test_area_matches_trapezoid():
    d = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    assert np.isclose(area(d), np.trapezoid(d, dx=0.25))


def test_effective_robustness_is_zero_on_a_line():
    clean = np.linspace(0.5, 0.9, 8)
    d = np.stack([clean, clean * 0.9], axis=1)
    from shift_stress_test.analysis.data import probit
    p = probit(d)
    line = np.stack([p[:, 0], 0.7 * p[:, 0] + 0.1], axis=1)
    from scipy.stats import norm
    er = er_curves(norm.cdf(line))
    assert np.allclose(er, 0, atol=1e-6)


def test_mean_limit_interpolates():
    ssim = np.array([1.0, 0.95, 0.90, 0.85])
    ret = np.array([1.0, 0.99, 0.97, 0.90])
    assert 0.90 < mean_limit(ret, ssim, 0.02) < 0.95
    assert np.isnan(mean_limit(np.ones(4), ssim, 0.02))


def test_mean_limit_uses_first_crossing_on_non_monotone_curves():
    ssim = np.array([1.0, 0.95, 0.90, 0.85])
    ret = np.array([1.0, 0.90, 0.99, 0.99])
    assert 0.95 < mean_limit(ret, ssim, 0.05) < 1.0


def test_pass_limit():
    ssim = np.array([1.0, 0.9, 0.8])
    share = np.array([1.0, 0.95, 0.80])
    assert 0.8 < pass_limit(share, ssim, 0.9) < 0.9


def test_holm_and_bh():
    p = np.array([0.01, 0.04, 0.03])
    assert np.allclose(holm(p), [0.03, 0.06, 0.06])
    assert np.allclose(benjamini_hochberg(p), [0.03, 0.04, 0.04])


def test_reml_without_heterogeneity():
    z, se = np.array([0.3, 0.3, 0.3]), np.array([0.1, 0.1, 0.1])
    m = reml_meta(z, se)
    assert np.isclose(m["mu"], 0.3) and m["tau2"] < 1e-3
