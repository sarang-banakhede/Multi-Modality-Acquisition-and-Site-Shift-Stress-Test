import numpy as np
import pytest

from shift_stress_test.config import load_config
from shift_stress_test.degrade import GENERIC, SPECIFIC, build_degradation
from shift_stress_test.degrade.match import match_image
from shift_stress_test.degrade.ssim import fov_mask, masked_ssim
from shift_stress_test.data.preprocess import luminance, resize_image

CFG = load_config(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "configs" / "config.yaml"))
RGB = ["isic2016", "kvasir_seg", "refuge2"]
CASES = [(d, n) for d in SPECIFIC for n in GENERIC + SPECIFIC[d]]


def image(dataset):
    rng = np.random.default_rng(0)
    h, w = 160, 200
    yy, xx = np.mgrid[0:h, 0:w]
    base = 90 + 60 * np.sin(xx / 9.0) * np.cos(yy / 11.0) + rng.normal(0, 4, (h, w))
    base[(yy - 80) ** 2 + (xx - 100) ** 2 < 40 ** 2] += 40
    base = np.clip(base, 5, 250)
    return np.stack([base, base * 0.8, base * 0.6], -1).astype(np.uint8) if dataset in RGB else base.astype(np.uint8)


@pytest.mark.parametrize("dataset,name", CASES)
def test_identity_dtype_shape_and_determinism(dataset, name):
    img = image(dataset)
    deg = build_degradation(name, CFG)
    state = deg.prepare(img, deg.rng(dataset, "img"))
    assert np.array_equal(deg(img, state, 0.0), img)
    a = deg(img, state, 0.6)
    state2 = deg.prepare(img, deg.rng(dataset, "img"))
    b = deg(img, state2, 0.6)
    assert a.dtype == np.uint8 and a.shape == img.shape
    assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("dataset,name", CASES)
def test_ssim_is_monotone_on_a_coarse_grid(dataset, name):
    img = image(dataset)
    deg = build_degradation(name, CFG)
    state = deg.prepare(img, deg.rng(dataset, "img"))
    clean = resize_image(img, 224)
    lum, mask = luminance(clean), fov_mask(luminance(clean))
    vals = [masked_ssim(lum, luminance(resize_image(deg(img, state, s), 224)), mask) for s in (0.1, 0.3, 0.6, 1.0)]
    assert all(b <= a + 0.03 for a, b in zip(vals, vals[1:])), vals
    assert vals[0] < 1.0


def test_matching_hits_the_target():
    img = image("kvasir_seg")
    deg = build_degradation("G2", CFG)
    out = match_image(deg, img, "kvasir_seg", "x", [0.95, 0.85, 0.75], 224, CFG.degrade)
    assert out["reachable"].all()
    assert np.allclose(out["ssim"], [0.95, 0.85, 0.75], atol=0.012)
    assert np.all(np.diff(out["strength"]) > 0)
