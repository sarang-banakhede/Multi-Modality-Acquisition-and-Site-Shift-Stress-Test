import numpy as np

from shift_stress_test.analysis.confirmatory import run_h1, run_h2, run_h4
from shift_stress_test.analysis.crossfit import DatasetContext
from shift_stress_test.config import load_config
from tests.fakes import FakeExternal, FakeTensor

CFG = load_config("tests/configs/tiny.yaml", ["analysis.permutation_reps=200", "analysis.bootstrap_reps=8"])


def run(seed, relation, shared):
    rng = np.random.default_rng(seed)
    ctxs = {}
    for d in range(2):
        T = FakeTensor(rng, relation=relation, shared=shared)
        T.dataset = f"d{d}"
        ctxs[T.dataset] = DatasetContext(T, FakeExternal(rng, T, shared), 4, rng)
    return {"H1": run_h1(ctxs, CFG, rng)[0]["p_permutation"], "H2": run_h2(ctxs, CFG, rng)["p_permutation"],
            "H4": run_h4(ctxs, CFG, rng)["p_permutation"]}


def test_false_positive_rate_under_the_null():
    ps = [run(s, 0.0, 0.0) for s in range(24)]
    for key in ("H1", "H2", "H4"):
        rate = np.mean([p[key] < 0.05 for p in ps])
        assert rate <= 0.2, (key, rate)


def test_power_when_a_relation_exists():
    ps = [run(100 + s, 0.9, 1.0) for s in range(6)]
    for key in ("H1", "H2", "H4"):
        assert np.mean([p[key] < 0.05 for p in ps]) >= 0.5, key
