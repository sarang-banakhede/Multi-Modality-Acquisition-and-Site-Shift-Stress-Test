import numpy as np

from shift_stress_test.analysis.levels import reach_levels
from shift_stress_test.config import Config


class Layout:
    def __init__(self, root):
        self.root = root
        self.out = root

    def degraded_file(self, dataset, split, name):
        return self.root / f"{name}.npz"


def make(tmp_path, shares, n=100):
    reach = np.array([np.arange(n) < int(s * n) for s in shares])
    np.savez(tmp_path / "G1.npz", reachable=reach)
    cfg = Config({"degrade": {"levels": [0.95 - 0.05 * i for i in range(len(shares))], "min_reachable": 0.8},
                  "analysis": {"min_levels": 3}})
    return Layout(tmp_path), cfg


def test_prefix_stops_at_first_level_below_the_threshold(tmp_path):
    layout, cfg = make(tmp_path, [1, 0.95, 0.9, 0.7, 0.99])
    assert reach_levels(layout, cfg, "d", "test", "G1")[0] == 3


def test_degradation_with_too_few_levels_is_excluded(tmp_path):
    layout, cfg = make(tmp_path, [1, 0.9, 0.5, 1])
    assert reach_levels(layout, cfg, "d", "test", "G1")[0] == 0


def test_all_levels_reachable(tmp_path):
    layout, cfg = make(tmp_path, [1, 1, 1, 0.85])
    assert reach_levels(layout, cfg, "d", "test", "G1")[0] == 4
