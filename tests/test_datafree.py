import json
from pathlib import Path

import pandas as pd

from shift_stress_test.analysis.levels import reach_levels
from shift_stress_test.config import load_config
from shift_stress_test.data.records import internal_ref, set_names
from shift_stress_test.layout import Layout

TINY = str(Path(__file__).resolve().parent / "configs" / "tiny.yaml")


def _layout(tmp_path):
    cfg = load_config(TINY, [f"paths.data={tmp_path / 'missing'}", f"paths.output={tmp_path / 'out'}", f"paths.splits={tmp_path / 'splits'}"])
    return cfg, Layout(cfg)


def test_names_come_from_the_manifest_without_images(tmp_path):
    cfg, layout = _layout(tmp_path)
    layout.splits.mkdir(parents=True)
    items = [{"id": i, "split": s, "image": f"data/toy/{s}/images/{i}.png"} for i, s in [("b", "test"), ("a", "test"), ("c", "train")]]
    (layout.splits / "toy.json").write_text(json.dumps({"items": items}))
    assert set_names(layout, internal_ref("toy", "test")) == ["a.png", "b.png"]


def test_reachability_falls_back_to_the_report(tmp_path):
    cfg, layout = _layout(tmp_path)
    shares = [1.0, 0.95, 0.9, 0.85, 0.7, 0.6, 0.5, 0.4]
    rows = [{"degradation": "G1", "level": k + 1, "reachable_share": s} for k, s in enumerate(shares)]
    path = layout.degrade_report("toy", "test")
    path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    k, share = reach_levels(layout, cfg, "toy", "test", "G1")
    assert k == 4 and list(share) == shares
