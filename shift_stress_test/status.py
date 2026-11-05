import pandas as pd

from .degrade import degradations_for
from .inference.runner import job_done
from .config import seed_map
from .layout import Layout


def print_status(cfg, models=None, datasets=None, seeds=None):
    layout = Layout(cfg)
    models = models or list(cfg.experiment.models) + list(cfg.experiment.get("pretrained_models", []))
    datasets = datasets or list(cfg.experiment.datasets)
    seeds = seeds if seeds is not None else list(cfg.experiment.seeds)
    smap = seed_map(cfg, models, seeds)
    runs = [(m, d, s) for d in datasets for m in models for s in smap[m]]
    trained = sum((layout.run_dir(*r) / "DONE").exists() for r in runs)
    reported = {}
    for d in datasets:
        for sp in ("val", "test"):
            f = layout.degrade_report(d, sp)
            reported[d, sp] = set(pd.read_csv(f)["degradation"]) if f.exists() else set()
    degraded = sum(layout.degraded_file(d, sp, n).exists() or n in reported[d, sp]
                   for d in datasets for sp in ("val", "test") for n in degradations_for(d))
    n_deg = 2 * sum(len(degradations_for(d)) for d in datasets)
    val = sum(job_done(layout, cfg, "infer_val", *r) for r in runs if not r[0].endswith("_pt"))
    n_val = sum(not r[0].endswith("_pt") for r in runs)
    test = sum(job_done(layout, cfg, "infer_test", *r) for r in runs)
    rows = [("train runs", trained, len(runs)), ("degraded sets", degraded, n_deg),
            ("validation inference", val, n_val), ("test + external inference", test, len(runs))]
    frozen = (layout.prereg_dir / "prereg.json").exists()
    analysed = (layout.analysis_dir / "results.json").exists()
    print(f"output: {layout.out}")
    for name, done, total in rows:
        print(f"  {name:<28} {done:>5} / {total:<5} {'#' * int(30 * done / max(total, 1)):<30}")
    print(f"  {'prereg frozen':<28} {'yes' if frozen else 'no'}")
    print(f"  {'analysis':<28} {'done' if analysed else 'pending'}")
