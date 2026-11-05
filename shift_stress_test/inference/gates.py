import numpy as np
import pandas as pd

from ..data.cache import load_cache
from ..data.records import internal_ref
from ..utils.io import load_json, save_json
from ..utils.log import get_logger


def update_gates(layout, name, payload):
    path = layout.gates_file
    gates = load_json(path) if path.exists() else {}
    gates[name] = payload
    save_json(gates, path)


def check_repro_train(cfg, layout, models, datasets, seeds):
    tol = cfg.gates.repro_train_tolerance
    rows = []
    for d in datasets:
        for m in models:
            for s in seeds:
                f = layout.eval_file(m, d, s, "best", "val", "clean")
                summary = layout.run_dir(m, d, s) / "summary.json"
                if not (f.exists() and summary.exists()):
                    continue
                diff = abs(float(pd.read_parquet(f, columns=["dice"])["dice"].mean()) - load_json(summary)["best_val_dice"])
                rows.append({"model": m, "dataset": d, "seed": s, "abs_diff": diff, "pass": diff <= tol})
    ok = bool(rows) and all(r["pass"] for r in rows)
    worst = max((r["abs_diff"] for r in rows), default=float("nan"))
    update_gates(layout, "G-repro-train", {"pass": ok, "tolerance": tol, "n_runs": len(rows), "worst_abs_diff": worst,
                                           "failures": [r for r in rows if not r["pass"]][:20]})
    get_logger().info(f"G-repro-train: {'PASS' if ok else 'FAIL'} ({len(rows)} runs, worst difference {worst:.2e})")
    if cfg.gates.strict and not ok:
        raise SystemExit("G-repro-train failed: the evaluation pipeline does not reproduce the training validation Dice")
    return ok


def check_integrity(cfg, layout, models, datasets, seeds, n_files=6, seed=0):
    rng = np.random.default_rng(seed)
    files = sorted(layout.out.glob("eval/*/*/seed*/best/test/*.masks.npz"))
    files = [f for f in files if f.parts[-6] in models and f.parts[-5] in datasets]
    if not files:
        return None
    picks = [files[i] for i in rng.choice(len(files), min(n_files, len(files)), replace=False)]
    if not all(layout.cache_file(f.parts[-5], "test").exists() or (layout.data / f.parts[-5] / "test" / "images").is_dir() for f in picks):
        get_logger().info("G-integrity: skipped, neither the image cache nor the data is available")
        return None
    worst = 0.0
    for f in picks:
        dataset = f.parts[-5]
        gt = load_cache(layout, cfg, internal_ref(dataset, "test"))["masks"]
        with np.load(f) as z:
            pred = np.unpackbits(z["masks"], axis=1)[:, : gt[0].size].reshape(-1, *gt[0].shape)
            index = z["index"]
        df = pd.read_parquet(f.with_name(f.name.replace(".masks.npz", ".parquet")), columns=["dice"])
        g = gt[index].astype(bool)
        p = pred.astype(bool)
        inter = (p & g).sum((1, 2))
        den = p.sum((1, 2)) + g.sum((1, 2))
        dice = np.where(den > 0, 2.0 * inter / np.maximum(den, 1), 1.0)
        worst = max(worst, float(np.abs(dice - df["dice"].to_numpy()).max()))
    ok = worst < 1e-6
    update_gates(layout, "G-integrity", {"pass": ok, "worst_abs_diff": worst, "files_checked": len(picks)})
    get_logger().info(f"G-integrity: {'PASS' if ok else 'FAIL'} ({len(picks)} files, worst difference {worst:.2e})")
    return ok
