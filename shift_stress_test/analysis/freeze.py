from pathlib import Path

import numpy as np
import pandas as pd

from ..inference.gates import update_gates
from ..utils.io import load_json, save_json, sha256_file
from ..utils.log import get_logger, progress
from .data import load_eval
from .descriptive import one_way_reliability
from .labelfree import choose_primary, dev_cells
from .levels import included
from .power import power_check


def code_hash():
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "shift_stress_test").rglob("*.py"))
    return {str(f.relative_to(root)): sha256_file(f) for f in files if f.exists()}


def val_reliability(layout, models, dataset, seeds):
    v = load_eval(layout, models, dataset, seeds, "val")
    runs = v.groupby(["degradation", "model", "seed"])["dice"].mean().reset_index()
    out = {}
    for name, sel in (("clean", runs["degradation"] == "clean"), ("degraded", runs["degradation"] != "clean")):
        mat = runs[sel].groupby(["model", "seed"])["dice"].mean().unstack()
        out[name] = one_way_reliability(mat.to_numpy())
    return out


def run_freeze(cfg, layout, models, datasets, seeds):
    log = get_logger()
    out = layout.prereg_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "prereg.json").exists():
        log.info("prereg.json already exists: the analysis is frozen, nothing is rewritten")
        return
    log.info("development step: choosing the label-free estimator on degraded validation cells (generic G1-G3)")
    dev = dev_cells(layout, cfg, models, datasets, seeds)
    dev.to_csv(out / "dev_cells.csv", index=False)
    choice = choose_primary(dev)
    save_json(choice, out / "estimator_choice.json")
    log.info(f"primary estimator: {choice['primary']} (L2 {choice['L2_variant']}, L3 {choice['L3_variant']})")

    cells = {d: {"val": included(layout, cfg, d, "val"), "test": included(layout, cfg, d, "test")} for d in datasets}
    save_json(cells, out / "included_cells.json")
    dropped = [f"{d}/{g}" for d, v in cells.items() for g, x in v["test"].items() if x["K"] == 0]
    log.info(f"degradations excluded by the 80% reachability rule (test): {dropped or 'none'}")

    rel = {d: val_reliability(layout, models, d, seeds) for d in datasets}
    rng = np.random.default_rng(cfg.analysis.seed)
    clean = [rel[d]["clean"] for d in datasets]
    deg = [rel[d]["degraded"] for d in datasets]
    power = power_check({"H1": (clean, deg), "H2": (deg, deg), "H4": (deg, clean)}, len(models), rng)
    save_json(power, out / "power_check.json")
    log.info("power check (minimal detectable pooled rho, 80% power): "
             + ", ".join(f"{k} {v['minimal_detectable_rho_80pct']}" for k, v in power.items()))

    files = {"config_used.yaml": layout.out / "config_used.yaml", "degrade_config.json": layout.out / "degrade_config.json",
             "estimator_choice.json": out / "estimator_choice.json", "power_check.json": out / "power_check.json",
             "included_cells.json": out / "included_cells.json"}
    for i, extra in enumerate(cfg.analysis.get("prereg_extra_files", [])):
        files[f"extra{i}_{Path(extra).name}"] = Path(extra)
    for d in datasets:
        for name in (d, cfg.experiment.external[d]):
            files[f"splits/{name}.json"] = layout.split_manifest(name)
    hashes = {k: sha256_file(v) for k, v in files.items() if Path(v).exists()}
    if cfg.analysis.hash_checkpoints:
        for m in models:
            for d in datasets:
                for s in progress(seeds, desc=f"hash {m}/{d}", leave=False):
                    for ck in ("best", "last"):
                        p = layout.run_dir(m, d, s) / f"{ck}.pt"
                        if p.exists():
                            hashes[f"train/{m}/{d}/seed{s}/{ck}.pt"] = sha256_file(p)
    gates = load_json(layout.gates_file) if layout.gates_file.exists() else {}
    save_json({"stage": "freeze", "note": "written before any test or external image is evaluated",
               "seeds": {"analysis": cfg.analysis.seed}, "levels": load_json(layout.out / "degrade_config.json")["levels"],
               "gates": gates, "file_sha256": hashes, "code_sha256": code_hash(), "models": models, "datasets": datasets,
               "train_seeds": seeds, "primary_estimator": choice["primary"]}, out / "prereg.json")
    update_gates(layout, "G-prereg", {"pass": True, "n_files": len(hashes)})
    update_gates(layout, "G-est", {"pass": True, "primary": choice["primary"], "n_dev_cells": int(len(dev))})
    pd.DataFrame([{"design": k, **r} for k, v in power.items() for r in v["table"]]).to_csv(out / "power_table.csv", index=False)
    log.info(f"prereg.json written ({len(hashes)} files hashed)")
