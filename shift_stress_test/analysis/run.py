import json
import traceback

import numpy as np
import pandas as pd

from ..inference.gates import check_integrity, update_gates
from ..utils.io import load_json, save_json
from ..utils.log import get_logger, progress
from .arms import pretrained_arm
from .confirmatory import e1_cells, run_h1, run_h2, run_h4
from .crossfit import DatasetContext
from .data import DatasetTensor, ExternalTensor
from .descriptive import (kendall_w_summary, mce_agreement, metric_sensitivity, pairwise_models, reliability,
                          score_summary, selection_regret, subgroup_table, variance_shares)
from .figures import make_figures
from .external import best_last_agreement, beta_ext, external_selection
from .heldout import collect, h5_bootstrap, summarise
from .levels import planned_levels
from .limits import confident_failure, envelope_gap, envelope_table, tail_table
from .meta import benjamini_hochberg, holm
from .mixed import h3_frame, run_h3
from .report import clean_tables, curve_table, model_level_table
from .scores import score_table
from .sensitivity import disattenuated, per_source_h4, run_variants


def run_analysis(cfg, layout, models, datasets, seeds):
    log = get_logger()
    if not (layout.prereg_dir / "prereg.json").exists():
        raise SystemExit("prereg.json is missing: run the 'freeze' stage first")
    out = layout.analysis_dir
    out.mkdir(parents=True, exist_ok=True)
    a = cfg.analysis
    rng = np.random.default_rng(a.seed)
    levels = planned_levels(layout, cfg)
    primary = load_json(layout.prereg_dir / "estimator_choice.json")["primary"]
    check_integrity(cfg, layout, models, datasets, seeds)
    errors, results = [], {"levels": levels, "primary_estimator": primary, "models": models, "seeds": seeds}

    def step(name, fn):
        log.info(f"analysis: {name}")
        try:
            return fn()
        except Exception:
            log.info(f"analysis step {name} FAILED\n{traceback.format_exc()}")
            errors.append(name)
            return None

    T, X = {}, {}
    for d in progress(datasets, desc="loading results"):
        t = DatasetTensor(layout, cfg, d, models, seeds, levels)
        if not t.degs:
            log.info(f"{d}: no degradation reaches the minimum number of levels, dataset skipped")
            continue
        T[d], X[d] = t, ExternalTensor(layout, cfg, d, models, seeds)
    datasets = list(T)
    results["datasets"] = datasets
    results["levels_per_degradation"] = {d: T[d].K for d in datasets}
    results["excluded_degradations"] = {d: T[d].excluded for d in datasets}
    ctxs = {d: DatasetContext(T[d], X[d], int(a.halves), rng) for d in datasets}

    def tables():
        clean, train = clean_tables(layout, models, datasets, seeds)
        clean.to_csv(out / "clean_and_external_per_run.csv", index=False)
        train.to_csv(out / "training_summary.csv", index=False)
        best_last_agreement(clean).to_csv(out / "best_vs_last.csv", index=False)
        pd.concat([curve_table(T[d]) for d in datasets], ignore_index=True).to_csv(out / "curves.csv", index=False)
        pd.concat([model_level_table(T[d], X[d]) for d in datasets], ignore_index=True).to_csv(out / "model_level.csv", index=False)
        scores = pd.concat([score_table(T[d], a.mild_levels) for d in datasets], ignore_index=True)
        scores.to_csv(out / "scores.csv", index=False)
        score_summary(scores).to_csv(out / "score_summary.csv", index=False)
        mce_agreement(scores).to_csv(out / "mce_agreement.csv", index=False)
        rel = pd.DataFrame([reliability(T[d], X[d]) for d in datasets])
        rel.to_csv(out / "reliability.csv", index=False)
        sub = pd.concat([subgroup_table(T[d]) for d in datasets], ignore_index=True)
        if len(sub):
            sub.to_csv(out / "subgroups.csv", index=False)
        return scores, rel

    tab = step("tables, scores, reliability", tables)
    scores, rel = tab if tab else (None, None)

    h1 = step("H1 accuracy on the line", lambda: run_h1(ctxs, cfg, rng))
    if h1:
        results["H1"] = h1[0]
        h1[1].to_csv(out / "h1_cells.csv", index=False)
    results["H2"] = step("H2 generic to specific", lambda: run_h2(ctxs, cfg, rng))
    results["H4"] = step("H4 synthetic to real", lambda: run_h4(ctxs, cfg, rng))
    results["beta_ext"] = step("E10 accuracy on the line under real shift", lambda: beta_ext(ctxs, cfg, rng))
    results["H3"] = step("H3 variance components", lambda: run_h3(h3_frame(list(T.values())), int(a.h3_bootstrap_reps), rng))

    labelfree = {}

    def heldout():
        tables_, store = [], {}
        for d in datasets:
            df, entry = collect(layout, cfg, models, seeds, T[d], X[d], primary)
            tables_.append(df)
            store[d] = entry
        cells = pd.concat(tables_, ignore_index=True)
        cells.to_csv(out / "labelfree_cells.csv", index=False)
        labelfree["cells"] = cells
        summ, tau = summarise(cells, primary)
        summ.to_csv(out / "labelfree_summary.csv", index=False)
        tau.to_csv(out / "labelfree_tau_bh.csv", index=False)
        reps = int(a.bootstrap_reps)
        return {"primary": h5_bootstrap(store, primary, a.mild_levels, reps, rng),
                "B1_stress_test": h5_bootstrap(store, "B1", a.mild_levels, max(reps // 4, 1), rng)}

    results["H5"] = step("H5 label-free selection", heldout)
    if results["H5"]:
        results["H5"] = {**results["H5"]["primary"], "B1_stress_test": results["H5"]["B1_stress_test"]}

    ps = []
    for k in ("H1", "H2", "H3", "H4", "H5"):
        r = results.get(k)
        if not r:
            continue
        p = (r["combined"]["p_bootstrap"] if k == "H5" else r.get("p_permutation", r.get("p_bootstrap", r.get("p_asymptotic"))))
        if p is not None and np.isfinite(p):
            ps.append((k, p))
    if ps:
        for (k, _), pa in zip(ps, holm([p for _, p in ps])):
            results[k]["p_holm"] = float(pa)

    def sensitivity():
        run_variants(T, X, cfg, rng, datasets).to_csv(out / "sensitivity_variants.csv", index=False)
        if rel is not None:
            disattenuated(results, rel).to_csv(out / "sensitivity_disattenuated.csv", index=False)
        per_source_h4(T, X, cfg, rng, datasets).to_csv(out / "sensitivity_per_external_source.csv", index=False)

    step("sensitivity checks", sensitivity)

    def exploratory():
        e1 = pd.DataFrame(e1_cells(ctxs, cfg, rng))
        if len(e1):
            e1["p_bh"] = benjamini_hochberg(e1["p_perm"].to_numpy())
        e1.to_csv(out / "e1_cell_associations.csv", index=False)
        pd.concat([selection_regret(T[d], rng, min(int(a.bootstrap_reps), 200)) for d in datasets]).to_csv(
            out / "e2_selection_regret.csv", index=False)
        if scores is not None:
            summary, pairs = pairwise_models(scores)
            pairs.to_csv(out / "e3_pairwise_models.csv", index=False)
            results["E3"] = summary
        sens, ret = zip(*[metric_sensitivity(T[d], X[d], cfg, rng) for d in datasets])
        pd.concat(sens).to_csv(out / "e4_metric_sensitivity_h1.csv", index=False)
        pd.concat(ret).to_csv(out / "e4_metric_retention.csv", index=False)
        results["E8_kendall_w_seeds"] = kendall_w_summary(T, rng)
        shares = {d: variance_shares(T[d]) for d in datasets}
        keys = next(iter(s for s in shares.values() if s), {})
        results["E9_variance_shares"] = {"per_dataset": shares, "mean": {k: float(np.mean([s[k] for s in shares.values() if s])) for k in keys}}

    step("exploratory E1-E4, E8, E9", exploratory)

    def limits():
        env = pd.concat([envelope_table(T[d], cfg, rng) for d in datasets])
        env.to_csv(out / "e5_operating_limits.csv", index=False)
        envelope_gap(env).to_csv(out / "e5_limit_gap.csv", index=False)

    step("E5 operating limits", limits)
    step("E6 tail risk, size and failure types", lambda: pd.concat([tail_table(T[d]) for d in datasets]).to_csv(
        out / "e6_tail_size_failure.csv", index=False))
    step("E7 confident failure", lambda: pd.concat([confident_failure(layout, cfg, T[d], models, seeds, rng) for d in datasets]).to_csv(
        out / "e7_confident_failure.csv", index=False))

    def ext_selection():
        pd.concat([external_selection(T[d], X[d], labelfree["cells"], primary) for d in datasets]).to_csv(
            out / "e10_external_selection.csv", index=False)

    if "cells" in labelfree:
        step("E10 external model selection", ext_selection)
    pt = [m for m in cfg.experiment.get("pretrained_models", [])]
    if pt:
        step("E11 pretrained arm", lambda: pretrained_arm(layout, cfg, datasets, seeds, models, pt, levels, rng).to_csv(
            out / "e11_pretrained_arm.csv", index=False))
    results["errors"] = errors
    save_json(results, out / "results.json")
    step("figures", lambda: make_figures(layout))
    update_gates(layout, "analysis", {"errors": errors})
    brief = {k: {x: v for x, v in results[k].items() if not isinstance(v, (dict, list))} for k in ("H1", "H2", "H3", "H4") if results.get(k)}
    log.info("confirmatory results:\n" + json.dumps(brief, indent=1, default=str))
    if errors:
        raise RuntimeError(f"analysis finished with failed steps: {errors}")
