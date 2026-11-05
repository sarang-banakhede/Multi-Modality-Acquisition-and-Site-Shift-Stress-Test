import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data import probit  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dd"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
FAMILY = {"classic_cnn": (BLUE, "o", "Classic CNN"), "dilated_multiscale_cnn": (ORANGE, "s", "Dilated CNN"),
          "hybrid_cnn_transformer": (AQUA, "^", "Hybrid"), "pure_transformer": (GRAY, "D", "Transformer")}


def style():
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
                         "axes.titleweight": "regular", "axes.titlelocation": "left", "legend.frameon": False,
                         "lines.linewidth": 1.6, "lines.markersize": 6})


def _read(folder, name):
    p = folder / name
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return None


def _save(fig, folder, name):
    fig.savefig(folder / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _family_legend(ax, loc="best"):
    handles = [plt.Line2D([], [], color=c, marker=m, ls="", markeredgecolor=SURFACE, label=n) for c, m, n in FAMILY.values()]
    ax.legend(handles=handles, loc=loc, fontsize=8)


def _z(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-9)


def fig_accuracy_line(folder, out):
    curves, h1, e2 = _read(folder, "curves.csv"), _read(folder, "h1_cells.csv"), _read(folder, "e2_selection_regret.csv")
    ml = _read(folder, "model_level.csv")
    if curves is None or h1 is None:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))
    ax = axes[0, 0]
    d0 = curves["dataset"].iloc[0]
    g = "G2" if (curves[(curves.dataset == d0)]["degradation"] == "G2").any() else curves[curves.dataset == d0]["degradation"].iloc[0]
    sub = curves[(curves.dataset == d0) & (curves.degradation == g)]
    kmax = sub["level"].max()
    x, y = sub[sub.level == 0].set_index("model")["dice"], sub[sub.level == kmax].set_index("model")["dice"]
    fam = dict(zip(ml["model"], ml["family"])) if ml is not None else {}
    for m in x.index:
        c, mk, _ = FAMILY.get(fam.get(m, "classic_cnn"), FAMILY["classic_cnn"])
        ax.scatter(probit(x[m]), probit(y[m]), color=c, marker=mk, s=46, edgecolor=SURFACE, zorder=3)
        ax.annotate(m, (probit(x[m]), probit(y[m])), fontsize=7, color=INK2, xytext=(4, 3), textcoords="offset points")
    b = np.polyfit(probit(x.values), probit(y.values), 1)
    xs = np.linspace(probit(x.values).min(), probit(x.values).max(), 20)
    ax.plot(xs, np.polyval(b, xs), color=INK2, lw=1, ls="--")
    ax.set_xlabel("clean Dice (probit)")
    ax.set_ylabel(f"Dice at level {kmax} (probit)")
    ax.set_title(f"(a) {d0}, {g}: one cell of the line")
    _family_legend(ax, "lower right")
    ax = axes[0, 1]
    rows = h1.groupby(["dataset", "degradation"])["pearson"].agg(["mean", "min", "max"]).reset_index()
    rows = rows.sort_values(["dataset", "degradation"]).reset_index(drop=True)
    for i, r in rows.iterrows():
        col = BLUE if r["degradation"].startswith("G") else ORANGE
        ax.plot([r["min"], r["max"]], [i, i], color=col, lw=1.2, alpha=0.5)
        ax.plot(r["mean"], i, "o", color=col, markeredgecolor=SURFACE)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{a} {b}" for a, b in zip(rows["dataset"], rows["degradation"])], fontsize=6.5)
    ax.axvline(0, color=INK2, lw=0.8)
    if (folder / "results.json").exists():
        beta = json.load(open(folder / "results.json")).get("H1", {}).get("beta")
        if beta is not None:
            ax.axvline(beta, color=INK, lw=1, ls=":")
    ax.invert_yaxis()
    ax.set_xlabel("within-dataset correlation of clean and degraded Dice (range over levels)")
    ax.set_title("(b) does clean Dice predict degraded Dice?")
    ax.legend(handles=[plt.Line2D([], [], color=BLUE, marker="o", ls="", label="generic"),
                       plt.Line2D([], [], color=ORANGE, marker="o", ls="", label="specific")], fontsize=8, loc="center left")
    ax = axes[1, 0]
    if e2 is not None:
        e2 = e2.assign(kind=np.where(e2["degradation"].str.startswith("G"), "generic", "specific"))
        for kind, col in (("generic", BLUE), ("specific", ORANGE)):
            m = e2[e2.kind == kind].groupby("level")["regret"].mean()
            ax.plot(m.index, m.values, "-o", color=col, markeredgecolor=SURFACE, label=kind)
        ax.set_xlabel("severity level")
        ax.set_ylabel("regret of choosing by clean Dice (Dice points)")
        ax.legend(fontsize=8)
    ax.set_title("(c) selection regret grows with severity")
    ax = axes[1, 1]
    sub = curves[(curves.dataset == d0) & (curves.degradation == g)]
    for m, grp in sub.groupby("model"):
        c, mk, _ = FAMILY.get(fam.get(m, "classic_cnn"), FAMILY["classic_cnn"])
        grp = grp.sort_values("level")
        ax.plot(grp["achieved_ssim"], grp["dice"] / grp["dice"].iloc[0], color=c, marker=mk, ms=4, lw=1.2, alpha=0.9)
    for tol in (0.98, 0.95):
        ax.axhline(tol, color=INK2, lw=0.7, ls=":")
    ax.invert_xaxis()
    ax.set_xlabel("SSIM")
    ax.set_ylabel("Dice retained (ratio to clean)")
    ax.set_title(f"(d) retention curves, {d0} {g}")
    _family_legend(ax, "lower left")
    fig.tight_layout()
    _save(fig, out, "fig2_accuracy_on_the_line")


def _forest(ax, items, title):
    for i, (name, est, lo, hi, col) in enumerate(items):
        if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [i, i], color=col, lw=1.6)
        ax.plot(est, i, "D" if name.startswith("pooled") else "o", color=col, markeredgecolor=SURFACE, ms=7)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([n for n, *_ in items], fontsize=8)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.invert_yaxis()
    ax.set_title(title)


def fig_transfer(folder, out):
    ml, res = _read(folder, "model_level.csv"), (json.load(open(folder / "results.json")) if (folder / "results.json").exists() else {})
    if ml is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    for ax, (xk, yk, title, xl, yl) in zip(axes[:2], [("auc_er_generic", "auc_er_specific", "(a) generic to specific", "generic effective robustness (z)", "specific effective robustness (z)"),
                                                      ("auc_er_all", "er_ext", "(b) synthetic to real shift", "synthetic effective robustness (z)", "external effective robustness (z)")]):
        for d, grp in ml.groupby("dataset"):
            if xk not in grp or grp[xk].isna().any():
                continue
            zx, zy = _z(grp[xk]), _z(grp[yk])
            for (_, r), a, b in zip(grp.iterrows(), zx, zy):
                c, mk, _ = FAMILY.get(r["family"], FAMILY["classic_cnn"])
                ax.scatter(a, b, color=c, marker=mk, s=34, edgecolor=SURFACE, alpha=0.9)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title)
    _family_legend(axes[0], "lower right")
    items = []
    for key, label, col in (("H2", "H2 generic to specific", BLUE), ("H4", "H4 synthetic to real", ORANGE), ("beta_ext", "E10 clean to external", AQUA)):
        r = res.get(key) or {}
        per = r.get("per_dataset", {})
        for d, v in per.items():
            items.append((f"{label[:2]} {d}", v, None, None, col))
        if "rho" in r:
            items.append((f"pooled {label[:2]}", r["rho"], r.get("ci_low"), r.get("ci_high"), col))
        elif "beta_ext" in r:
            items.append(("pooled E10", r["beta_ext"], r.get("ci_low"), r.get("ci_high"), col))
    _forest(axes[2], items, "(c) correlation across the 8 models")
    axes[2].set_xlabel("Spearman / Pearson correlation")
    fig.tight_layout()
    _save(fig, out, "fig3_transfer")


def fig_drivers(folder, out):
    res = json.load(open(folder / "results.json")) if (folder / "results.json").exists() else {}
    e6 = _read(folder, "e6_tail_size_failure.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    ax = axes[0]
    h3 = (res.get("H3") or {}).get("shares_ml")
    if h3:
        names = list(h3)
        vals = [h3[k] for k in names]
        cols = [BLUE if k.startswith("model") else GRAY for k in names]
        ax.barh(range(len(names)), vals, color=cols, height=0.6)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("share of variance of the probit Dice drop")
    ax.set_title("(a) H3 variance shares (blue = architecture)")
    ax = axes[1]
    e9 = (res.get("E9_variance_shares") or {}).get("mean")
    if e9:
        names = list(e9)
        ax.barh(range(len(names)), [e9[k] for k in names], color=[BLUE if "model" in k else GRAY for k in names], height=0.6)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("share of per-image Dice loss")
    ax.set_title("(b) E9 image-level shares")
    ax = axes[2]
    if e6 is not None and "delta_fp_area" in e6:
        last = e6.sort_values("level").groupby(["dataset", "degradation", "model"]).tail(1)
        g = last.groupby("family")[["delta_fp_area", "delta_fn_area"]].mean()
        x = np.arange(len(g))
        ax.bar(x - 0.19, g["delta_fp_area"], width=0.36, color=ORANGE, label="false positive area")
        ax.bar(x + 0.19, g["delta_fn_area"], width=0.36, color=BLUE, label="false negative area")
        ax.set_xticks(x)
        ax.set_xticklabels([FAMILY.get(f, (0, 0, f))[2] for f in g.index], fontsize=8)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_ylabel("change vs clean prediction (px)")
        ax.legend(fontsize=8)
    ax.set_title("(c) failure types at the deepest level")
    fig.tight_layout()
    _save(fig, out, "fig4_drivers")


def fig_limits(folder, out):
    env, gap = _read(folder, "e5_operating_limits.csv"), _read(folder, "e5_limit_gap.csv")
    if env is None or gap is None or gap.empty:
        return
    g = gap[(gap.delta == 0.05)].groupby(["dataset", "degradation"])[["mean_limit", "individual_limit"]].mean().reset_index()
    g = g.sort_values(["dataset", "degradation"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.2, 0.22 * len(g) + 1.4))
    for i, r in g.iterrows():
        if np.isfinite(r["mean_limit"]) and np.isfinite(r["individual_limit"]):
            ax.plot([r["mean_limit"], r["individual_limit"]], [i, i], color=GRID, lw=2)
        ax.plot(r["mean_limit"], i, "o", color=BLUE, markeredgecolor=SURFACE, label="mean Dice within 5%" if i == 0 else None)
        ax.plot(r["individual_limit"], i, "s", color=ORANGE, markeredgecolor=SURFACE, label="90% of images within 0.05 Dice" if i == 0 else None)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels([f"{a} {b}" for a, b in zip(g["dataset"], g["degradation"])], fontsize=6.5)
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel("operating limit (SSIM down to which the model stays usable, mean over models)")
    ax.set_title("Individual-level limits are stricter than mean-based limits")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    _save(fig, out, "fig6_operating_limits")


def fig_labelfree(folder, out):
    summ, cf = _read(folder, "labelfree_summary.csv"), _read(folder, "e7_confident_failure.csv")
    res = json.load(open(folder / "results.json")) if (folder / "results.json").exists() else {}
    if summ is None:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    kinds = [k for k in ("heldout_specific", "external", "in_family") if k in set(summ["kind"])]
    cols = {"heldout_specific": ORANGE, "external": AQUA, "in_family": BLUE}
    for ax, metric, title in ((axes[0, 0], "regret", "(a) selection regret (Dice points, lower is better)"), (axes[0, 1], "mae", "(b) error of the estimated Dice")):
        est = list(dict.fromkeys(summ["estimator"]))
        x = np.arange(len(est))
        w = 0.8 / max(len(kinds), 1)
        for j, k in enumerate(kinds):
            vals = [summ[(summ.kind == k) & (summ.estimator == e)][metric].mean() for e in est]
            ax.bar(x + (j - (len(kinds) - 1) / 2) * w, vals, width=w * 0.92, color=cols[k], label=k.replace("_", " "))
        ax.set_xticks(x)
        ax.set_xticklabels(est, rotation=35, ha="right", fontsize=7.5)
        ax.set_title(title)
        ax.legend(fontsize=7.5)
    ax = axes[1, 0]
    h5 = res.get("H5") or {}
    items = []
    for label, d, col in (("primary", h5, ORANGE), ("stress test B1", h5.get("B1_stress_test") or {}, BLUE)):
        for key in ("combined", "specific_mild", "external"):
            r = d.get(key)
            if r:
                items.append((f"{label}: {key.replace('_', ' ')}", r["delta_regret"], r["ci_low"], r["ci_high"], col))
    _forest(ax, items, "(c) change in regret vs clean-Dice selection")
    ax.set_xlabel("Delta regret (negative = better than clean Dice)")
    ax = axes[1, 1]
    if cf is not None and not cf.empty:
        g = cf[cf["mild"]].groupby("family")[["failure_rate", "confident_failure_rate"]].mean()
        x = np.arange(len(g))
        ax.bar(x - 0.19, g["failure_rate"], width=0.36, color=GRAY, label="failure rate")
        ax.bar(x + 0.19, g["confident_failure_rate"], width=0.36, color=ORANGE, label="confident failure rate")
        ax.set_xticks(x)
        ax.set_xticklabels([FAMILY.get(f, (0, 0, f))[2] for f in g.index], fontsize=8)
        ax.legend(fontsize=8)
    ax.set_title("(d) failures in the mild range")
    fig.tight_layout()
    _save(fig, out, "fig5_label_free")


def make_figures(layout):
    style()
    folder = layout.analysis_dir
    out = folder / "figures"
    out.mkdir(exist_ok=True)
    for fn in (fig_accuracy_line, fig_transfer, fig_drivers, fig_limits, fig_labelfree):
        fn(folder, out)
    return out
