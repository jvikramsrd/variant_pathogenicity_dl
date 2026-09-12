"""Generate Figures 2-6 of the manuscript from committed run artifacts.

Every number plotted here is read from a results CSV in ``data/processed/``;
nothing is recomputed, so a figure cannot drift from the table it illustrates.
Figure 1 is a schematic and is drawn, not plotted; everything else is here.
Figure 4 needs ``scripts/recalibrate_grid.py`` to have run, and is skipped with
a note when its inputs are absent.

    python scripts/make_figures.py [--out_dir docs/figures]

Palette: three categorical hues validated for colour-vision deficiency at all
pairs (worst CVD dE 9.2, normal-vision dE 24.0). The aqua slot sits below 3:1
against the surface, so every series is also direct-labelled and given its own
marker shape -- identity is never carried by hue alone, in print or on screen.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.provenance import REPLICATE_KEYS, assert_comparable  # noqa: E402
GRID_DIR = ROOT / "data/processed/stage2b_grid"
BASELINE_CSV = ROOT / "data/processed/mmr_transfer_scratch/mmr_transfer_results_lopo.csv"
MASTER_CSV = ROOT / "data/mmr/processed/extended/extended_dataset.csv"
CALIB_PANEL = GRID_DIR / "calibration_panel.csv"
CALIB_PROBS = GRID_DIR / "calibration_probs.csv"

#: The label sources leave-one-gene-out trains and scores on. The DMS pool is
#: excluded upstream by prepare_split, so a composition figure that counted it
#: as evaluable would overstate the cohort by a factor of 25.
CLINICAL_SOURCES = ("clinvar", "pg_clinical")
LABEL_SOURCE_NAMES = {"dms": "DMS (proxy)", "clinvar": "ClinVar",
                      "pg_clinical": "ProteinGym clinical"}

#: The arm Figures 3 and 6 highlight; Figure 4 reports its calibration.
HEADLINE_CELL = "esmpri_concat_frozen_pllr-residual_seed42"

SCOREABLE = ["MLH1", "MSH2", "MSH6"]
GENES = SCOREABLE + ["PMS2"]

# Categorical slots 1-3 of the validated palette, plus recessive ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#dcdbd4"

# Hue x shape: the secondary encoding that keeps the arms apart in greyscale.
ARMS = [
    ("Curated priors only, no ESM", BLUE, "o"),
    ("ESM-2 650M frozen + priors", ORANGE, "s"),
    ("ESM-2 650M full FT + priors", AQUA, "^"),
]


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.fontsize": 8,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def load_cells() -> pd.DataFrame:
    """Every fine-tune cell and ablation, one row per (cell, held-out gene)."""
    frames = []
    for f in sorted(glob.glob(str(GRID_DIR / "esm_finetune_results_siamese_lopo_*.csv"))):
        frames.append(pd.read_csv(f))
    d = pd.concat(frames, ignore_index=True)
    d["family"] = np.where(d.cell_slug.str.startswith("ablate_"), "ablation", "grid")
    return d


def load_baseline() -> pd.DataFrame:
    d = pd.read_csv(BASELINE_CSV)
    d["cell_slug"] = "priors_only"
    d["family"] = "baseline"
    return d


def mean3(d: pd.DataFrame, col: str = "roc_auc") -> pd.Series:
    """Mean over the scoreable genes -- the convention used throughout the paper."""
    return d[d.holdout_gene.isin(SCOREABLE)].groupby("cell_slug")[col].mean()


def arm_of(slug: str) -> str:
    """Collapse a cell slug to its arm, dropping the seed suffix.

    Seeds are replicates of one arm, not separate arms: plotting them as
    separate rows would show the same configuration three times and invite
    the reader to rank two draws of the same cell against each other.
    """
    return re.sub(r"_seed\d+$", "", slug)


def by_arm(d: pd.DataFrame) -> pd.DataFrame:
    """Per-arm mean, SD and seed count of the scoreable-gene AUROC."""
    per_cell = mean3(d).rename("m").reset_index()
    per_cell["arm"] = per_cell.cell_slug.map(arm_of)
    g = per_cell.groupby("arm").m.agg(["mean", "std", "size"])
    g["std"] = g["std"].fillna(0.0)
    return g.sort_values("mean")


# --------------------------------------------------------------------------- fig 3
def figure3(cells: pd.DataFrame, base: pd.DataFrame, out: Path) -> None:
    """Per-gene AUROC with bootstrap CIs for the three headline arms."""
    stem = "esmpri_concat_{}_pllr-residual_seed{}"
    arms = {
        ARMS[0][0]: base,
        ARMS[1][0]: cells[cells.cell_slug == stem.format("frozen", 42)],
        ARMS[2][0]: cells[cells.cell_slug == stem.format("full", 42)],
    }
    extra_seeds = {
        ARMS[1][0]: [stem.format("frozen", s) for s in (43, 44)],
        ARMS[2][0]: [stem.format("full", s) for s in (43, 44)],
    }

    fig, ax = plt.subplots(figsize=(6.6, 3.6), layout="constrained")
    ax.set_axisbelow(True)
    ax.xaxis.grid(True)
    offsets = [-0.26, 0.0, 0.26]   # legend order, top to bottom

    for gi, gene in enumerate(GENES):
        if gene == "PMS2":                       # the unscoreable fold
            ax.axhspan(gi - 0.5, gi + 0.5, color=GRID, alpha=0.35, lw=0)
        for (label, colour, marker), off in zip(ARMS, offsets):
            row = arms[label][arms[label].holdout_gene == gene]
            if row.empty:
                continue
            y = gi + off
            x = row.roc_auc.iloc[0]
            lo, hi = row.roc_auc_ci_low.iloc[0], row.roc_auc_ci_high.iloc[0]
            ax.plot([lo, hi], [y, y], color=colour, lw=2, solid_capstyle="round",
                    zorder=2)
            ax.plot([x], [y], marker=marker, ms=6.5, color=colour, mec="white",
                    mew=1.0, ls="none", zorder=3, label=label if gi == 0 else None)
            for slug in extra_seeds.get(label, []):
                r = cells[(cells.cell_slug == slug) & (cells.holdout_gene == gene)]
                if not r.empty:                  # seed replicates, open marks
                    ax.plot(r.roc_auc, [y], marker=marker, ms=4, mfc="white",
                            mec=colour, mew=1.0, ls="none", zorder=4)

    ax.set_yticks(range(len(GENES)))
    ax.set_yticklabels([f"$\\it{{{g}}}$" for g in GENES])
    ax.set_ylim(len(GENES) - 0.5, -0.5)
    ax.set_xlim(0.82, 1.015)
    ax.set_xlabel("AUROC on the held-out gene (95% bootstrap CI, 10,000 resamples)")
    fig.suptitle("Held-out-gene discrimination: curated features versus ESM-2",
                 color=INK, x=0.005, ha="left", fontsize=9.5)
    ax.text(1.012, 3 + 0.42, "not scoreable: 21 variants, 4 negatives",
            ha="right", va="center", fontsize=7, color=INK_2, style="italic")
    handles, _ = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker="o", ls="none", mfc="white",
                              mec=MUTED, mew=1.0, ms=4,
                              label="seeds 43, 44 (point only)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=2,
               handletextpad=0.4, columnspacing=1.4, labelcolor=INK_2)
    fig.savefig(out / "fig3_main.png")
    fig.savefig(out / "fig3_main.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- fig 5
def figure5(cells: pd.DataFrame, base: pd.DataFrame, out: Path) -> None:
    """All arms, seed-averaged, ranked against the curated-features baseline."""
    g = by_arm(cells)
    fam = cells.assign(arm=cells.cell_slug.map(arm_of)).groupby("arm").family.first()
    baseline = mean3(base).iloc[0]

    # This figure is half a picture without the feature-family ablations, and
    # it would still render: grid cells alone plot fine and the legend would go
    # on advertising a series with no points. That is exactly the silent drift
    # the provenance gate exists to catch, so say it out loud instead.
    if not (fam == "ablation").any():
        print("  WARNING: fig 5 has no feature-family ablation arms. "
              "esm_finetune_results_siamese_lopo_ablate_*.csv is absent from "
              f"{GRID_DIR.name}; the ablation series will be empty and Table 4 "
              "has no artifacts behind it.")

    fig, ax = plt.subplots(figsize=(6.6, 4.8), layout="constrained")
    ax.set_axisbelow(True)
    ax.xaxis.grid(True)

    ax.axvline(baseline, color=BLUE, lw=1.4, zorder=1)
    # Right-aligned into the plot: left-aligned, the label runs off the axis
    # whenever the baseline sits near the upper limit, as it does here.
    ax.text(baseline - 0.003, len(g) + 0.15,
            f"curated priors only, no ESM · {baseline:.3f}",
            ha="right", va="center", fontsize=7.5, color=BLUE)

    for i, (arm, row) in enumerate(g.iterrows()):
        is_abl = fam[arm] == "ablation"
        colour = ORANGE if is_abl else MUTED
        val, sd, n = row["mean"], row["std"], int(row["size"])
        ax.plot([0.83, val], [i, i], color=GRID, lw=0.8, zorder=1)
        if n > 1:                                # +- 1 SD over the seeds
            ax.plot([val - sd, val + sd], [i, i], color=colour, lw=2,
                    solid_capstyle="round", zorder=2)
        ax.plot([val], [i], "o" if is_abl else "s", ms=6 if is_abl else 4.5,
                color=colour, mec="white", mew=0.8, zorder=3)
        label = f"{val:.3f}" + (f" ± {sd:.3f}" if n > 1 else "")
        ax.text(val + sd + 0.0035, i, label, va="center", fontsize=7,
                color=INK if is_abl else INK_2,
                fontweight="bold" if is_abl else "normal",
                bbox=dict(fc="white", ec="none", pad=0.8))

    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(
        [f"{a}  ({int(g.loc[a, 'size'])} seed{'s' if g.loc[a, 'size'] > 1 else ''})"
         for a in g.index], fontsize=7, color=INK_2)
    for tick, arm in zip(ax.get_yticklabels(), g.index):
        if fam[arm] == "ablation":
            tick.set_color(INK)
            tick.set_fontweight("bold")
    ax.set_ylim(-0.7, len(g) + 0.5)
    ax.set_xlim(0.83, 0.995)
    ax.set_xlabel("Mean AUROC over the scoreable genes ($\\it{MLH1}$, $\\it{MSH2}$, $\\it{MSH6}$)")
    fig.suptitle("Every arm, seed-averaged, against the curated-features baseline",
                 color=INK, x=0.005, ha="left", fontsize=9.5)
    handles = [plt.Line2D([], [], marker="s", ls="none", color=MUTED, mec="white",
                          ms=4.5, label="grid cell"),
               plt.Line2D([], [], color=MUTED, lw=2, label="± 1 SD over seeds")]
    # Only claim the ablation series when there is one. A legend entry with no
    # points on the axes is a caption that misdescribes its own figure.
    if (fam == "ablation").any():
        handles.insert(0, plt.Line2D([], [], marker="o", ls="none", color=ORANGE,
                                     mec="white", ms=6,
                                     label="feature-family ablation"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3,
               labelcolor=INK_2, columnspacing=1.6)
    fig.savefig(out / "fig5_ablation.png")
    fig.savefig(out / "fig5_ablation.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- fig 6
def figure6(cells: pd.DataFrame, base: pd.DataFrame, out: Path) -> None:
    """Per-gene spread across all arms, with cohort sizes and the PMS2 caveat."""
    allrows = pd.concat([cells, base], ignore_index=True)
    stem = "esmpri_concat_{}_pllr-residual_seed42"
    highlight = {ARMS[0][0]: "priors_only",
                 ARMS[1][0]: stem.format("frozen"),
                 ARMS[2][0]: stem.format("full")}

    fig, ax = plt.subplots(figsize=(6.6, 3.7), layout="constrained")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    rng = np.random.default_rng(0)
    n_holdout: dict[str, int] = {}

    for gi, gene in enumerate(GENES):
        sub = allrows[allrows.holdout_gene == gene]
        if gene == "PMS2":
            ax.axvspan(gi - 0.5, gi + 0.5, color=GRID, alpha=0.35, lw=0)
        jitter = rng.uniform(-0.16, 0.16, len(sub))
        ax.plot(gi + jitter, sub.roc_auc, "o", ms=3.4, color=MUTED, mec="none",
                alpha=0.75, ls="none", zorder=2)
        for (label, colour, marker) in ARMS:
            r = sub[sub.cell_slug == highlight[label]]
            if r.empty:
                continue
            ax.plot([gi], r.roc_auc, marker=marker, ms=7, color=colour,
                    mec="white", mew=1.1, ls="none", zorder=4,
                    label=label if gi == 0 else None)
        n_holdout[gene] = int(sub.n_holdout.iloc[0])

    ax.text(3, 0.885, "only 4 negatives —\nexcluded from every mean",
            ha="center", va="center", fontsize=7, color=INK_2, style="italic")
    ax.set_xticks(range(len(GENES)))
    ax.set_xticklabels([f"$\\it{{{g}}}$\nn = {n_holdout[g]}" for g in GENES])
    ax.set_xlim(-0.5, len(GENES) - 0.5)
    ax.set_ylim(0.775, 1.02)
    ax.set_ylabel("AUROC on the held-out gene")
    fig.suptitle("Per-gene performance across all arms", color=INK,
                 x=0.005, ha="left", fontsize=9.5)
    fig.legend(loc="outside lower center", ncol=3,
               handletextpad=0.4, columnspacing=1.6, labelcolor=INK_2)
    fig.savefig(out / "fig6_pergene.png")
    fig.savefig(out / "fig6_pergene.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- fig 2
def figure2(out: Path) -> None:
    """Dataset composition: what was assembled, what carries a label, what is scored.

    The three panels narrow deliberately. (a) is the assembled table, (b) the
    labelled subset, (c) the rows any model actually sees. The drop from 74,328
    to 683 is the paper's central limitation, and a composition figure that
    stopped at (b) would hide it behind a five-figure count that is real but
    is not the evaluation cohort.
    """
    m = pd.read_csv(MASTER_CSV, low_memory=False)
    lab = m[m.label.notna()].copy()
    clin = lab[lab.label_source.isin(CLINICAL_SOURCES)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0), layout="constrained",
                             gridspec_kw={"width_ratios": [1.05, 1.0, 1.15]})

    # (a) evidence-source coverage over the assembled table -------------------
    ax = axes[0]
    counts = (m.sources.fillna("").str.split("|").explode()
              .replace("", np.nan).dropna().value_counts())
    y = np.arange(len(counts))[::-1]
    ax.barh(y, counts.to_numpy(), color=MUTED, height=0.68, zorder=2)
    for yi, v in zip(y, counts.to_numpy()):
        ax.text(v + counts.max() * 0.02, yi, f"{v:,}", va="center",
                fontsize=6.8, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels([s.replace("_", " ") for s in counts.index], fontsize=7)
    ax.set_xlim(0, counts.max() * 1.28)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_title(f"(a)  Evidence sources\n{len(m):,} assembled variants",
                 fontsize=8.5, loc="left", color=INK)

    # (b) labelled subset by provenance, benign vs pathogenic -----------------
    ax = axes[1]
    order = ["dms", "clinvar", "pg_clinical"]
    order = [s for s in order if s in set(lab.label_source)]
    h = 0.34
    for i, (lv, colour, name) in enumerate([(0.0, BLUE, "benign"),
                                            (1.0, ORANGE, "pathogenic")]):
        vals = [int(((lab.label_source == s) & (lab.label == lv)).sum())
                for s in order]
        yy = np.arange(len(order))[::-1] + (h / 2 if i == 0 else -h / 2)
        ax.barh(yy, vals, height=h, color=colour, label=name, zorder=2)
        for yi, v in zip(yy, vals):
            ax.text(v * 1.35, yi, f"{v:,}", va="center", fontsize=6.6, color=INK_2)
    ax.set_xscale("log")
    ax.set_xlim(1, 10 ** 5.4)
    ax.set_yticks(np.arange(len(order))[::-1])
    ax.set_yticklabels([LABEL_SOURCE_NAMES.get(s, s) for s in order], fontsize=7)
    ax.set_xlabel("variants (log scale)", fontsize=7.5)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_title(f"(b)  Labelled subset\n{len(lab):,} of {len(m):,} carry a label",
                 fontsize=8.5, loc="left", color=INK)
    # Direct labels rather than a legend: the panel has two series and no room
    # for a box that does not overlap a value.
    row_dms = (len(order) - 1) - order.index("dms")
    # Inside the bar, in white: the only placement on a log axis that cannot
    # collide with the value label to its right.
    ax.text(1.7, row_dms + h / 2, "benign", fontsize=6.8, color="white",
            va="center", ha="left", fontweight="bold")
    ax.text(1.7, row_dms - h / 2, "pathogenic", fontsize=6.8, color="white",
            va="center", ha="left", fontweight="bold")
    ax.text(1.3, row_dms - 0.43, "a single MSH2 assay", fontsize=6.4,
            style="italic", color=INK_2, va="center")

    # (c) the cohort leave-one-gene-out actually scores ------------------------
    ax = axes[2]
    x = np.arange(len(GENES))
    bottoms = np.zeros(len(GENES))
    bars = [("clinvar", 0.0, BLUE, "", "ClinVar benign"),
            ("pg_clinical", 0.0, BLUE, "///", "PG-clinical benign"),
            ("clinvar", 1.0, ORANGE, "", "ClinVar pathogenic"),
            ("pg_clinical", 1.0, ORANGE, "///", "PG-clinical pathogenic")]
    for src, lv, colour, hatch, name in bars:
        vals = np.array([int(((clin.gene == g) & (clin.label_source == src)
                              & (clin.label == lv)).sum()) for g in GENES],
                        dtype=float)
        ax.bar(x, vals, bottom=bottoms, color=colour, hatch=hatch, width=0.66,
               edgecolor="white", linewidth=0.7, label=name, zorder=2)
        bottoms += vals
    for xi, tot in zip(x, bottoms):
        ax.text(xi, tot + 4, f"{int(tot)}", ha="center", fontsize=7, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\it{{{g}}}$" for g in GENES])
    ax.set_ylabel("clinical labels", fontsize=7.5)
    ax.set_ylim(0, bottoms.max() * 1.62)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.4, loc="upper left", handlelength=1.5,
              labelcolor=INK_2, borderpad=0.3, labelspacing=0.3)
    ax.set_title(f"(c)  Scored under LOPO\n{len(clin):,} clinical labels",
                 fontsize=8.5, loc="left", color=INK)

    fig.savefig(out / "fig2_composition.png")
    fig.savefig(out / "fig2_composition.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- fig 4
def _reliability(prob: np.ndarray, truth: np.ndarray, n_bins: int = 10):
    """Mean predicted probability and observed frequency per equal-width bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() == 0:
            continue
        xs.append(float(prob[sel].mean()))
        ys.append(float(truth[sel].mean()))
        ns.append(int(sel.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def figure4(out: Path) -> None:
    """Reliability of the headline arm before and after calibration, and its
    confusion matrix at the selected threshold.

    Reads ``calibration_probs.csv`` / ``calibration_panel.csv`` rather than
    recalibrating here, so the picture and Table 5 cannot disagree. PMS2 is
    excluded from the pooled curve for the reason it is excluded from every
    mean in this paper: 21 held-out variants, 4 of them negative.
    """
    probs = pd.read_csv(CALIB_PROBS)
    panel = pd.read_csv(CALIB_PANEL)

    arm_probs = probs[(probs.cell_slug == HEADLINE_CELL)
                      & (probs.holdout_gene.isin(SCOREABLE))]
    arm_panel = panel[(panel.cell_slug == HEADLINE_CELL)
                      & (panel.holdout_gene.isin(SCOREABLE))]
    if arm_probs.empty:
        raise SystemExit(
            f"{CALIB_PROBS.name} carries no rows for {HEADLINE_CELL}; "
            "run scripts/recalibrate_grid.py first.")

    series = [("uncalibrated", BLUE, "o"), ("temperature", ORANGE, "s"),
              ("isotonic", AQUA, "^")]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2), layout="constrained",
                             gridspec_kw={"width_ratios": [1.35, 1.0]})

    # (a) reliability ---------------------------------------------------------
    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), lw=0.9, color=MUTED, zorder=1)
    ax.text(0.62, 0.55, "perfect", fontsize=6.8, color=MUTED, rotation=38,
            style="italic")
    for name, colour, marker in series:
        d = arm_probs[arm_probs.method == name]
        if d.empty:
            continue
        xs, ys, ns = _reliability(d["prob"].to_numpy(), d["label"].to_numpy())
        ax.plot(xs, ys, marker=marker, ms=4.6, lw=1.4, color=colour,
                mec="white", mew=0.8, zorder=3)
        ece = arm_panel.loc[arm_panel.method == name, "ece_uniform"].mean()
        ax.plot([], [], marker=marker, ms=4.6, lw=1.4, color=colour,
                label=f"{name}  (ECE {ece:.3f})")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed fraction pathogenic")
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=7, labelcolor=INK_2)
    ax.set_title("(a)  Reliability, held-out genes pooled",
                 fontsize=8.5, loc="left", color=INK)

    # (b) confusion at the selected threshold ---------------------------------
    ax = axes[1]
    best = "temperature" if (arm_panel.method == "temperature").any() else "uncalibrated"
    row = arm_panel[arm_panel.method == best]
    tn, fp = int(row.tval_tn.sum()), int(row.tval_fp.sum())
    fn, tp = int(row.tval_fn.sum()), int(row.tval_tp.sum())
    cm = np.array([[tn, fp], [fn, tp]], dtype=float)
    row_totals = cm.sum(axis=1, keepdims=True)
    # A row of zeros (no true-benign or no true-pathogenic examples at all in
    # the pooled held-out genes) would otherwise divide 0/0 into NaN and let
    # imshow render an undefined cell silently rather than failing loudly.
    if (row_totals == 0).any():
        raise ValueError(
            "figure4: confusion-matrix row has zero examples "
            f"(row_totals={row_totals.ravel().tolist()}); cannot normalise.")
    rates = cm / row_totals

    ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i, j])}\n{rates[i, j]:.0%}", ha="center",
                    va="center", fontsize=8.5,
                    color="white" if rates[i, j] > 0.55 else INK)
    ax.set_xticks([0, 1], ["called\nbenign", "called\npathogenic"], fontsize=7.5)
    ax.set_yticks([0, 1], ["benign", "pathogenic"], fontsize=7.5)
    ax.set_ylabel("true label", fontsize=7.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"(b)  Confusion, {best}\nrow-normalised, n = {int(cm.sum())}",
                 fontsize=8.5, loc="left", color=INK)

    fig.savefig(out / "fig4_calibration.png")
    fig.savefig(out / "fig4_calibration.pdf")
    plt.close(fig)

def check_provenance(strict: bool) -> None:
    """Refuse to plot cells that were not computed on the same data and splits.

    Pooling incomparable runs into one figure is the same error as pooling them
    into one table, and a figure is the more persuasive of the two. Summaries
    written before ``src/provenance.py`` carry no identity block; those are
    reported and, unless ``strict``, tolerated -- run
    ``scripts/backfill_provenance.py`` on the build machine to close that gap.
    """
    records, missing = {}, []
    for f in sorted(GRID_DIR.glob("esm_finetune_summary_*.json")):
        block = json.loads(f.read_text()).get("provenance")
        slug = f.name[len("esm_finetune_summary_"):-len(".json")]
        (records.setdefault(slug, block) if block else missing.append(slug))
    if missing:
        msg = (f"{len(missing)} of {len(missing) + len(records)} summaries carry "
               f"no provenance block (e.g. {missing[0]}); comparability is "
               f"assumed, not verified. Run scripts/backfill_provenance.py on "
               f"the build machine.")
        if strict:
            raise SystemExit("provenance --strict: " + msg)
        print(f"  WARNING: {msg}")
    if len(records) > 1:
        assert_comparable(records)                     # same table, same splits
        by_arm: dict[str, dict] = {}
        for slug, rec in records.items():
            by_arm.setdefault(arm_of(slug), {})[slug] = rec
        for arm, group in by_arm.items():
            assert_comparable(group, keys=REPLICATE_KEYS)   # seeds of one arm
        print(f"  provenance: {len(records)} cells comparable "
              f"({len(by_arm)} arms)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=Path, default=ROOT / "docs/figures")
    ap.add_argument("--strict_provenance", action="store_true",
                    help="Fail rather than warn when a cell has no identity block.")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    style()
    check_provenance(args.strict_provenance)
    cells, base = load_cells(), load_baseline()
    print(f"{cells.cell_slug.nunique()} cells + baseline; "
          f"{len(cells)} cell-gene rows")
    figure2(args.out_dir)
    figure3(cells, base, args.out_dir)
    figure5(cells, base, args.out_dir)
    figure6(cells, base, args.out_dir)
    if CALIB_PROBS.exists() and CALIB_PANEL.exists():
        figure4(args.out_dir)
    else:
        print("  skipping fig 4: run scripts/recalibrate_grid.py first")
    for f in sorted(args.out_dir.glob("fig*")):
        # --out_dir may be given relative, in which case it is not under ROOT
        # as written; resolve before asking for the repo-relative name.
        try:
            name = f.resolve().relative_to(ROOT)
        except ValueError:
            name = f
        print(f"  {name}  {f.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
