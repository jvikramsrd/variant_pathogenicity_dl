"""Generate Figures 3, 5 and 6 of the manuscript from committed run artifacts.

Every number plotted here is read from a results CSV in ``data/processed/``;
nothing is recomputed, so a figure cannot drift from the table it illustrates.
Figures 1 and 2 are schematics/composition plots and are not produced here.

    python scripts/make_figures.py [--out_dir docs/figures]

Palette: three categorical hues validated for colour-vision deficiency at all
pairs (worst CVD dE 9.2, normal-vision dE 24.0). The aqua slot sits below 3:1
against the surface, so every series is also direct-labelled and given its own
marker shape -- identity is never carried by hue alone, in print or on screen.
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GRID_DIR = ROOT / "data/processed/stage2b_grid"
BASELINE_CSV = ROOT / "data/processed/mmr_transfer_scratch/mmr_transfer_results_lopo.csv"

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
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
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

    fig, ax = plt.subplots(figsize=(6.6, 3.5))
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
    ax.set_title("Held-out-gene discrimination: curated features versus ESM-2",
                 color=INK, loc="left", pad=10)
    ax.text(1.012, 3 + 0.42, "not scoreable: 21 variants, 4 negatives",
            ha="right", va="center", fontsize=7, color=INK_2, style="italic")
    handles, _ = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker="o", ls="none", mfc="white",
                              mec=MUTED, mew=1.0, ms=4,
                              label="seeds 43, 44 (point only)"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, -0.32),
              ncol=4, handletextpad=0.4, columnspacing=1.4, labelcolor=INK_2)
    fig.savefig(out / "fig3_main.png")
    fig.savefig(out / "fig3_main.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- fig 5
def figure5(cells: pd.DataFrame, base: pd.DataFrame, out: Path) -> None:
    """All arms, seed-averaged, ranked against the curated-features baseline."""
    g = by_arm(cells)
    fam = cells.assign(arm=cells.cell_slug.map(arm_of)).groupby("arm").family.first()
    baseline = mean3(base).iloc[0]

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.set_axisbelow(True)
    ax.xaxis.grid(True)

    ax.axvline(baseline, color=BLUE, lw=1.4, zorder=1)
    ax.text(baseline + 0.002, len(g) + 0.15,
            f"curated priors only, no ESM · {baseline:.3f}",
            ha="left", va="center", fontsize=7.5, color=BLUE)

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
    ax.set_title("Every arm, seed-averaged, against the curated-features baseline",
                 color=INK, loc="left", pad=10)
    handles = [plt.Line2D([], [], marker="o", ls="none", color=ORANGE, mec="white",
                          ms=6, label="feature-family ablation"),
               plt.Line2D([], [], marker="s", ls="none", color=MUTED, mec="white",
                          ms=4.5, label="grid cell"),
               plt.Line2D([], [], color=MUTED, lw=2, label="± 1 SD over seeds")]
    ax.legend(handles=handles, loc="lower right", labelcolor=INK_2,
              bbox_to_anchor=(1.0, -0.02))
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

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
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
    ax.set_ylim(0.785, 1.02)
    ax.set_ylabel("AUROC on the held-out gene")
    ax.set_title("Per-gene performance across all arms", color=INK, loc="left",
                 pad=10)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.34), ncol=3,
              handletextpad=0.4, columnspacing=1.6, labelcolor=INK_2)
    fig.savefig(out / "fig6_pergene.png")
    fig.savefig(out / "fig6_pergene.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=Path, default=ROOT / "docs/figures")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    style()
    cells, base = load_cells(), load_baseline()
    print(f"{cells.cell_slug.nunique()} cells + baseline; "
          f"{len(cells)} cell-gene rows")
    figure3(cells, base, args.out_dir)
    figure5(cells, base, args.out_dir)
    figure6(cells, base, args.out_dir)
    for f in sorted(args.out_dir.glob("fig*")):
        print(f"  {f.relative_to(ROOT)}  {f.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
