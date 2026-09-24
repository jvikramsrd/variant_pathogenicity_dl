"""Figures for the manuscript (PDF for the paper, PNG for review), from paper/analysis.py frames."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 300,
})
# Okabe-Ito, colour-blind safe.
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]


def _save(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def feature_heatmap(fig_dir: Path, summary: pd.DataFrame, arms, order, models, model_label) -> None:
    arm = {a.key: a for a in arms}
    s = summary.set_index(["arm", "model"])
    grid = np.full((len(order), len(models)), np.nan)
    sd = np.full_like(grid, np.nan)
    for i, key in enumerate(order):
        for j, m in enumerate(models):
            if (key, m) in s.index:
                grid[i, j] = s.loc[(key, m), "auc_mean"]
                sd[i, j] = s.loc[(key, m), "auc_sd"]
    cmap = LinearSegmentedColormap.from_list("auc", ["#b2182b", "#f7f7f7", "#2166ac"])
    cmap.set_bad("#e0e0e0")
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    image = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0.3, vmax=1.0, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isfinite(grid[i, j]):
                colour = "white" if grid[i, j] > 0.9 or grid[i, j] < 0.42 else "black"
                ax.text(j, i, f"{grid[i, j]:.3f}\n±{sd[i, j]:.3f}", ha="center", va="center",
                        fontsize=6.2, color=colour)
            else:
                ax.text(j, i, "no features", ha="center", va="center", fontsize=6, color="#707070")
    ax.set_xticks(range(len(models)), [model_label[m] for m in models])
    ax.set_yticks(range(len(order)), [f"{k}  {arm[k].label}" for k in order])
    ax.axvline(1.5, color="black", lw=0.8)
    ax.axhline(len(order) - 2.5, color="black", lw=0.8, ls=":")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Tabular models  |  sequence-window models (window + the arm's features)")
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.01)
    bar.set_label("AUROC (mean of MLH1, MSH2, MSH6)")
    bar.ax.axhline(0.5, color="black", lw=0.6)
    _save(fig, fig_dir / "fig_features")


def baselines(fig_dir: Path, base: pd.DataFrame, summary: pd.DataFrame, model_label) -> None:
    s = summary.set_index(["arm", "model"])
    rows = [(r.label.replace("$-\\log_{10}$", "−log₁₀")
             + (" (post hoc)" if r.baseline == "rank_mean" else ""), r.auc, r.low, r.high, OKABE[5])
            for r in base.itertuples()]
    for key, m in (("M", "gbm"), ("M", "mlp"), ("A5", "gbm"), ("A5", "mlp")):
        r = s.loc[(key, m)]
        rows.append((f"{model_label[m]} trained, arm {key}", r.auc_mean, r.auc_ci_low, r.auc_ci_high, OKABE[0]))
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    for i, (label, value, low, high, colour) in enumerate(rows):
        ax.errorbar(value, i, xerr=[[value - low], [high - value]], fmt="o", color=colour, ms=4, capsize=2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("AUROC, mean of MLH1, MSH2, MSH6 (95% bootstrap CI)")
    ax.axvline(0.5, color="grey", lw=0.6, ls=":")
    ax.set_xlim(0.75, 1.0)
    _save(fig, fig_dir / "fig_baselines")


def label_forest(fig_dir: Path, comparisons: pd.DataFrame, models, model_label, word) -> None:
    c = comparisons.set_index("comparison")
    arms = [("BThree", "B3 expert panel only", OKABE[2]), ("BOne", "B1 MSH2 DMS only", OKABE[1]),
            ("BTwo", "B2 ClinVar + MSH2 DMS", OKABE[3])]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), sharey=True)
    for ax, metric, title in zip(axes, ("auc", "mcc"), ("ΔAUROC vs A5 (ClinVar, all features)",
                                                        "ΔMCC vs A5 (ClinVar, all features)")):
        for offset, (key, label, colour) in zip((-0.22, 0.0, 0.22), arms):
            for i, m in enumerate(models):
                r = c.loc[f"{key}VsAFive{word(m)}"]
                ax.errorbar(r[f"{metric}_delta"], i + offset,
                            xerr=[[r[f"{metric}_delta"] - r[f"{metric}_low"]],
                                  [r[f"{metric}_high"] - r[f"{metric}_delta"]]],
                            fmt="o", color=colour, ms=3.2, capsize=1.5, lw=0.9,
                            label=label if i == 0 else None)
        ax.axvline(0, color="black", lw=0.7)
        ax.set_title(title)
        ax.set_yticks(range(len(models)), [model_label[m] for m in models])
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.09), ncol=3, frameon=False)
    fig.text(0.5, -0.14, "B3 scored on MLH1, MSH2, MSH6.  B1 and B2 scored on MLH1 and MSH6: with MSH2 held out "
             "no DMS label can train, so MSH2 is identical to A5 by construction.  95% paired bootstrap CIs.",
             ha="center", fontsize=6.5)
    _save(fig, fig_dir / "fig_labels")


def circularity(fig_dir: Path, table: pd.DataFrame, order, scoreable, obs: pd.DataFrame, arms,
                model_label) -> None:
    arm = {a.key: a for a in arms}
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.8, 2.9), gridspec_kw={"width_ratios": [1.05, 1.25]})
    rng = np.random.default_rng(0)
    for i, gene in enumerate(scoreable):
        rows = table.loc[order[gene]]
        for label, colour, shift in ((1, OKABE[3], -0.17), (0, OKABE[0], 0.17)):
            v = rows.loc[rows.label__clinvar == label, "feature_gnomad_log10_af"].to_numpy()
            x = i + shift + rng.uniform(-0.1, 0.1, len(v))
            left.scatter(x, v, s=4, color=colour, alpha=0.6, lw=0,
                         label=("pathogenic / likely pathogenic" if label else "benign / likely benign")
                         if i == 0 else None)
    left.axhline(-3, color="grey", lw=0.6, ls="--")
    left.text(len(scoreable) - 0.5, -2.9, "BS1 (0.1%)", fontsize=6, ha="right", va="bottom", color="grey")
    left.set_xticks(range(len(scoreable)), scoreable)
    left.set_ylabel("log₁₀ gnomAD allele frequency\n(−7 = not observed)")
    left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, ncol=1, markerscale=2.5)
    left.set_title("a  Frequency by ClinVar label", loc="left")

    names = {"neg_log_af": "−log₁₀ AF (no training)", "alphamissense": "AlphaMissense (no training)",
             "esm1b": "ESM-1b (no training)"}
    labels = [names[r.model] if r.condition == "baseline" else f"{model_label[r.model]}, {r.condition}"
              for r in obs.itertuples()]
    y = np.arange(len(obs))
    for i, r in enumerate(obs.itertuples()):
        right.plot([r.auc_all, r.auc_observed], [i, i], color="#b0b0b0", lw=1, zorder=1)
    right.errorbar(obs.auc_all, y, xerr=[obs.auc_all - obs.all_low, obs.all_high - obs.auc_all], fmt="o",
                   color=OKABE[0], ms=3.5, capsize=1.5, lw=0.8, label="all held-out variants")
    right.errorbar(obs.auc_observed, y, xerr=[obs.auc_observed - obs.observed_low,
                                              obs.observed_high - obs.auc_observed],
                   fmt="s", color=OKABE[1], ms=3.5, capsize=1.5, lw=0.8, label="observed in gnomAD only")
    right.axvline(0.5, color="black", lw=0.6, ls=":")
    right.set_yticks(y, labels)
    right.invert_yaxis()
    right.set_xlabel("AUROC, mean of MLH1, MSH2, MSH6")
    right.legend(loc="upper center", bbox_to_anchor=(0.45, -0.18), frameon=False, ncol=2)
    right.set_title("b  All held-out variants vs. gnomAD-observed only", loc="left")
    fig.tight_layout()
    _save(fig, fig_dir / "fig_circularity")


def per_gene(fig_dir: Path, per_gene_frame: pd.DataFrame, models, model_label) -> None:
    g = per_gene_frame.groupby(["arm", "model", "gene"]).auc.agg(["mean", "std"]).reset_index()
    genes = ["MLH1", "MSH2", "MSH6", "PMS2"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True)
    for ax, key, title in zip(axes, ("M", "A5"), ("M: gnomAD + AlphaMissense", "A5: all feature sources")):
        width = 0.8 / len(models)
        for j, m in enumerate(models):
            block = g[(g.arm == key) & (g.model == m)].set_index("gene").reindex(genes)
            x = np.arange(len(genes)) - 0.4 + width * (j + 0.5)
            ax.bar(x, block["mean"], width, yerr=block["std"], color=OKABE[j % len(OKABE)],
                   label=model_label[m], error_kw={"lw": 0.6, "capsize": 1})
        ax.set_xticks(range(len(genes)), [f"{x}{'*' if x == 'PMS2' else ''}" for x in genes])
        ax.set_ylim(0.5, 1.0)
        ax.set_title(title)
    axes[0].set_ylabel("AUROC (mean ± SD, 3 seeds)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    fig.text(0.5, -0.03, "*PMS2: n = 21 (4 benign); not scoreable, never in a headline mean.", ha="center",
             fontsize=6.5)
    _save(fig, fig_dir / "fig_per_gene")


def design(fig_dir: Path) -> None:
    """Study design, drawn (no data)."""
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_xlim(0, 104)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, text, colour, size=5.7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=colour, ec="#404040", lw=0.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#404040"))

    source, label, model, evaluation = "#e8f1fa", "#fdf0e0", "#eaf5ee", "#f3e8f5"
    box(1, 36, 22, 8, "Every missense substitution\nin MLH1, MSH2, MSH6, PMS2", source)
    box(1, 19, 22, 13, "Features\ngnomAD v4 frequency\nAlphaMissense\nAlphaFold structure\ngenomic position",
        source)
    box(1, 2, 22, 13, "Labels\nClinVar ≥2 stars\n(expert-panel subset)\nProteinGym DMS\n(one MSH2 assay)",
        label)
    box(30, 25, 30, 19, "Axis A: feature sources\n(ClinVar labels)\nA0 sequence window only\n"
        "A1 gnomAD   A2 AlphaMissense\nA3 structure   A4 genomic\nM gnomAD + AlphaMissense\nA5 all sources",
        model)
    box(30, 2, 30, 18, "Axis B: label sources\n(all features)\nA5 ClinVar\nB3 ClinVar expert panel only\n"
        "B1 MSH2 DMS only\nB2 ClinVar + MSH2 DMS", label)
    box(66, 12, 17, 22, "7 models × 3 seeds\n\nGBM, MLP,\nwindow MLP, CNN,\nBiLSTM,\nBiLSTM + attention,\n"
        "Transformer", model)
    box(88, 5, 15, 35, "Leave one\ngene out\n\nscored on the\nheld-out gene's\nClinVar labels\n\n"
        "threshold from\ninner validation\n(training genes)\n\nAUROC, MCC,\npaired bootstrap", evaluation)
    arrow(12, 35.2, 12, 32.8)
    arrow(23.8, 27, 29.2, 33)
    arrow(23.8, 23, 29.2, 14)
    arrow(23.8, 11, 29.2, 29)
    arrow(23.8, 8, 29.2, 9)
    arrow(60.8, 34, 65.2, 26)
    arrow(60.8, 11, 65.2, 19)
    arrow(83.8, 23, 87.2, 23)
    _save(fig, fig_dir / "fig_design")


def write_all(fig_dir: Path, summary, per_gene_frame, base, comparisons, table, order, scoreable, dms_genes,
              obs, arm_map, feature_order, models, model_label, word) -> None:
    arms = list(arm_map.values())
    design(fig_dir)
    feature_heatmap(fig_dir, summary, arms, feature_order, models, model_label)
    baselines(fig_dir, base, summary, model_label)
    label_forest(fig_dir, comparisons, models, model_label, word)
    circularity(fig_dir, table, order, scoreable, obs, arms, model_label)
    per_gene(fig_dir, per_gene_frame, models, model_label)
