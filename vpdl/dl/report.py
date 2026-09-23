"""Collect every DL run into paper-ready tables, with the checks that make them
citable.

`vpdl-dl results --runs runs/dl --out results/dl` walks the run directories and
writes:

    cells.csv          one row per (cell, held-out gene): every metric with its
                       bootstrap CI, plus the provenance keys (dataset sha256,
                       split hash, feature schema, feature version, git commit)
    arms.csv           mean +/- SD over seeds per (arm, model, gene), and the
                       mean over scoreable genes
    pretraining.csv    one row per pretraining arm/fold (val MLM loss, perplexity)
    calibration.csv    per-gene ECE / Brier / slope, raw and calibrated
    functional.csv     independent functional-assay validation
    paired_*.csv       paired Delta-AUC against a named reference arm, with CIs
    tables.md/.tex     the same numbers formatted for a manuscript
    checks.json        comparability, seed counts, dirty code, skipped genes,
                       leakage criticals — read this before citing anything
    environment.json   library versions, platform, hardware, git commits seen
    MANIFEST.json      sha256 of every file above, written last

Two rules from this project's own history are enforced rather than trusted:
runs built on different tables are never pooled (`assert_comparable`, landmine
L13), and a table that pools fewer than three seeds says so in `checks.json`
(observed MCC SD up to 0.056 on this panel).

Nothing here recomputes a metric. Every number is read from the artefacts the
runs wrote, so a paper table and a run directory cannot drift apart.
"""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from vpdl.analysis import SCOREABLE_MIN_N, parse_cell
from vpdl.provenance import assert_comparable, file_sha256, write_manifest

logger = logging.getLogger(__name__)

__all__ = ["METRICS", "collect_cells", "arm_table", "collect_pretraining",
           "collect_side_tables", "paired_against", "auto_references",
           "comparability_checks", "environment", "to_latex", "forest_plot",
           "write_paper_bundle"]

METRICS = ("roc_auc", "pr_auc", "mcc", "f1", "sensitivity", "specificity", "brier", "ece")
_CI_METRICS = ("roc_auc", "pr_auc", "mcc")
_SUPERSEDED = "superseded"


def _ci(value: Any) -> tuple[float, float]:
    """``"(0.76, 0.85)"`` (as CSV round-trips a tuple) -> floats."""
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return float(value[0]), float(value[1])
    try:
        low, high = ast.literal_eval(str(value))
        return float(low), float(high)
    except (ValueError, SyntaxError, TypeError):
        return float("nan"), float("nan")


def collect_cells(runs_root: Path | str) -> pd.DataFrame:
    """One row per (cell, gene) from every ``summary_*.json`` + ``results_*.csv``.

    Recursive, so one call covers `runs/dl/main`, `runs/dl/probes`, ... A
    ``superseded/`` directory (older artefacts an overwrite moved aside) is
    skipped.
    """
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(Path(runs_root).rglob("summary_*.json")):
        if _SUPERSEDED in summary_path.parts:
            continue
        summary = json.loads(summary_path.read_text())
        slug = summary["cell"]
        arm, model, seed = parse_cell(slug)
        provenance = summary.get("provenance", {})
        git = provenance.get("git") or {}
        results_path = summary_path.with_name(f"results_{slug}.csv")
        if not results_path.exists():
            logger.warning("%s has no results CSV; skipped", slug)
            continue
        results = pd.read_csv(results_path)
        common = {
            "runs_dir": str(summary_path.parent), "cell": slug, "arm": arm, "model": model,
            "seed": seed, "split": summary.get("split", "logo"),
            "train_sources": "+".join(summary.get("train_sources", [])),
            "eval_source": summary.get("eval_source"),
            "drop_groups": "+".join(summary.get("drop_groups", [])),
            "embedding_blocks": ";".join(summary.get("embedding_blocks", [])),
            "tag": summary.get("tag", ""),
            "genes_skipped": ";".join(summary.get("genes_skipped", {})),
            "runtime_s": summary.get("runtime_s"),
            "dataset_sha256": provenance.get("dataset_sha256"),
            "split_hash": provenance.get("split_hash"),
            "feature_schema": provenance.get("feature_schema"),
            "feature_version": provenance.get("feature_version"),
            "n_features": len(provenance.get("sources", [])) and None,
            "git_commit": git.get("commit"), "git_dirty": git.get("dirty"),
            # Which files were uncommitted when the cell ran. With two branches
            # in one repository, the answer is usually "the other branch's" —
            # the report says so rather than leaving a bare flag.
            "git_dirty_paths": ";".join(git.get("dirty_paths") or []),
            "leakage_warnings": (provenance.get("leakage") or {}).get("warnings"),
        }
        for row in results.to_dict("records"):
            entry = dict(common)
            entry.update({k: row.get(k) for k in ("gene", "fold", "n", "n_positive",
                                                  "threshold", *METRICS)})
            for metric in _CI_METRICS:
                low, high = _ci(row.get(f"{metric}_ci"))
                entry[f"{metric}_ci_low"], entry[f"{metric}_ci_high"] = low, high
            entry["scoreable"] = bool(row.get("n", 0) >= SCOREABLE_MIN_N)
            rows.append(entry)
    if not rows:
        raise ValueError(f"no summary_*.json under {runs_root}")
    return pd.DataFrame(rows)


def arm_table(cells: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- SD over seeds per (runs_dir, arm, model, split, gene).

    Adds a ``mean:scoreable`` gene row per arm: each seed's mean over genes with
    n >= 50 first, then mean and SD across seeds — the order that keeps the SD a
    seed spread rather than a gene spread.
    """
    keys = ["runs_dir", "arm", "model", "split", "tag", "embedding_blocks"]
    frames = []
    for name, group in cells.groupby(keys, dropna=False):
        per_gene = group.groupby("gene", dropna=False)
        summary = per_gene.agg(
            n=("n", "first"), n_positive=("n_positive", "first"),
            seeds=("seed", "nunique"), scoreable=("scoreable", "first"),
            **{f"{m}_{stat}": (m, stat) for m in METRICS for stat in ("mean", "std")},
        ).reset_index()
        scoreable = group[group["scoreable"]]
        if len(scoreable):
            per_seed = scoreable.groupby("seed")[list(METRICS)].mean()
            overall = {"gene": "mean:scoreable", "n": int(scoreable.groupby("seed")["n"].sum()
                                                          .mean()),
                       "n_positive": int(scoreable.groupby("seed")["n_positive"].sum().mean()),
                       "seeds": scoreable["seed"].nunique(), "scoreable": True}
            for metric in METRICS:
                overall[f"{metric}_mean"] = per_seed[metric].mean()
                overall[f"{metric}_std"] = per_seed[metric].std(ddof=1) if len(per_seed) > 1 \
                    else np.nan
            summary = pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)
        for key, value in zip(keys, name if isinstance(name, tuple) else (name,)):
            summary[key] = value
        frames.append(summary)
    table = pd.concat(frames, ignore_index=True)
    return table[keys + [c for c in table.columns if c not in keys]]


def collect_pretraining(runs_root: Path | str) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(runs_root).rglob("pretrain_summary.json")):
        payload = json.loads(path.read_text())
        corpus = payload.get("corpus") or {}
        fit = payload.get("fit") or {}
        rows.append({
            "dir": str(path.parent), "arm": payload.get("arm"),
            "backbone": (payload.get("config") or {}).get("backbone"),
            "corpus_mode": corpus.get("mode"), "holdout": corpus.get("holdout"),
            "corpus_train": corpus.get("n_train"), "corpus_val": corpus.get("n_val"),
            "excluded_homologs": corpus.get("n_excluded_homologs"),
            "objectives": json.dumps({k: v for k, v in (payload.get("objectives") or {}).items()
                                      if isinstance(v, (int, float)) and v}),
            "trainable_params": (payload.get("peft") or {}).get("trainable_params"),
            "epochs_run": fit.get("epochs_run"), "best_epoch": fit.get("best_epoch"),
            "val_mlm_loss": payload.get("best_val_mlm_loss"),
            "val_perplexity": payload.get("val_mlm_perplexity"),
            "note": payload.get("note"),
        })
    return pd.DataFrame(rows)


def collect_side_tables(runs_root: Path | str) -> dict[str, pd.DataFrame]:
    """Calibration, functional-validation and `vpdl paired` CSVs already written."""
    out: dict[str, list[pd.DataFrame]] = {"calibration": [], "functional": [], "paired_runs": []}
    patterns = {"calibration": "calibration_*.csv", "functional": "functional_validation.csv",
                "paired_runs": "paired_vs_*.csv"}
    for name, pattern in patterns.items():
        for path in sorted(Path(runs_root).rglob(pattern)):
            if _SUPERSEDED in path.parts:
                continue
            frame = pd.read_csv(path)
            if len(frame):
                out[name].append(frame.assign(source=str(path)))
    return {name: (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())
            for name, frames in out.items()}


def paired_against(reference: tuple[str, str, str], target_dirs: Sequence[str],
                   cells: pd.DataFrame, n_bootstrap: int = 10_000) -> pd.DataFrame:
    """Paired Delta-AUC of every arm in `target_dirs` against one reference arm.

    `reference` is ``(runs_dir, arm, model)``. Refuses to pool directories whose
    cells were built on different tables or split the data differently
    (landmine L13) — the comparison would be between two test sets.
    """
    from vpdl.analysis import load_predictions, paired_table

    reference_dir, reference_arm, reference_model = reference
    dirs = list(dict.fromkeys([reference_dir, *target_dirs]))
    involved = cells[cells["runs_dir"].isin(dirs)]
    assert_comparable([{"dataset_sha256": sha, "split_hash": split}
                       for sha, split in involved[["dataset_sha256", "split_hash"]]
                       .drop_duplicates().itertuples(index=False)])
    frames = [load_predictions(d) for d in dirs]
    predictions = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["cell", "variant_key"])
    return paired_table(predictions, (reference_arm, reference_model), n_bootstrap=n_bootstrap)


def auto_references(cells: pd.DataFrame) -> list[tuple[tuple[str, str, str], list[str]]]:
    """Reference arms a paper needs, inferred from what was actually run.

    1. the existing-model baseline (the published six-feature MLP arm), compared
       against every other directory;
    2. within any directory holding both a P0 arm and per-fold pretrained arms,
       the P0 arm (so P1-P4 are measured against the original checkpoint).
    """
    comparisons: list[tuple[tuple[str, str, str], list[str]]] = []
    baseline = cells[(cells["model"] == "mlp") & (cells["embedding_blocks"] == "")
                     & (cells["split"] == "logo")
                     & cells["arm"].str.startswith("train-clinvar__drop-")]
    if len(baseline):
        row = baseline.sort_values("arm").iloc[0]
        others = [d for d in cells["runs_dir"].unique()]
        comparisons.append(((row["runs_dir"], row["arm"], row["model"]), others))
    for runs_dir, group in cells.groupby("runs_dir"):
        p0 = group[group["embedding_blocks"].str.contains("@P0", na=False)
                   & ~group["embedding_blocks"].str.contains("perfold", na=False)]
        perfold = group[group["embedding_blocks"].str.contains("perfold", na=False)]
        if len(p0) and len(perfold):
            arms = p0[["arm", "model"]].drop_duplicates()
            if len(arms) == 1:
                comparisons.append(((runs_dir, arms.iloc[0]["arm"], arms.iloc[0]["model"]),
                                    [runs_dir]))
            else:
                logger.warning("%s: %d candidate P0 arms; pass --compare explicitly",
                               runs_dir, len(arms))
    return comparisons


def comparability_checks(cells: pd.DataFrame, leakage_dir: Path | str | None = None
                         ) -> dict[str, Any]:
    """Everything that would make a table misleading, collected in one place."""
    problems: list[str] = []
    notes: list[str] = []

    tables = cells["dataset_sha256"].dropna().unique()
    if len(tables) > 1:
        counts = cells.groupby("dataset_sha256")["cell"].nunique().to_dict()
        problems.append(f"cells come from {len(tables)} different tables: {counts}. "
                        "Arms on different tables must not be pooled or compared.")
    splits = cells.groupby("split")["split_hash"].nunique().to_dict()
    for split, distinct in splits.items():
        if distinct > 1:
            problems.append(f"split '{split}' has {distinct} different split hashes")

    for (arm, model), group in cells.groupby(["arm", "model"]):
        seeds = sorted(int(v) for v in group["seed"].unique())
        if len(seeds) < 3:
            problems.append(f"{arm}:{model} has {len(seeds)} seed(s) {seeds}; three is this "
                            "project's floor for reading a difference")
        if group["feature_schema"].nunique(dropna=True) > 1:
            problems.append(f"{arm}:{model} seeds disagree on the feature schema — "
                            "they are not replicates")
    dirty = cells[cells["git_dirty"] == True]                                  # noqa: E712
    if len(dirty):
        paths = sorted({p for row in dirty.get("git_dirty_paths", pd.Series(dtype=str))
                        .dropna() for p in str(row).split(";") if p})
        dl_paths = [p for p in paths if p.startswith(("vpdl/dl", "vpdl/models", "vpdl/sources",
                                                      "vpdl/experiment", "vpdl/features",
                                                      "vpdl/evaluate", "vpdl/analysis",
                                                      "vpdl/splits", "vpdl/assemble",
                                                      "vpdl/provenance"))]
        where = (f"DL code was uncommitted: {dl_paths[:5]}" if dl_paths else
                 f"only non-DL files were uncommitted ({paths[:5]}), so the DL code that ran "
                 "IS the committed code")
        (problems if dl_paths else notes).append(
            f"{dirty['cell'].nunique()} cell(s) ran with an unclean working tree — {where}")
    skipped = cells[cells["genes_skipped"].astype(str).str.len() > 0]
    if len(skipped):
        notes.append(f"{skipped['cell'].nunique()} cell(s) skipped a fold (untrainable): "
                     f"{sorted(skipped['genes_skipped'].unique())}")
    small = cells[~cells["scoreable"]]
    if len(small):
        notes.append(f"{small['gene'].nunique()} gene(s) below n={SCOREABLE_MIN_N} are reported "
                     f"but excluded from headline means: {sorted(small['gene'].unique())}")

    leakage: dict[str, Any] = {}
    if leakage_dir and Path(leakage_dir).exists():
        for path in sorted(Path(leakage_dir).glob("leakage_*.json")):
            payload = json.loads(path.read_text())
            leakage[payload.get("scheme", path.stem)] = {
                "critical": payload.get("n_critical", 0),
                "warnings": sum(f.get("severity") == "warning"
                                for f in payload.get("findings", []))}
            if payload.get("n_critical"):
                problems.append(f"leakage report {path.name} has "
                                f"{payload['n_critical']} critical finding(s)")
    return {"problems": problems, "notes": notes, "leakage": leakage,
            "n_cells": int(cells["cell"].nunique()),
            "n_arms": int(cells.groupby(["arm", "model"]).ngroups),
            "dataset_sha256": sorted(tables.tolist()),
            "citable": not problems}


def environment(cells: pd.DataFrame, runs_root: Path | str) -> dict[str, Any]:
    versions: dict[str, Any] = {}
    platforms: list[Any] = []
    for path in sorted(Path(runs_root).rglob("summary_*.json")):
        if _SUPERSEDED in path.parts:
            continue
        provenance = json.loads(path.read_text()).get("provenance", {})
        versions.update(provenance.get("versions") or {})
        if provenance.get("platform") and provenance["platform"] not in platforms:
            platforms.append(provenance["platform"])
    hardware_path = Path(runs_root) / "hardware.json"
    registry = Path(runs_root) / "registry.jsonl"
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commits": sorted(c for c in cells["git_commit"].dropna().unique()),
        "library_versions": versions, "platforms": platforms,
        "hardware": json.loads(hardware_path.read_text()) if hardware_path.exists() else None,
        "registry_runs": sum(1 for _ in open(registry)) if registry.exists() else 0,
    }


def to_latex(frame: pd.DataFrame, caption: str, label: str, float_format: str = "%.3f") -> str:
    """booktabs table, no jinja2 / Styler dependency."""
    columns = list(frame.columns)

    def cell(value: Any) -> str:
        if isinstance(value, float):
            return "--" if not np.isfinite(value) else float_format % value
        return (str(value).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
                .replace("#", r"\#"))

    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabular}{" + "l" * len(columns) + "}", r"\toprule",
             " & ".join(cell(c) for c in columns) + r" \\", r"\midrule"]
    lines += [" & ".join(cell(v) for v in row) + r" \\"
              for row in frame.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)


def _markdown(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:                            # tabulate absent
        return "```\n" + frame.to_string(index=False) + "\n```"


def _headline(arms: pd.DataFrame) -> pd.DataFrame:
    """The table a paper leads with: one row per arm, mean +/- SD over seeds."""
    rows = arms[arms["gene"] == "mean:scoreable"].copy()
    out = pd.DataFrame({
        "arm": rows["arm"], "model": rows["model"], "split": rows["split"],
        "seeds": rows["seeds"], "n": rows["n"],
    })
    for metric in ("roc_auc", "mcc", "pr_auc", "brier", "ece"):
        out[metric] = [f"{m:.3f} ± {s:.3f}" if np.isfinite(s) else f"{m:.3f}"
                       for m, s in zip(rows[f"{metric}_mean"], rows[f"{metric}_std"])]
    return out.sort_values(["split", "arm", "model"]).reset_index(drop=True)


def forest_plot(paired: pd.DataFrame, path: Path | str, title: str) -> Path | None:
    """Delta-AUC with CIs, one row per arm (mean rows only). Needs matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    rows = paired[paired["gene"].astype(str).str.startswith("mean:")]
    if rows.empty:
        rows = paired
    rows = rows.reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(7, 0.4 * len(rows) + 1.6))
    y = np.arange(len(rows))
    axis.errorbar(rows["delta"], y,
                  xerr=[rows["delta"] - rows["ci_low"], rows["ci_high"] - rows["delta"]],
                  fmt="o", capsize=3, linewidth=1)
    axis.axvline(0, color="grey", linestyle="--", linewidth=1)
    axis.set_yticks(y)
    axis.set_yticklabels([f"{a}:{m}" for a, m in zip(rows["arm"], rows["model"])], fontsize=7)
    axis.set_xlabel("paired ΔAUC vs reference (95% CI)")
    axis.set_title(title, fontsize=9)
    figure.tight_layout()
    path = Path(path)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


@dataclass
class PaperBundle:
    out_dir: Path
    files: dict[str, Path] = field(default_factory=dict)
    checks: dict[str, Any] = field(default_factory=dict)


def write_paper_bundle(runs_root: Path | str, out_dir: Path | str,
                       comparisons: Iterable[tuple[tuple[str, str, str], Sequence[str]]] | None = None,
                       canonical_report: Path | str | None = None,
                       leakage_dir: Path | str | None = None,
                       n_bootstrap: int = 10_000, plots: bool = True) -> PaperBundle:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = PaperBundle(out_dir)

    cells = collect_cells(runs_root)
    arms = arm_table(cells)
    bundle.files["cells"] = out_dir / "cells.csv"
    bundle.files["arms"] = out_dir / "arms.csv"
    cells.to_csv(bundle.files["cells"], index=False)
    arms.to_csv(bundle.files["arms"], index=False)

    pretraining = collect_pretraining(runs_root)
    if len(pretraining):
        bundle.files["pretraining"] = out_dir / "pretraining.csv"
        pretraining.to_csv(bundle.files["pretraining"], index=False)
    for name, frame in collect_side_tables(runs_root).items():
        if len(frame):
            bundle.files[name] = out_dir / f"{name}.csv"
            frame.to_csv(bundle.files[name], index=False)

    paired_tables: dict[str, pd.DataFrame] = {}
    for reference, targets in (comparisons if comparisons is not None
                               else auto_references(cells)):
        runs_dir, arm, model = reference
        name = f"paired_{Path(runs_dir).name}_{arm}_{model}".replace(":", "-")[:120]
        if targets is None:
            targets = list(cells["runs_dir"].unique())
        try:
            table = paired_against(reference, targets, cells, n_bootstrap)
        except (ValueError, KeyError) as error:
            logger.warning("paired comparison against %s:%s skipped — %s", arm, model, error)
            continue
        if table.empty:
            continue
        paired_tables[name] = table
        bundle.files[name] = out_dir / f"{name}.csv"
        table.to_csv(bundle.files[name], index=False)
        if plots:
            plot = forest_plot(table, out_dir / f"{name}.png", f"vs {arm}:{model}")
            if plot:
                bundle.files[f"{name}_plot"] = plot

    headline = _headline(arms)
    sections = [f"# DL results — generated {datetime.now(timezone.utc).date()}", "",
                "Generated by `vpdl-dl results`; do not edit. Read `checks.json` before "
                "citing any number.", ""]
    latex = []
    if canonical_report and Path(canonical_report).exists():
        report = json.loads(Path(canonical_report).read_text())
        composition = pd.DataFrame([
            {"gene": gene, "rows": report["rows_per_gene"].get(gene),
             "pathogenic": counts.get("pathogenic"), "benign": counts.get("benign")}
            for gene, counts in (report.get("clinical_labels") or {}).items()])
        sections += ["## Table 1 — dataset composition", "", _markdown(composition), "",
                     f"QC flags: `{json.dumps(report.get('qc_flag_counts', {}))}`", "",
                     f"PMS2 homology gate: `{json.dumps(report.get('pms2', {}))}`", ""]
        latex.append(to_latex(composition, "Canonical dataset composition.", "tab:composition"))
    sections += ["## Table 2 — arms (mean over scoreable genes, ± SD over seeds)", "",
                 _markdown(headline), ""]
    latex.append(to_latex(headline, "Arms, mean over scoreable held-out genes.", "tab:arms"))
    per_gene = arms[arms["gene"] != "mean:scoreable"][
        ["arm", "model", "split", "gene", "n", "n_positive", "seeds", "roc_auc_mean",
         "roc_auc_std", "mcc_mean", "mcc_std"]]
    sections += ["## Table 3 — per held-out gene", "", _markdown(per_gene.round(4)), ""]
    latex.append(to_latex(per_gene.round(4), "Per held-out gene.", "tab:pergene"))
    for name, table in paired_tables.items():
        sections += [f"## Table 4 — {name}", "", _markdown(table), ""]
        latex.append(to_latex(table, f"Paired difference: {name}.", f"tab:{name[:40]}"))

    bundle.files["tables_md"] = out_dir / "tables.md"
    bundle.files["tables_tex"] = out_dir / "tables.tex"
    bundle.files["tables_md"].write_text("\n".join(sections), encoding="utf-8")
    bundle.files["tables_tex"].write_text("\n\n".join(latex), encoding="utf-8")

    bundle.checks = comparability_checks(cells, leakage_dir)
    bundle.files["checks"] = out_dir / "checks.json"
    bundle.files["checks"].write_text(json.dumps(bundle.checks, indent=2, default=str))
    bundle.files["environment"] = out_dir / "environment.json"
    bundle.files["environment"].write_text(json.dumps(environment(cells, runs_root), indent=2,
                                                      default=str))
    # Manifest last, so its checksums describe the files on disk (landmine L10).
    manifest = out_dir / "MANIFEST.json"
    write_manifest(manifest, artefacts=list(bundle.files.values()),
                   meta={"kind": "dl-results", "runs_root": str(runs_root),
                         "citable": bundle.checks["citable"],
                         "n_cells": bundle.checks["n_cells"]})
    bundle.files["manifest"] = manifest
    return bundle
