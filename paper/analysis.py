"""Every number, table and figure in the manuscript, generated from the saved runs.

    python paper/analysis.py            # writes paper/generated/ (about a minute)

Inputs are the DGX ablation grid (``runs/dl/*/predictions_*.csv``), its summary
(``results/dl``) and the table it trained on (``data/built/canonical_full.csv``).
Nothing is typed by hand: the manuscript reads ``generated/numbers.tex`` (one macro
per quoted number) and ``generated/tab_*.tex``.

The script refuses to write anything if the inputs are not the ones the grid used:

* ``results/dl/checks.json`` must say ``citable: true`` and name one dataset hash;
* ``data/built/canonical_full.csv`` must hash to that value;
* every metric recomputed here from the per-variant predictions must equal the value
  the DGX wrote into ``results/dl/cells.csv`` (AUROC, MCC; to 1e-9);
* every arm must have scored the identical variants, with identical labels, per gene.

Statistics. Per gene, a cell's metric is averaged over its three seeds; the headline
is the mean over the scoreable genes (n >= 50: MLH1, MSH2, MSH6). A difference
between two arms is PAIRED: each bootstrap resample draws variants within each gene
once and scores both arms on it (10,000 resamples, percentile interval). A gene on
which two arms made identical predictions — MSH2 whenever the only extra training
source is the MSH2 DMS assay, which leaves training when MSH2 is held out — is a
design constant, not a measurement, and is left out of that comparison and named.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vpdl.analysis import parse_cell  # noqa: E402
from vpdl.dl.calibration import brier_score, expected_calibration_error  # noqa: E402
from vpdl.evaluate import roc_auc  # noqa: E402

OUT = ROOT / "paper" / "generated"
FIG = OUT / "figures"
DATA = ROOT / "data" / "built" / "canonical_full.csv"
ZS_DATA = ROOT / "data" / "built" / "canonical_zs.csv"
RESULTS = ROOT / "results" / "dl"

N_BOOT = 10_000
SEEDS = (42, 43, 44)
SCOREABLE_MIN_N = 50
GENES = ("MLH1", "MSH2", "MSH6", "PMS2")

# ---------------------------------------------------------------------------
# The design: which folder and arm name each experimental condition lives under.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    key: str        # short id used in the text (A0 ... B3, M)
    runs: str       # runs/dl/<runs>
    name: str       # arm name as written by vpdl-dl
    label: str      # human label
    axis: str       # "feature" | "label"


ARMS = [
    Arm("A0", "ablation_features", "train-clinvar__drop-domains-genomic-gnomad-prior_scores-structure",
        "Sequence window only", "feature"),
    Arm("A1", "ablation_features", "train-clinvar__drop-domains-genomic-prior_scores-structure",
        "gnomAD frequency", "feature"),
    Arm("A2", "ablation_features", "train-clinvar__drop-domains-genomic-gnomad-structure",
        "AlphaMissense", "feature"),
    Arm("A3", "ablation_features", "train-clinvar__drop-domains-genomic-gnomad-prior_scores",
        "AlphaFold structure", "feature"),
    Arm("A4", "ablation_features", "train-clinvar__drop-domains-gnomad-prior_scores-structure",
        "Genomic position", "feature"),
    Arm("M", "main", "train-clinvar__drop-domains-genomic-structure",
        "gnomAD + AlphaMissense", "feature"),
    Arm("A5", "ablation_features", "train-clinvar",
        "All feature sources", "feature"),
    Arm("B3", "ablation_labels", "train-clinvar_expert",
        "ClinVar expert panel only", "label"),
    Arm("B1", "ablation_labels", "train-pg_dms",
        "MSH2 DMS only", "label"),
    Arm("B2", "ablation_labels", "train-clinvar+pg_dms",
        "ClinVar + MSH2 DMS", "label"),
]
ARM = {arm.key: arm for arm in ARMS}
FEATURE_ORDER = ["A0", "A4", "A3", "A1", "A2", "M", "A5"]
LABEL_ORDER = ["A5", "B3", "B1", "B2"]

TABULAR = ["gbm", "mlp"]
SEQUENCE = ["aa_mlp", "cnn", "bilstm", "bilstm_attn", "transformer"]
MODELS = TABULAR + SEQUENCE
MODEL_LABEL = {"gbm": "GBM", "mlp": "MLP", "aa_mlp": "Window MLP", "cnn": "CNN",
               "bilstm": "BiLSTM", "bilstm_attn": "BiLSTM+attn", "transformer": "Transformer"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def matches(path: Path, expected: str) -> bool:
    """Byte-identical, or identical once CRLF is read as LF (git autocrlf on Windows checkouts)."""
    if sha256(path) == expected:
        return True
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == expected


# ---------------------------------------------------------------------------
# Loading, with the integrity checks that make the numbers citable.
# ---------------------------------------------------------------------------


def check_inputs() -> dict:
    checks = json.loads((RESULTS / "checks.json").read_text())
    if not checks.get("citable") or checks.get("problems"):
        raise SystemExit(f"results/dl/checks.json is not citable: {checks.get('problems')}")
    if len(checks["dataset_sha256"]) != 1:
        raise SystemExit(f"cells span several datasets: {checks['dataset_sha256']}")
    expected = checks["dataset_sha256"][0]
    if not matches(DATA, expected):
        raise SystemExit(f"{DATA} hashes to {sha256(DATA)[:12]}, the grid used {expected[:12]}")
    manifest = json.loads((RESULTS / "MANIFEST.json").read_text())
    for name, entry in manifest["artefacts"].items():
        if not matches(ROOT / entry["path"], entry["sha256"]):
            raise SystemExit(f"{entry['path']} changed since results/dl/MANIFEST.json was written")
    return {"dataset_sha256": expected, "n_cells": checks["n_cells"], "notes": checks["notes"]}


def exact_scores(values: np.ndarray) -> np.ndarray:
    """The model's own score values, recovered from their CSV text.

    The models emit float32; the CSV holds each value's shortest float32 spelling, which
    parses to a slightly different float64. The threshold was compared with the float32
    value, and GBM scores tie AT the threshold often enough for the difference to flip
    predictions. If every value round-trips through float32, use the float32 values.
    """
    as32 = values.astype(np.float32)
    round_trips = np.array([float(np.format_float_positional(v, unique=True)) for v in as32]) == values
    return as32.astype(np.float64) if round_trips.all() else values


def load_predictions() -> pd.DataFrame:
    frames = []
    for arm in ARMS:
        for path in sorted((ROOT / "runs" / "dl" / arm.runs).glob(f"predictions_{arm.name}__*.csv")):
            cell = path.stem.removeprefix("predictions_")
            name, model, seed = parse_cell(cell)
            if name != arm.name or model not in MODELS:
                continue            # a longer arm name sharing the prefix, or gbm-es50
            frame = pd.read_csv(path, float_precision="round_trip")
            frame["score"] = exact_scores(frame["score"].to_numpy(float))
            frames.append(frame.assign(arm=arm.key, model=model, seed=seed))
    predictions = pd.concat(frames, ignore_index=True)
    counts = predictions.groupby(["arm", "model"]).seed.nunique()
    if (counts != len(SEEDS)).any():
        raise SystemExit(f"cells without three seeds:\n{counts[counts != len(SEEDS)]}")
    return predictions


def load_valpreds(arm: Arm, model: str) -> pd.DataFrame:
    frames = [pd.read_csv(path, float_precision="round_trip").assign(seed=parse_cell(path.stem.removeprefix("valpreds_"))[2])
              for path in sorted((ROOT / "runs" / "dl" / arm.runs)
                                 .glob(f"valpreds_{arm.name}__{model}__seed*.csv"))]
    return pd.concat(frames, ignore_index=True)


def verify_against_cells(predictions: pd.DataFrame) -> int:
    """Recompute AUROC and MCC per cell and gene; they must equal the DGX's own values."""
    cells = pd.read_csv(RESULTS / "cells.csv")
    checked = 0
    for (arm_key, model, seed, gene), group in predictions.groupby(["arm", "model", "seed", "gene"]):
        arm = ARM[arm_key]
        row = cells[(cells.runs_dir == f"runs/dl/{arm.runs}") & (cells.arm == arm.name)
                    & (cells.model == model) & (cells.seed == seed) & (cells.gene == gene)]
        if len(row) != 1:
            raise SystemExit(f"{arm_key}/{model}/{seed}/{gene}: {len(row)} rows in cells.csv")
        row = row.iloc[0]
        y, s = group.label.to_numpy(int), group.score.to_numpy(float)
        auc = roc_auc(y, s)
        mcc = _mcc_counts(*_confusion(y, s >= group.threshold.iloc[0]))
        if int(row.n) != len(y) or abs(auc - row.roc_auc) > 1e-9 or abs(mcc - row.mcc) > 1e-9:
            raise SystemExit(f"{arm_key}/{model}/{seed}/{gene}: recomputed AUROC {auc} MCC {mcc} "
                             f"n {len(y)} vs cells.csv {row.roc_auc} {row.mcc} {row.n}")
        checked += 1
    return checked


def verify_same_test_sets(predictions: pd.DataFrame) -> dict[str, pd.Series]:
    """Every (arm, model, seed) scored the same variants with the same labels, per gene."""
    reference: dict[str, pd.Series] = {}
    for (arm, model, seed, gene), group in predictions.groupby(["arm", "model", "seed", "gene"]):
        labels = group.set_index("variant_key").label.sort_index()
        if gene not in reference:
            reference[gene] = labels
        elif not labels.equals(reference[gene]):
            raise SystemExit(f"{arm}/{model}/{seed}: {gene} test set differs from the others")
    return reference


# ---------------------------------------------------------------------------
# Vectorised paired bootstrap. Resample variants within each gene; the same
# resample scores every arm in a comparison.
# ---------------------------------------------------------------------------


def _confusion(y: np.ndarray, predicted: np.ndarray, w: np.ndarray | None = None):
    w = np.ones(len(y)) if w is None else w
    pos, neg = y == 1, y == 0
    tp = (w * (pos & predicted)).sum(-1)
    fn = (w * (pos & ~predicted)).sum(-1)
    fp = (w * (neg & predicted)).sum(-1)
    tn = (w * (neg & ~predicted)).sum(-1)
    return tp, fp, tn, fn


def _mcc_counts(tp, fp, tn, fn):
    tp, fp, tn, fn = (np.asarray(v, float) for v in (tp, fp, tn, fn))
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1), 0.0)
    return float(out) if out.ndim == 0 else out


def _pair_matrix(s_pos: np.ndarray, s_neg: np.ndarray) -> np.ndarray:
    return (s_pos[:, None] > s_neg[None, :]) + 0.5 * (s_pos[:, None] == s_neg[None, :])


@dataclass
class GeneData:
    """One gene's labels and one condition's per-seed scores and thresholds."""
    y: np.ndarray                  # (n,)
    scores: np.ndarray             # (n, k) k seeds (k = 1 for a zero-training score)
    thresholds: np.ndarray | None  # (k,) or None


def _weights(rng: np.random.Generator, n: int, n_boot: int) -> np.ndarray:
    picks = rng.integers(0, n, size=(n_boot, n))
    weights = np.zeros((n_boot, n))
    np.add.at(weights, (np.repeat(np.arange(n_boot), n), picks.ravel()), 1.0)
    return weights


def auc_given_weights(gene: GeneData, w: np.ndarray) -> np.ndarray:
    """Seed-mean AUROC for each weight row (w: (B, n)) — Mann-Whitney on weighted pairs."""
    pos, neg = gene.y == 1, gene.y == 0
    wp, wn = w[:, pos], w[:, neg]
    denom = wp.sum(1) * wn.sum(1)
    values = []
    for k in range(gene.scores.shape[1]):
        pair = _pair_matrix(gene.scores[pos, k], gene.scores[neg, k])
        with np.errstate(invalid="ignore", divide="ignore"):
            values.append(((wp @ pair) * wn).sum(1) / denom)
    return np.mean(values, axis=0)


def mcc_given_weights(gene: GeneData, w: np.ndarray) -> np.ndarray:
    values = []
    for k in range(gene.scores.shape[1]):
        predicted = gene.scores[:, k] >= gene.thresholds[k]
        values.append(_mcc_counts(*_confusion(gene.y, predicted, w)))
    return np.mean(values, axis=0)


METRIC = {"auc": auc_given_weights, "mcc": mcc_given_weights}


def bootstrap(conditions: list[dict[str, GeneData]], genes: list[str], metric: str,
              seed: int = 0, n_boot: int = N_BOOT) -> dict:
    """Point and 95% CI of the gene-mean metric for 1 condition, or of the difference for 2."""
    rng = np.random.default_rng(seed)
    function = METRIC[metric]
    point, draws = [], []
    for gene in genes:
        base = conditions[0][gene]
        ones = np.ones((1, len(base.y)))
        w = _weights(rng, len(base.y), n_boot)
        values = [function(c[gene], ones)[0] for c in conditions]
        boot = [function(c[gene], w) for c in conditions]
        point.append(values[0] - values[1] if len(conditions) == 2 else values[0])
        draws.append(boot[0] - boot[1] if len(conditions) == 2 else boot[0])
    means = np.nanmean(np.vstack(draws), axis=0)
    means = means[np.isfinite(means)]
    return {"point": float(np.mean(point)), "low": float(np.percentile(means, 2.5)),
            "high": float(np.percentile(means, 97.5)), "genes": genes,
            "per_gene": dict(zip(genes, point))}


# ---------------------------------------------------------------------------
# Condition construction.
# ---------------------------------------------------------------------------


def trained_condition(predictions: pd.DataFrame, arm: str, model: str,
                      order: dict[str, pd.Index], keep: set | None = None) -> dict[str, GeneData]:
    out = {}
    rows = predictions[(predictions.arm == arm) & (predictions.model == model)]
    for gene, group in rows.groupby("gene"):
        index = order[gene] if keep is None else order[gene][order[gene].isin(keep)]
        wide = group.pivot_table(index="variant_key", columns="seed", values="score").loc[index]
        thresholds = group.groupby("seed").threshold.first().loc[wide.columns].to_numpy()
        labels = group.drop_duplicates("variant_key").set_index("variant_key").label.loc[index]
        out[gene] = GeneData(labels.to_numpy(int), wide.to_numpy(float), thresholds)
    return out


def feature_condition(table: pd.DataFrame, column: str, sign: float,
                      order: dict[str, pd.Index], keep: set | None = None) -> dict[str, GeneData]:
    """A zero-training score: rank held-out variants by one raw column (oriented by `sign`)."""
    out = {}
    for gene, index in order.items():
        index = index if keep is None else index[index.isin(keep)]
        rows = table.loc[index]
        out[gene] = GeneData(rows["label__clinvar"].to_numpy(int),
                             (sign * rows[column].to_numpy(float))[:, None], None)
    return out


def identical_genes(a: dict[str, GeneData], b: dict[str, GeneData]) -> list[str]:
    return [g for g in a if g in b and a[g].scores.shape == b[g].scores.shape
            and np.array_equal(a[g].scores, b[g].scores)]


def compare(a: dict[str, GeneData], b: dict[str, GeneData], genes: list[str], metric: str) -> dict:
    same = identical_genes(a, b)
    usable = [g for g in genes if g in a and g in b and g not in same]
    result = bootstrap([a, b], usable, metric)
    result["identical_genes_dropped"] = [g for g in genes if g in same]
    result["excludes_zero"] = bool(result["low"] > 0 or result["high"] < 0)
    return result


# ---------------------------------------------------------------------------
# Formatting.
# ---------------------------------------------------------------------------

MACROS: dict[str, str] = {}


def macro(name: str, value) -> str:
    if not name.isalpha():
        raise ValueError(f"LaTeX macro names are letters only: {name!r}")
    if name in MACROS and MACROS[name] != str(value):
        raise ValueError(f"macro {name} defined twice with different values")
    MACROS[name] = str(value)
    return str(value)


def f3(x: float) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.3f}"


def signed(x: float) -> str:
    return f"{x:+.3f}".replace("-", "\N{MINUS SIGN}") if np.isfinite(x) else "n/a"


def tex_signed(x: float) -> str:
    return f"{x:+.3f}".replace("-", "$-$") if np.isfinite(x) else "n/a"


def ci_tex(result: dict) -> str:
    return f"{tex_signed(result['point'])} [{tex_signed(result['low'])}, {tex_signed(result['high'])}]"


def word(key: str) -> str:
    """Macro-safe spelling of an arm/model key: digits become words."""
    for digit, name in zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five",
                                          "Six", "Seven", "Eight", "Nine"]):
        key = key.replace(digit, name)
    return "".join(part[:1].upper() + part[1:] for part in key.replace("+", "_").split("_"))


# ---------------------------------------------------------------------------
# The analyses.
# ---------------------------------------------------------------------------


def arm_summary(predictions: pd.DataFrame, scoreable: list[str]) -> pd.DataFrame:
    """Per (arm, model): per-gene metric per seed -> mean over genes per seed -> mean, SD over seeds."""
    from vpdl.evaluate import _average_precision
    rows = []
    for (arm, model, seed, gene), group in predictions.groupby(["arm", "model", "seed", "gene"]):
        y, s = group.label.to_numpy(int), group.score.to_numpy(float)
        rows.append({"arm": arm, "model": model, "seed": seed, "gene": gene, "n": len(y),
                     "n_pathogenic": int(y.sum()),
                     "auc": roc_auc(y, s),
                     "mcc": _mcc_counts(*_confusion(y, s >= group.threshold.iloc[0])),
                     "pr_auc": _average_precision(y, s),
                     "brier": brier_score(y, s), "ece": expected_calibration_error(y, s)})
    per_gene = pd.DataFrame(rows)
    per_gene.to_csv(OUT / "per_gene_per_seed.csv", index=False)
    out = []
    for (arm, model), group in per_gene.groupby(["arm", "model"]):
        genes = [g for g in scoreable if g in set(group.gene)]
        by_seed = group[group.gene.isin(genes)].groupby("seed")[["auc", "mcc", "pr_auc", "brier", "ece"]].mean()
        record = {"arm": arm, "model": model, "genes": "+".join(genes),
                  "n": int(group[(group.seed == SEEDS[0]) & group.gene.isin(genes)].n.sum())}
        for metric in by_seed.columns:
            record[f"{metric}_mean"] = by_seed[metric].mean()
            record[f"{metric}_sd"] = by_seed[metric].std(ddof=1)
        out.append(record)
    return pd.DataFrame(out), per_gene


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    provenance = check_inputs()

    table = pd.read_csv(DATA, low_memory=False)
    from vpdl.splits import variant_keys
    table.index = variant_keys(table)
    if not table.index.is_unique:
        raise SystemExit("variant keys are not unique in the dataset")

    predictions = load_predictions()
    checked = verify_against_cells(predictions)
    test_sets = verify_same_test_sets(predictions)
    order = {g: labels.index for g, labels in test_sets.items()}
    for gene, labels in test_sets.items():     # the dataset agrees with every prediction file
        if not (table.loc[labels.index, "label__clinvar"].astype(int).to_numpy() == labels.to_numpy()).all():
            raise SystemExit(f"{gene}: dataset labels differ from the prediction files")

    scoreable = [g for g in GENES if g in test_sets and len(test_sets[g]) >= SCOREABLE_MIN_N]
    dms_genes = [g for g in scoreable if g != "MSH2"]          # genes where DMS labels can train
    macro("nCellsChecked", checked)
    macro("nCells", predictions.groupby(["arm", "model", "seed"]).ngroups)
    macro("nArms", predictions.groupby(["arm", "model"]).ngroups)
    macro("datasetSha", provenance["dataset_sha256"][:12])
    macro("nBoot", f"{N_BOOT:,}".replace(",", "{,}"))

    # -- dataset composition --------------------------------------------------
    labelled = table[table.label__clinvar.notna()]
    comp = []
    for gene in GENES:
        rows = labelled[labelled.gene == gene]
        comp.append({"gene": gene, "rows": int((table.gene == gene).sum()), "n": len(rows),
                     "pathogenic": int((rows.label__clinvar == 1).sum()),
                     "benign": int((rows.label__clinvar == 0).sum()),
                     "expert": int(rows.label__clinvar_expert.notna().sum()),
                     "dms": int(table[(table.gene == gene)].label__pg_dms.notna().sum()),
                     "gnomad_observed_benign": int(((rows.label__clinvar == 0) & (rows.feature_gnomad_observed == 1)).sum()),
                     "gnomad_observed_pathogenic": int(((rows.label__clinvar == 1) & (rows.feature_gnomad_observed == 1)).sum())})
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT / "composition.csv", index=False)
    for record in comp.itertuples():
        g = word(record.gene)
        for field in ("rows", "n", "pathogenic", "benign", "expert", "dms"):
            macro(f"{field}{g}", f"{getattr(record, field):,}".replace(",", "{,}"))
    macro("nLabelled", len(labelled))
    macro("nLabelledScoreable", int(comp[comp.gene.isin(scoreable)].n.sum()))
    macro("nBenignScoreable", int(comp[comp.gene.isin(scoreable)].benign.sum()))
    macro("nPathogenicScoreable", int(comp[comp.gene.isin(scoreable)].pathogenic.sum()))
    macro("nExpert", int(labelled.label__clinvar_expert.notna().sum()))
    macro("nTwoStar", int((labelled.clinvar_star_rating == 2).sum()))
    macro("nThreeStar", int((labelled.clinvar_star_rating == 3).sum()))
    macro("nDms", f"{int(table.label__pg_dms.notna().sum()):,}".replace(",", "{,}"))
    macro("nDmsBenign", f"{int((table.label__pg_dms == 0).sum()):,}".replace(",", "{,}"))
    macro("nDmsPathogenic", f"{int((table.label__pg_dms == 1).sum()):,}".replace(",", "{,}"))
    macro("nRows", f"{len(table):,}".replace(",", "{,}"))
    macro("nConflictQuarantined", int((table.label_source == "conflict_quarantined").sum()))
    b_obs = int(((labelled.label__clinvar == 0) & (labelled.feature_gnomad_observed == 1)).sum())
    p_obs = int(((labelled.label__clinvar == 1) & (labelled.feature_gnomad_observed == 1)).sum())
    n_b, n_p = int((labelled.label__clinvar == 0).sum()), int((labelled.label__clinvar == 1).sum())
    macro("benignObserved", b_obs)
    macro("benignTotal", n_b)
    macro("benignObservedPct", f"{100 * b_obs / n_b:.0f}")
    macro("pathObserved", p_obs)
    macro("pathTotal", n_p)
    macro("pathObservedPct", f"{100 * p_obs / n_p:.0f}")
    bs1 = int(((labelled.label__clinvar == 0) & ((labelled.feature_acmg_bs1 == 1) | (labelled.feature_acmg_ba1 == 1))).sum())
    macro("benignBsOne", bs1)
    macro("pathBsOne", int(((labelled.label__clinvar == 1) & ((labelled.feature_acmg_bs1 == 1) | (labelled.feature_acmg_ba1 == 1))).sum()))
    macro("pmsHomologyRows", f"{int(table.qc_flags.fillna('').str.contains('pms2_homology_unconfirmed').sum()):,}".replace(",", "{,}"))

    # -- per-arm summary ------------------------------------------------------
    summary, per_gene = arm_summary(predictions, scoreable)
    summary.to_csv(OUT / "arm_summary.csv", index=False)

    # Bootstrap CI of each cell's gene-mean AUROC (variant resampling; seeds averaged).
    conditions = {(a, m): trained_condition(predictions, a, m, order)
                  for a, m in predictions.groupby(["arm", "model"]).groups}
    cell_ci = {}
    for (a, m), cond in conditions.items():
        genes = dms_genes if a == "B1" else scoreable
        cell_ci[(a, m)] = bootstrap([cond], genes, "auc")
    summary["auc_ci_low"] = [cell_ci[(r.arm, r.model)]["low"] for r in summary.itertuples()]
    summary["auc_ci_high"] = [cell_ci[(r.arm, r.model)]["high"] for r in summary.itertuples()]
    summary.to_csv(OUT / "arm_summary.csv", index=False)
    for r in summary.itertuples():
        stem = f"{word(r.arm)}{word(r.model)}"
        macro(f"auc{stem}", f3(r.auc_mean))
        macro(f"aucSd{stem}", f3(r.auc_sd))
        macro(f"mcc{stem}", f3(r.mcc_mean))
        macro(f"mccSd{stem}", f3(r.mcc_sd))
        macro(f"brier{stem}", f3(r.brier_mean))
        macro(f"ece{stem}", f3(r.ece_mean))
        macro(f"aucCi{stem}", f"[{f3(r.auc_ci_low)}, {f3(r.auc_ci_high)}]")

    # -- zero-training baselines -----------------------------------------------
    zs = pd.read_csv(ZS_DATA, low_memory=False)
    zs.index = variant_keys(zs)
    esm_cols = ["zeroshot_esm1b_P0_masked_marginal", "zeroshot_esm2_650m_P0_masked_marginal"]
    for column in esm_cols:
        table[column] = zs[column].reindex(table.index)
    baselines = {
        "neg_log_af": ("feature_gnomad_log10_af", -1.0, r"$-\log_{10}$ gnomAD AF"),
        "alphamissense": ("feature_alphamissense_score", 1.0, "AlphaMissense"),
        # The zero-shot columns hold ``pathogenicity`` = -log-ratio, already oriented
        # (vpdl.dl.plm.zeroshot.zeroshot_frame -> vpdl.features.pllr_to_pathogenicity).
        "esm1b": ("zeroshot_esm1b_P0_masked_marginal", 1.0, "ESM-1b masked marginal"),
        "esm2": ("zeroshot_esm2_650m_P0_masked_marginal", 1.0, "ESM-2 650M masked marginal"),
    }
    base_conditions, base_rows = {}, []
    for key, (column, sign, label) in baselines.items():
        heldout = pd.Index(np.concatenate([order[g] for g in scoreable]))
        missing = int(table.loc[heldout, column].isna().sum())
        if missing:
            raise SystemExit(f"baseline {column}: {missing} held-out variants have no value")
        cond = feature_condition(table, column, sign, order)
        base_conditions[key] = cond
        result = bootstrap([cond], scoreable, "auc")
        if result["point"] < 0.5:     # orientation is fixed by the column's definition above;
            raise SystemExit(f"baseline {key}: AUROC {result['point']:.3f} < 0.5 - orientation error")
        base_rows.append({"baseline": key, "label": label, "auc": result["point"],
                          "low": result["low"], "high": result["high"], **{
                              f"auc_{g}": v for g, v in result["per_gene"].items()}})
        macro(f"base{word(key)}", f3(result["point"]))
        macro(f"baseCi{word(key)}", f"[{f3(result['low'])}, {f3(result['high'])}]")
    base_table = pd.DataFrame(base_rows)
    base_table.to_csv(OUT / "baselines.csv", index=False)
    # Exploratory: the simplest combination of the two strongest inputs, with no training -
    # the mean of their within-gene percentile ranks over the held-out variants (labels unused).
    combo = {}
    for gene, index in order.items():
        sub = table.loc[index]
        score = (sub["feature_gnomad_log10_af"].rank(pct=True, ascending=False)
                 + sub["feature_alphamissense_score"].rank(pct=True)) / 2
        combo[gene] = GeneData(sub.label__clinvar.to_numpy(int), score.to_numpy(float)[:, None], None)
    base_conditions["rank_mean"] = combo
    result = bootstrap([combo], scoreable, "auc")
    macro("baseRankMean", f3(result["point"]))
    macro("baseCiRankMean", f"[{f3(result['low'])}, {f3(result['high'])}]")
    base_table = pd.concat([base_table, pd.DataFrame([{
        "baseline": "rank_mean", "label": "Mean rank of $-\\log_{10}$ AF and AlphaMissense",
        "auc": result["point"], "low": result["low"], "high": result["high"],
        **{f"auc_{g}": v for g, v in result["per_gene"].items()}}])], ignore_index=True)
    base_table.to_csv(OUT / "baselines.csv", index=False)

    # -- pre-specified comparisons ---------------------------------------------
    comparisons = []

    def add(name: str, a: tuple, b: tuple, genes: list[str], family: str, a_cond=None, b_cond=None):
        ca = a_cond if a_cond is not None else conditions[a]
        cb = b_cond if b_cond is not None else conditions[b]
        row = {"comparison": name, "family": family, "a": "/".join(a), "b": "/".join(b)}
        for metric in ("auc", "mcc"):
            if metric == "mcc" and (ca[genes[0]].thresholds is None or cb[genes[0]].thresholds is None):
                continue
            result = compare(ca, cb, genes, metric)
            row.update({f"{metric}_delta": result["point"], f"{metric}_low": result["low"],
                        f"{metric}_high": result["high"], f"{metric}_excludes_zero": result["excludes_zero"],
                        "genes": "+".join(result["genes"]),
                        "identical_genes_dropped": "+".join(result["identical_genes_dropped"])})
            row.update({f"{metric}_{g}": v for g, v in result["per_gene"].items()})
            macro(f"d{metric.capitalize()}{name}", ci_tex(result))
            macro(f"d{metric.capitalize()}{name}Point", tex_signed(result["point"]))
            macro(f"d{metric.capitalize()}{name}Abs", f"{abs(result['point']):.3f}")
        comparisons.append(row)

    # Feature axis: each source against the complete feature set, per model.
    for model in MODELS:
        m = word(model)
        for key in ("A1", "A2", "A3", "A4", "M"):
            if (key, model) in conditions:
                add(f"{word(key)}Vs{word('A5')}{m}", (key, model), ("A5", model), scoreable, "feature_vs_all")
        if ("A0", model) in conditions:
            for key in ("A1", "A2", "A3", "A4", "A5"):
                add(f"{word(key)}Vs{word('A0')}{m}", (key, model), ("A0", model), scoreable, "feature_vs_sequence")
    # Label axis: DMS labels can only enter training on the genes that are not MSH2.
    for model in MODELS:
        m = word(model)
        add(f"BTwoVsAFive{m}", ("B2", model), ("A5", model), dms_genes, "label")
        add(f"BOneVsAFive{m}", ("B1", model), ("A5", model), dms_genes, "label")
        add(f"BThreeVsAFive{m}", ("B3", model), ("A5", model), scoreable, "label")
    # Model class: every model against GBM within the two strongest feature arms.
    for key in ("M", "A5"):
        for model in MODELS:
            if model != "gbm":
                add(f"{word(model)}VsGbm{word(key)}", (key, model), (key, "gbm"), scoreable, "model_vs_gbm")
    # Supervision over the zero-training baselines.
    for key in ("M", "A5"):
        for model in MODELS:
            add(f"{word(key)}{word(model)}VsNegLogAf", (key, model), ("-", "neg_log_af"), scoreable,
                "trained_vs_baseline", b_cond=base_conditions["neg_log_af"])
            add(f"{word(key)}{word(model)}VsRankMean", (key, model), ("-", "rank_mean"), scoreable,
                "trained_vs_baseline", b_cond=base_conditions["rank_mean"])
            add(f"{word(key)}{word(model)}VsAlphamissense", (key, model), ("-", "alphamissense"), scoreable,
                "trained_vs_baseline", b_cond=base_conditions["alphamissense"])
    comp_frame = pd.DataFrame(comparisons)
    comp_frame.to_csv(OUT / "comparisons.csv", index=False)

    # Counting statements the text makes about families of comparisons.
    fam = comp_frame
    lab = fam[fam.family == "label"]
    b2 = lab[lab.comparison.str.startswith("BTwoVsAFive")]
    macro("nModelsBTwoMccWorse", int(((b2.mcc_high < 0)).sum()))
    macro("nModelsBTwoAucWorse", int(((b2.auc_high < 0)).sum()))
    macro("nModelsBTwoAucBetter", int(((b2.auc_low > 0)).sum()))
    b3 = lab[lab.comparison.str.startswith("BThreeVsAFive")]
    macro("nModelsBThreeAucExcl", int(b3.auc_excludes_zero.sum()))
    macro("nModelsBThreeMccExcl", int(b3.mcc_excludes_zero.sum()))
    b1 = lab[lab.comparison.str.startswith("BOneVsAFive")]
    macro("nModelsBOneAucWorse", int((b1.auc_high < 0).sum()))
    macro("nModelsBOneMccWorse", int((b1.mcc_high < 0).sum()))
    fa = fam[(fam.family == "feature_vs_all") & fam.comparison.str.startswith("MVsAFive")]
    macro("nMVsAFiveAucNull", int((~fa.auc_excludes_zero).sum()))
    macro("nMVsAFiveMccBetter", int((fa.mcc_low > 0).sum()))
    macro("maxAbsMVsAFiveAuc", f"{fa.auc_delta.abs().max():.3f}")
    b2 = lab[lab.comparison.str.startswith("BTwoVsAFive")]
    macro("nModelsBTwoAucNull", int((~b2.auc_excludes_zero).sum()))
    macro("bTwoMccWorst", tex_signed(b2.mcc_delta.min()))
    macro("bTwoMccLeast", tex_signed(b2.mcc_delta.max()))
    b1 = lab[lab.comparison.str.startswith("BOneVsAFive")]
    macro("bOneAucWorst", tex_signed(b1.auc_delta.min()))
    macro("bOneAucLeast", tex_signed(b1.auc_delta.max()))
    worst = summary.loc[summary.mcc_sd.idxmax()]
    macro("maxMccSd", f3(worst.mcc_sd))
    macro("maxMccSdCell", f"{worst.arm}/{MODEL_LABEL[worst.model]}")
    macro("maxAucSd", f3(summary.auc_sd.max()))
    macro("nComparisons", len(comparisons))
    mv = fam[fam.family == "model_vs_gbm"]
    for key in ("M", "A5"):
        sub = mv[mv.comparison.str.endswith(word(key))]
        seq = sub[sub.comparison.str.match("^(" + "|".join(word(m) for m in SEQUENCE) + ")Vs")]
        macro(f"nSeqBeatGbmAuc{word(key)}", int((seq.auc_low > 0).sum()))
        macro(f"nSeqWorseGbmAuc{word(key)}", int((seq.auc_high < 0).sum()))
    tv = fam[fam.family == "trained_vs_baseline"]
    for key in ("M", "A5"):
        for base in ("NegLogAf", "RankMean", "Alphamissense"):
            sub = tv[tv.comparison.str.startswith(word(key)) & tv.comparison.str.endswith(base)]
            macro(f"nBeat{base}{word(key)}", int((sub.auc_low > 0).sum()))

    # -- circularity: performance where gnomAD presence cannot separate the classes --
    observed = set(table.index[table.feature_gnomad_observed == 1])
    obs_rows = []
    targets = [("baseline", "neg_log_af"), ("baseline", "alphamissense"), ("baseline", "esm1b"),
               ("M", "gbm"), ("M", "mlp"), ("M", "bilstm"), ("A5", "gbm"), ("A5", "mlp"),
               ("A1", "gbm"), ("A1", "mlp"), ("A2", "gbm"), ("A2", "mlp"), ("A0", "bilstm_attn")]
    for a, m in targets:
        if a == "baseline":
            column, sign, _ = baselines[m]
            full, sub = base_conditions[m], feature_condition(table, column, sign, order, keep=observed)
        else:
            full, sub = conditions[(a, m)], trained_condition(predictions, a, m, order, keep=observed)
        r_all = bootstrap([full], scoreable, "auc")
        r_obs = bootstrap([sub], scoreable, "auc")
        obs_rows.append({"condition": a, "model": m, "auc_all": r_all["point"], "all_low": r_all["low"],
                         "all_high": r_all["high"], "auc_observed": r_obs["point"],
                         "observed_low": r_obs["low"], "observed_high": r_obs["high"]})
        stem = f"{word(a)}{word(m)}"
        macro(f"aucObs{stem}", f3(r_obs["point"]))
        macro(f"aucObsCi{stem}", f"[{f3(r_obs['low'])}, {f3(r_obs['high'])}]")
    obs = pd.DataFrame(obs_rows)
    obs.to_csv(OUT / "observed_subset.csv", index=False)
    obs_counts = {g: (int(table.loc[order[g][order[g].isin(observed)], "label__clinvar"].eq(0).sum()),
                      int(table.loc[order[g][order[g].isin(observed)], "label__clinvar"].eq(1).sum()))
                  for g in scoreable}
    macro("nObsBenign", sum(v[0] for v in obs_counts.values()))
    macro("nObsPath", sum(v[1] for v in obs_counts.values()))
    for g, (nb, npth) in obs_counts.items():
        macro(f"nObsBenign{word(g)}", nb)
        macro(f"nObsPath{word(g)}", npth)

    # -- B-1 sensitivity: the rows whose gnomAD feature disagrees with a recount ----
    lab_rows = table[table.label__clinvar.notna()]
    first = 10 ** lab_rows.feature_gnomad_log10_af
    feature_observed = lab_rows.feature_gnomad_observed == 1
    display_observed = lab_rows.gnomad_af.notna()
    disagree = (feature_observed != display_observed) | (
        feature_observed & display_observed & ~np.isclose(first, lab_rows.gnomad_af, rtol=1e-6))
    affected = set(lab_rows.index[disagree])
    macro("nBOne", len(affected))
    macro("nBOneGenes", ", ".join(f"{g} {n}" for g, n in lab_rows[disagree].gene.value_counts().sort_index().items()))
    keep = set(table.index) - affected
    sens = []
    for (a, m) in [("M", "mlp"), ("M", "gbm"), ("A5", "gbm"), ("A5", "mlp"), ("A1", "gbm")]:
        full = bootstrap([conditions[(a, m)]], scoreable, "auc")
        reduced = bootstrap([trained_condition(predictions, a, m, order, keep=keep)], scoreable, "auc")
        sens.append({"arm": a, "model": m, "auc_all": full["point"], "auc_without_b1": reduced["point"],
                     "change": reduced["point"] - full["point"]})
    sens = pd.DataFrame(sens)
    sens.to_csv(OUT / "b1_sensitivity.csv", index=False)
    macro("bOneMaxChange", f"{sens.change.abs().max():.3f}")
    key_rows = []
    for name in ("BTwoVsAFiveGbm", "BTwoVsAFiveMlp", "AOneVsAFiveGbm", "MVsAFiveGbm"):
        base_row = comp_frame[comp_frame.comparison == name].iloc[0]
        a_key, a_model = base_row.a.split("/")
        b_key, b_model = base_row.b.split("/")
        genes = base_row.genes.split("+")
        r = compare(trained_condition(predictions, a_key, a_model, order, keep=keep),
                    trained_condition(predictions, b_key, b_model, order, keep=keep), genes, "auc")
        key_rows.append({"comparison": name, "delta_all": base_row.auc_delta,
                         "delta_without_b1": r["point"], "low": r["low"], "high": r["high"]})
    pd.DataFrame(key_rows).to_csv(OUT / "b1_sensitivity_comparisons.csv", index=False)

    # -- threshold transfer: what MCC loses when the threshold comes from other labels --
    thresholds = predictions.groupby(["arm", "model", "seed"]).threshold.agg(["min", "max"])
    thresholds.to_csv(OUT / "thresholds.csv")
    prevalence = {}
    for key in ("A5", "B1", "B2", "B3"):
        vp = load_valpreds(ARM[key], "gbm")
        prevalence[key] = float(vp.label.mean())
        macro(f"valPrev{word(key)}", f"{prevalence[key]:.2f}")
    macro("testPrev", f"{labelled[labelled.gene.isin(scoreable)].label__clinvar.mean():.2f}")

    # -- features each arm trained on. cells.csv leaves n_features empty (vpdl/dl/report.py),
    # so resolve them with the training CLI's own function from the arm's drop groups. ------
    from vpdl.dl.cli import _columns
    feature_columns = {}
    for arm in ARMS:
        drops = arm.name.split("__drop-")[1].split("-") if "__drop-" in arm.name else []
        feature_columns[arm.key] = _columns(table, drops, allow_proxy_leak=True)
        macro(f"nFeat{word(arm.key)}", len(feature_columns[arm.key]))
        # Each run recorded a hash of the columns it trained on; the reconstruction must match.
        from vpdl.provenance import schema_hash
        expected = schema_hash(feature_columns[arm.key])
        for summary_path in (ROOT / "runs" / "dl" / arm.runs).glob(f"summary_{arm.name}__*__seed*.json"):
            if parse_cell(summary_path.stem.removeprefix("summary_"))[0] != arm.name:
                continue
            recorded = json.loads(summary_path.read_text())["provenance"].get("feature_schema_columns")
            if recorded != expected:
                raise SystemExit(f"{summary_path.name}: feature schema {recorded} != reconstructed {expected}")
    (OUT / "feature_columns.json").write_text(json.dumps(feature_columns, indent=1) + "\n", encoding="utf-8")

    # -- GBM with early stopping (run for arm M only): a sensitivity check on GBM's defaults ----
    es_frames = []
    for path in sorted((ROOT / "runs" / "dl" / "main").glob("predictions_*__gbm-es50__seed*.csv")):
        frame = pd.read_csv(path, float_precision="round_trip")
        frame["score"] = exact_scores(frame["score"].to_numpy(float))
        es_frames.append(frame.assign(arm="M", model="gbm-es50", seed=parse_cell(path.stem.removeprefix("predictions_"))[2]))
    es = pd.concat(es_frames, ignore_index=True)
    es_condition = trained_condition(es, "M", "gbm-es50", order)
    r = bootstrap([es_condition], scoreable, "auc")
    macro("aucMGbmEs", f3(r["point"]))
    r = compare(es_condition, conditions[("M", "gbm")], scoreable, "auc")
    macro("dAucGbmEsVsGbm", ci_tex(r))
    r = compare(es_condition, conditions[("M", "gbm")], scoreable, "mcc")
    macro("dMccGbmEsVsGbm", ci_tex(r))
    r = compare(es_condition, conditions[("M", "mlp")], scoreable, "auc")
    macro("dAucGbmEsVsMlp", ci_tex(r))

    # -- repeat executions: cells re-run later were archived, not reported. Did they reproduce? --
    repeats, identical = 0, 0
    for archived in sorted((ROOT / "runs" / "dl").glob("*/superseded/*/predictions_*.csv")):
        current = archived.parents[2] / archived.name
        if not current.exists():
            current = next((ROOT / "runs" / "dl" / arm.runs / archived.name for arm in ARMS
                            if (ROOT / "runs" / "dl" / arm.runs / archived.name).exists()), None)
        if current is None:
            continue
        repeats += 1
        identical += (archived.read_bytes().replace(b"\r\n", b"\n")
                      == current.read_bytes().replace(b"\r\n", b"\n"))
    macro("nRepeatRuns", repeats)
    macro("nRepeatIdentical", identical)

    # -- tables and figures ------------------------------------------------------
    import paper_tables as T
    T.write_all(OUT, summary, per_gene, base_table, comp_frame, comp, obs, sens, ARMS,
                FEATURE_ORDER, LABEL_ORDER, MODELS, MODEL_LABEL, scoreable, dms_genes, word)
    import paper_figures as F
    F.write_all(FIG, summary, per_gene, base_table, comp_frame, table, order, scoreable, dms_genes,
                obs, ARM, FEATURE_ORDER, MODELS, MODEL_LABEL, word)

    # -- numbers.tex and the manifest ---------------------------------------------
    lines = ["% Generated by paper/analysis.py - do not edit. One macro per number in the text."]
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in sorted(MACROS.items())]
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "MANIFEST.json")
    (OUT / "MANIFEST.json").write_text(json.dumps({
        "generator": "paper/analysis.py",
        "inputs": {"dataset": {"path": str(DATA.relative_to(ROOT).as_posix()), "sha256": provenance["dataset_sha256"]},
                   "zero_shot": {"path": str(ZS_DATA.relative_to(ROOT).as_posix()), "sha256": sha256(ZS_DATA)},
                   "results_manifest_sha256": sha256(RESULTS / "MANIFEST.json"),
                   "cells_verified": checked},
        "n_bootstrap": N_BOOT,
        "outputs": {p.relative_to(OUT).as_posix(): sha256(p) for p in outputs},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"ok: {checked} cell-gene metrics verified against results/dl/cells.csv; "
          f"{len(MACROS)} numbers, {len(comp_frame)} comparisons -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
