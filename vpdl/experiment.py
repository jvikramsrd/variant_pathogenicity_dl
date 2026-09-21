"""One cell of the experiment matrix, and the sweep that fills it.

A cell is ``(sources, model, seed)`` evaluated leave-one-gene-out. The paper's
claim is a comparison between cells that differ only in ``sources``, so
everything else — split construction, threshold selection, metric computation,
provenance — is shared code rather than per-arm code.

Threshold selection deserves special mention: it happens on an inner-validation
split carved out of the TRAINING genes, never on the held-out gene. v1 computed
inner-validation probabilities and then discarded them, which left no leak-free
split on which to calibrate and blocked an entire manuscript figure.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vpdl.assemble import resolve_labels
from vpdl.evaluate import best_threshold_by_mcc, evaluation_report
from vpdl.features import (
    build_feature_matrix,
    drop_gene_constant,
    local_windows,
    resolve_ablation,
)
from vpdl.models import build_model, derive_seed
from vpdl.provenance import provenance_record, schema_hash
from vpdl.splits import assert_no_group_straddle, group_keys, variant_keys

logger = logging.getLogger(__name__)

__all__ = ["CellConfig", "CellResult", "run_cell", "run_sweep",
           "SEQUENCE_WINDOW_MODELS"]

# Models consuming residue windows rather than a feature matrix. They take a
# different fit signature, so run_cell branches rather than pretending the
# interfaces match.
SEQUENCE_WINDOW_MODELS = frozenset({"bilstm"})


def _windows_for(
    frame: pd.DataFrame,
    sequences: Mapping[str, str],
    radius: int = 7,
) -> list[tuple[str, str]]:
    """Build (wild-type, variant) residue windows for every row."""
    return [
        local_windows(
            sequences[row.uniprot_id], int(row.position),
            row.wt_aa, row.mut_aa, radius=radius,
        )
        for row in frame.itertuples()
    ]


@dataclass
class CellConfig:
    """One cell of the matrix.

    Three different "sources" questions, kept apart deliberately:

    * `sources` — what the table was BUILT from. Provenance only.
    * `train_sources` — which label sources this arm TRAINS on. **This is the
      experiment's independent variable.**
    * `eval_source` — what every arm is SCORED against. Held fixed, so all arms
      are compared on the identical held-out variants with identical ground
      truth. Scoring the pooled arm on DMS labels and the ClinVar arm on
      clinical labels would compare two different test sets, not two
      training regimes.
    """

    sources: tuple[str, ...]
    train_sources: tuple[str, ...] = ("clinvar",)
    eval_source: str = "clinvar"
    model: str = "gbm"
    seed: int = 42
    drop_groups: tuple[str, ...] = ()
    allow_proxy_leak: bool = False
    inner_val_fraction: float = 0.2
    n_bootstrap: int = 10_000
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def arm(self) -> str:
        """Everything that defines an arm; seeds of one arm share this string.

        Includes the ablation: a run with gnomAD and a run without it are
        different arms, never replicates of each other.
        """
        ablation = ("__drop-" + "-".join(sorted(self.drop_groups))
                    if self.drop_groups else "")
        return "train-" + ("+".join(sorted(self.train_sources)) or "none") + ablation

    @property
    def slug(self) -> str:
        return f"{self.arm}__{self.model}__seed{self.seed}"


@dataclass
class CellResult:
    config: CellConfig
    per_gene: list[dict[str, Any]]
    predictions: pd.DataFrame
    val_predictions: pd.DataFrame
    provenance: dict[str, Any]
    runtime_s: float
    skipped: dict[str, str] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        # A held-out fold under 50 variants (PMS2, at n=21) is reported but kept
        # out of the headline mean, matching v1's "scoreable genes" convention.
        scoreable = [row for row in self.per_gene if row.get("n", 0) >= 50]
        return {
            "cell": self.config.slug,
            "arm": self.config.arm,
            "sources": list(self.config.sources),
            "train_sources": list(self.config.train_sources),
            "eval_source": self.config.eval_source,
            "model": self.config.model,
            "seed": self.config.seed,
            "drop_groups": list(self.config.drop_groups),
            "genes_evaluated": [row["gene"] for row in self.per_gene],
            "genes_skipped": self.skipped,
            "mean_roc_auc_all": float(np.nanmean(
                [row["roc_auc"] for row in self.per_gene]
            )) if self.per_gene else float("nan"),
            "mean_roc_auc_scoreable": float(np.nanmean(
                [row["roc_auc"] for row in scoreable]
            )) if scoreable else float("nan"),
            "runtime_s": round(self.runtime_s, 1),
            "provenance": self.provenance,
        }


def _inner_split(
    frame: pd.DataFrame,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Group-disjoint inner-validation split over the training genes.

    Grouped by ``uniprot:position`` for the same reason the outer split is:
    two variants at one residue must not sit on opposite sides of any
    partition boundary.
    """
    groups = group_keys(frame)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)

    n_val = max(1, int(round(len(unique) * fraction)))
    val_groups = set(unique[:n_val])
    is_val = np.array([group in val_groups for group in groups])
    return np.where(~is_val)[0], np.where(is_val)[0]


def run_cell(
    table: pd.DataFrame,
    config: CellConfig,
    feature_columns: Sequence[str],
    dataset_path: Path | str,
    out_dir: Path | str,
    sequences: Mapping[str, str] | None = None,
) -> CellResult:
    """Train and evaluate one cell leave-one-gene-out. Writes artefacts.

    `sequences` is required only for models in :data:`SEQUENCE_WINDOW_MODELS`.
    """
    started = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_column = f"label__{config.eval_source}"
    if eval_column not in table.columns:
        raise ValueError(
            f"Cell {config.slug}: the table has no '{config.eval_source}' labels "
            f"to evaluate against. Every build must include {config.eval_source}, "
            "even for arms that do not train on it — it defines the test set."
        )

    work = table.assign(
        _train=resolve_labels(table, config.train_sources),
        _eval=table[eval_column],
    )
    work = work[work["_train"].notna() | work["_eval"].notna()].reset_index(drop=True)

    columns = resolve_ablation(
        list(feature_columns), list(config.drop_groups),
        allow_proxy_leak=config.allow_proxy_leak,
    )
    # Leave-one-gene-out makes any gene-constant column a gene-identity label
    # (gnomAD's pLI / o-e missense / missense-Z). Measured on the FULL table so
    # the feature schema is identical across arms, whatever they train on.
    columns = drop_gene_constant(table, columns)
    if not columns:
        raise ValueError(f"Cell {config.slug}: ablation removed every feature.")

    per_gene: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    val_predictions: list[pd.DataFrame] = []
    held_out_all: list[str] = []
    skipped: dict[str, str] = {}

    # Say what is being trained on BEFORE any metric exists. A metric computed on
    # the wrong table looks exactly like a metric computed on the right one; a
    # gene with zero rows or a hash you do not recognise does not.
    from vpdl.provenance import file_sha256
    logger.info("%s | dataset sha256=%s… train on %s, score on %s",
                config.slug, file_sha256(dataset_path)[:12],
                "+".join(config.train_sources), config.eval_source)
    for gene_name, rows in work.groupby("gene"):
        logger.info(
            "%s | %-5s train-labels=%5d (path=%d)   eval-labels=%4d (path=%d)",
            config.slug, gene_name,
            int(rows["_train"].notna().sum()), int((rows["_train"] == 1).sum()),
            int(rows["_eval"].notna().sum()), int((rows["_eval"] == 1).sum()),
        )

    eval_genes = sorted(
        str(g) for g in work.loc[work["_eval"].notna(), "gene"].dropna().unique()
        if str(g).strip()
    )

    for split_index, gene in enumerate(eval_genes):
        train_positions = np.where((work["gene"] != gene) & work["_train"].notna())[0]
        test_positions = np.where((work["gene"] == gene) & work["_eval"].notna())[0]
        assert_no_group_straddle(work, train_positions, test_positions)

        train_frame = work.iloc[train_positions].reset_index(drop=True)
        test_frame = work.iloc[test_positions].reset_index(drop=True)

        # Recorded BEFORE the skip check: the split hash identifies the
        # evaluation set, so an arm that cannot train one fold still shares its
        # test set with the other arms. The skip itself is reported separately.
        held_out_all.extend(variant_keys(test_frame).tolist())

        # A fold with nothing to learn from is reported, not crashed on and not
        # silently dropped. A DMS-only arm hits this by construction on MSH2:
        # every DMS label is MSH2, so holding it out leaves nothing to train on.
        if len(train_frame) < 20 or train_frame["_train"].nunique() < 2:
            reason = (f"untrainable: {len(train_frame)} training rows, "
                      f"{train_frame['_train'].nunique()} class(es)")
            logger.warning("%s | holdout=%s SKIPPED — %s", config.slug, gene, reason)
            skipped[gene] = reason
            continue

        # Keyed on the held-out GENE, not the loop position: a gene dropping out
        # of the panel would otherwise shift every subsequent split's seed and
        # silently change results for a cell that did not change.
        split_key = gene
        inner_train, inner_val = _inner_split(
            train_frame, config.inner_val_fraction,
            derive_seed(config.seed, split_key),
        )

        matrix = build_feature_matrix(
            train_frame.iloc[inner_train], columns,
            labels=train_frame.iloc[inner_train]["_train"].to_numpy(),
        )
        X_train = matrix.X
        X_val = matrix.transform(train_frame.iloc[inner_val])
        X_test = matrix.transform(test_frame)

        # Training and inner-validation labels come from the arm's own sources,
        # so a DMS-only arm's threshold is chosen on DMS labels — not on clinical
        # ones, which would leak the evaluation signal into it. That makes MCC
        # partly a threshold-transfer measure; ROC-AUC is the fair cross-arm
        # comparison because it needs no threshold.
        y_train = train_frame.iloc[inner_train]["_train"].to_numpy(dtype=int)
        y_val = train_frame.iloc[inner_val]["_train"].to_numpy(dtype=int)
        y_test = test_frame["_eval"].to_numpy(dtype=int)

        if config.model in SEQUENCE_WINDOW_MODELS:
            if not sequences:
                raise ValueError(
                    f"Model {config.model!r} consumes sequence windows, so "
                    "run_cell needs `sequences={uniprot_id: sequence}`. Without "
                    "them the wild-type and variant windows cannot be built, "
                    "and a silently identical pair is regression landmine L12."
                )
            windows_train = _windows_for(train_frame.iloc[inner_train], sequences)
            windows_val = _windows_for(train_frame.iloc[inner_val], sequences)
            windows_test = _windows_for(test_frame, sequences)

            model = build_model(
                config.model, seed=config.seed, split_index=split_key,
                n_tabular_features=X_train.shape[1], **config.model_kwargs,
            )
            model.fit(
                windows_train, y_train, tabular=X_train,
                windows_val=windows_val, y_val=y_val, tabular_val=X_val,
            )
            val_scores = model.predict_proba(windows_val, tabular=X_val)
            test_scores = model.predict_proba(windows_test, tabular=X_test)
        else:
            model = build_model(
                config.model,
                seed=config.seed,
                split_index=split_key,
                **({"n_features": X_train.shape[1]}
                   if config.model in {"mlp"} else {}),
                **config.model_kwargs,
            )
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

            val_scores = model.predict_proba(X_val)
            test_scores = model.predict_proba(X_test)

        # Threshold from inner validation only. Selecting it on `test_scores`
        # would inflate every threshold-dependent metric below.
        threshold, _ = best_threshold_by_mcc(y_val, val_scores)

        report = evaluation_report(
            y_test, test_scores, threshold=threshold,
            n_bootstrap=config.n_bootstrap, seed=config.seed,
            check_orientation=False,   # reported, not enforced, per fold
        )
        row = {"gene": gene, "split_index": split_index} | report.as_dict()
        per_gene.append(row)

        keys = variant_keys(test_frame)
        predictions.append(pd.DataFrame({
            "cell": config.slug, "gene": gene, "variant_key": keys,
            "label": y_test, "score": test_scores, "threshold": threshold,
        }))
        val_predictions.append(pd.DataFrame({
            "cell": config.slug, "gene": gene,
            "variant_key": variant_keys(train_frame.iloc[inner_val]),
            "label": y_val, "score": val_scores,
        }))

        logger.info(
            "%s | holdout=%s n=%d ROC-AUC=%.4f MCC=%.4f",
            config.slug, gene, report.n, report.roc_auc, report.mcc,
        )

    predictions_frame = (pd.concat(predictions, ignore_index=True)
                         if predictions else pd.DataFrame())
    val_frame = (pd.concat(val_predictions, ignore_index=True)
                 if val_predictions else pd.DataFrame())

    provenance = provenance_record(
        dataset_path=dataset_path,
        feature_columns=columns,
        held_out_keys=held_out_all,
        sources=config.sources,
    ) | {"feature_schema_columns": schema_hash(columns)}

    result = CellResult(
        config=config, per_gene=per_gene, predictions=predictions_frame,
        val_predictions=val_frame, provenance=provenance,
        runtime_s=time.time() - started, skipped=skipped,
    )

    pd.DataFrame(per_gene).to_csv(out_dir / f"results_{config.slug}.csv", index=False)
    predictions_frame.to_csv(out_dir / f"predictions_{config.slug}.csv", index=False)
    val_frame.to_csv(out_dir / f"valpreds_{config.slug}.csv", index=False)
    (out_dir / f"summary_{config.slug}.json").write_text(
        json.dumps(result.summary(), indent=2, default=str)
    )
    return result


def run_sweep(
    table: pd.DataFrame,
    dataset_path: Path | str,
    build_sources: Sequence[str],
    train_source_sets: Sequence[tuple[str, ...]],
    feature_columns: Sequence[str],
    models: Sequence[str] = ("gbm",),
    seeds: Sequence[int] = (42, 43, 44),
    out_dir: Path | str = "runs",
    drop_groups: Sequence[tuple[str, ...]] = ((),),
    eval_source: str = "clinvar",
    sequences: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Fill the matrix over ONE table: train-source set x model x ablation x seed.

    One table, so every arm shares the dataset hash, the feature schema and the
    held-out test set — the provenance gate passes because the arms genuinely
    are comparable, not because it was loosened.

    Three seeds is the floor, not a default to lower. v1 measured MCC standard
    deviations up to 0.056 across seeds of one arm, so any single-seed
    difference smaller than that is noise being read as a finding.
    """
    if len(seeds) < 3:
        logger.warning(
            "Running %d seed(s). Differences below the seed spread are not "
            "interpretable; three is this project's floor.", len(seeds)
        )

    rows: list[dict[str, Any]] = []
    for train_sources in train_source_sets:
        for model in models:
            for groups in drop_groups:
                for seed in seeds:
                    config = CellConfig(
                        sources=tuple(build_sources),
                        train_sources=tuple(train_sources),
                        eval_source=eval_source,
                        model=model, seed=seed, drop_groups=tuple(groups),
                    )
                    result = run_cell(table, config, feature_columns,
                                      dataset_path, out_dir, sequences=sequences)
                    rows.append(result.summary())

    frame = pd.DataFrame(rows)
    frame.to_csv(Path(out_dir) / "sweep_summary.csv", index=False)
    return frame
