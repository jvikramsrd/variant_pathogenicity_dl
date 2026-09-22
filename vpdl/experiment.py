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
           "apply_train_caps", "SEQUENCE_WINDOW_MODELS", "FRAME_MODELS",
           "MATRIX_WIDTH_MODELS"]

# Models consuming residue windows rather than a feature matrix. They take a
# different fit signature, so run_cell branches rather than pretending the
# interfaces match. The DL-branch window baselines (vpdl/models/seqwin.py) take
# exactly the bilstm arm's inputs.
SEQUENCE_WINDOW_MODELS = frozenset({"bilstm", "aa_mlp", "cnn", "bilstm_attn", "transformer"})

# Models consuming variant rows + sequences (a protein language model trained
# end to end), plus the tabular matrix as a side input.
FRAME_MODELS = frozenset({"plm_finetune"})

# Matrix models whose constructor needs the feature width.
MATRIX_WIDTH_MODELS = frozenset({"mlp", "fusion"})


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
    # ((source, max_labels), ...): subsample the rows labelled ONLY by that
    # source. The dose-response control for pooling — see apply_train_caps.
    train_caps: tuple[tuple[str, int], ...] = ()
    model: str = "gbm"
    seed: int = 42
    drop_groups: tuple[str, ...] = ()
    allow_proxy_leak: bool = False
    inner_val_fraction: float = 0.2
    n_bootstrap: int = 10_000
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    # DL branch. Defaults reproduce the original cell exactly (and its slug).
    # split: vpdl.dl.splits scheme. embedding_blocks: feature-store specs
    # "family/model_tag/key:representation" appended to the matrix.
    # score_rows: also score held-out-gene rows beyond the labelled test set
    # ("functional" = rows with validation-only assay values, "all" = every
    # row), written to scores_<slug>.csv. tag: distinguishes two configurations
    # of one model (model_kwargs are not otherwise in the slug).
    split: str = "logo"
    embedding_blocks: tuple[str, ...] = ()
    score_rows: str = "none"
    tag: str = ""

    @property
    def arm(self) -> str:
        """Everything that defines an arm; seeds of one arm share this string.

        Includes the ablation: a run with gnomAD and a run without it are
        different arms, never replicates of each other.
        """
        ablation = ("__drop-" + "-".join(sorted(self.drop_groups))
                    if self.drop_groups else "")
        caps = "".join(f"__cap-{source}{limit}"
                       for source, limit in sorted(self.train_caps))
        split = f"__split-{self.split}" if self.split != "logo" else ""
        blocks = "".join(f"__emb-{_block_token(b)}" for b in self.embedding_blocks)
        return ("train-" + ("+".join(sorted(self.train_sources)) or "none")
                + caps + ablation + split + blocks)

    @property
    def slug(self) -> str:
        model = f"{self.model}-{self.tag}" if self.tag else self.model
        return f"{self.arm}__{model}__seed{self.seed}"


def _block_token(spec: str) -> str:
    """Filename-safe slug token for an embedding-block spec."""
    import re

    if spec.startswith("perfold="):
        location, _, representation = spec.rpartition(":")
        return re.sub(r"[^A-Za-z0-9_.+@=-]", "-",
                      f"perfold-{Path(location[len('perfold='):]).stem}-{representation}")
    location, _, representation = spec.partition(":")
    parts = location.split("/")
    short = f"{parts[1]}-{parts[2][:8]}" if len(parts) == 3 else location
    return re.sub(r"[^A-Za-z0-9_.+@=-]", "-", f"{short}-{representation}")


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
            "train_caps": dict(self.config.train_caps),
            "eval_source": self.config.eval_source,
            "model": self.config.model,
            "seed": self.config.seed,
            "drop_groups": list(self.config.drop_groups),
            "split": self.config.split,
            "embedding_blocks": list(self.config.embedding_blocks),
            "tag": self.config.tag,
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


def apply_train_caps(
    table: pd.DataFrame,
    train_label: pd.Series,
    train_sources: Sequence[str],
    caps: Sequence[tuple[str, int]],
    seed: int,
) -> pd.Series:
    """Subsample the rows whose training label comes ONLY from a capped source.

    Pooling ClinVar with the full MSH2 DMS assay cost 0.09 AUROC on MLH1 in the
    first v2 grid (2026-09-21). Two explanations predict that: the assay's
    labels mean something different from clinical ones, or 16,749 MSH2 rows
    simply swamp 276 clinical ones. Capping the assay at several sizes
    separates them — if even a few hundred DMS labels hurt, it is the labels.

    Rows another training source also labels keep their label, and evaluation
    labels are never touched, so capped and uncapped arms still share one test
    set and remain directly comparable.
    """
    capped = train_label.copy()
    for source, limit in caps:
        if source not in train_sources:
            raise ValueError(
                f"Cannot cap '{source}': this arm does not train on it "
                f"({list(train_sources)})."
            )
        others = [f"label__{name}" for name in train_sources
                  if name != source and f"label__{name}" in table.columns]
        sole = table[f"label__{source}"].notna() & capped.notna()
        if others:
            sole &= table[others].isna().all(axis=1)

        positions = np.flatnonzero(sole.to_numpy())
        if len(positions) <= limit:
            logger.info("Cap %s=%d: only %d sole-source labels, nothing removed.",
                        source, limit, len(positions))
            continue
        rng = np.random.default_rng(derive_seed(seed, f"cap:{source}"))
        drop = rng.choice(positions, size=len(positions) - limit, replace=False)
        capped.iloc[drop] = np.nan
        logger.info("Cap %s=%d: %d sole-source training labels -> %d.",
                    source, limit, len(positions), limit)
    return capped


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
    dl_context: Any = None,
) -> CellResult:
    """Train and evaluate one cell leave-one-gene-out. Writes artefacts.

    `sequences` is required only for models in :data:`SEQUENCE_WINDOW_MODELS`
    and :data:`FRAME_MODELS`. `dl_context` (:class:`vpdl.dl.runner.DLContext`)
    supplies feature-store embeddings, functional-validation keys and export
    settings for the DL branch; None reproduces the original cell exactly.
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

    train_label = resolve_labels(table, config.train_sources)
    if config.train_caps:
        train_label = apply_train_caps(table, train_label, config.train_sources,
                                       config.train_caps, config.seed)
    work = table.assign(_train=train_label, _eval=table[eval_column])
    work = work[work["_train"].notna() | work["_eval"].notna()].reset_index(drop=True)

    columns = resolve_ablation(
        list(feature_columns), list(config.drop_groups),
        allow_proxy_leak=config.allow_proxy_leak,
    )
    # Leave-one-gene-out makes any gene-constant column a gene-identity label
    # (gnomAD's pLI / o-e missense / missense-Z). Measured on the FULL table so
    # the feature schema is identical across arms, whatever they train on.
    columns = drop_gene_constant(table, columns)
    # Sequence-window and PLM models have their own input; for them (and for
    # embedding-only arms) an empty tabular side is a legitimate arm.
    if not columns and not (config.embedding_blocks or config.model in SEQUENCE_WINDOW_MODELS
                            or config.model in FRAME_MODELS):
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

    from vpdl.dl.leakage import leakage_gate
    from vpdl.dl.splits import GENE_DISJOINT_SCHEMES, make_folds

    folds = make_folds(work, config.split, seed=config.seed)
    if config.score_rows not in ("none", "functional", "all"):
        raise ValueError(f"score_rows must be none|functional|all, got {config.score_rows!r}")
    if config.score_rows != "none" and config.split not in GENE_DISJOINT_SCHEMES:
        raise ValueError(f"score_rows={config.score_rows!r} needs a gene-disjoint split; "
                         f"under {config.split!r} the scored genes were trained on.")
    functional_keys = (set(dl_context.functional_keys) if dl_context is not None
                       and config.score_rows == "functional" else set())
    # Refuse to train through critical leakage (duplicates, straddling folds,
    # functional data reaching training, label-derived features).
    leakage = leakage_gate(work, folds, config.split, columns, config.train_sources,
                           config.eval_source,
                           validating_functional=config.score_rows == "functional",
                           functional_keys=functional_keys)
    if (config.embedding_blocks or config.model == "fusion") and dl_context is None:
        from vpdl.dl.runner import DLContext
        dl_context = DLContext()
    extra_scores: list[pd.DataFrame] = []

    for split_index, fold in enumerate(folds):
        train_positions, test_positions = fold.train_positions, fold.test_positions
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
            logger.warning("%s | holdout=%s SKIPPED — %s", config.slug, fold.name, reason)
            skipped[fold.name] = reason
            continue

        # Keyed on the held-out GENE (the fold name under leave-one-gene-out),
        # not the loop position: a gene dropping out of the panel would
        # otherwise shift every subsequent split's seed and silently change
        # results for a cell that did not change.
        split_key = fold.name
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
        model_kwargs = dict(config.model_kwargs)
        augment = None
        if dl_context is not None and (config.embedding_blocks or config.model == "fusion"):
            augment = dl_context.augmenter(config, columns, train_frame.iloc[inner_train],
                                           fold=fold)
            X_train = augment(train_frame.iloc[inner_train], X_train)
            X_val = augment(train_frame.iloc[inner_val], X_val)
            X_test = augment(test_frame, X_test)
            if config.model == "fusion":
                model_kwargs = augment.fusion_kwargs() | model_kwargs

        # Training and inner-validation labels come from the arm's own sources,
        # so a DMS-only arm's threshold is chosen on DMS labels — not on clinical
        # ones, which would leak the evaluation signal into it. That makes MCC
        # partly a threshold-transfer measure; ROC-AUC is the fair cross-arm
        # comparison because it needs no threshold.
        y_train = train_frame.iloc[inner_train]["_train"].to_numpy(dtype=int)
        y_val = train_frame.iloc[inner_val]["_train"].to_numpy(dtype=int)
        y_test = test_frame["_eval"].to_numpy(dtype=int)

        if config.model in SEQUENCE_WINDOW_MODELS or config.model in FRAME_MODELS:
            if not sequences:
                raise ValueError(
                    f"Model {config.model!r} consumes sequence windows, so "
                    "run_cell needs `sequences={uniprot_id: sequence}`. Without "
                    "them the wild-type and variant windows cannot be built, "
                    "and a silently identical pair is regression landmine L12."
                )

        if config.model in SEQUENCE_WINDOW_MODELS:
            windows_train = _windows_for(train_frame.iloc[inner_train], sequences)
            windows_val = _windows_for(train_frame.iloc[inner_val], sequences)
            windows_test = _windows_for(test_frame, sequences)

            model = build_model(
                config.model, seed=config.seed, split_index=split_key,
                n_tabular_features=X_train.shape[1], **model_kwargs,
            )
            model.fit(
                windows_train, y_train, tabular=X_train,
                windows_val=windows_val, y_val=y_val, tabular_val=X_val,
            )
            val_scores = model.predict_proba(windows_val, tabular=X_val)
            test_scores = model.predict_proba(windows_test, tabular=X_test)

            def score(frame, X):
                return model.predict_proba(_windows_for(frame, sequences), tabular=X)
        elif config.model in FRAME_MODELS:
            model = build_model(config.model, seed=config.seed, split_index=split_key,
                                n_tabular_features=X_train.shape[1], **model_kwargs)
            model.fit_frames(train_frame.iloc[inner_train], y_train,
                             train_frame.iloc[inner_val], y_val, tabular=X_train,
                             tabular_val=X_val, sequences=sequences)
            val_scores = model.predict_frames(train_frame.iloc[inner_val], X_val, sequences)
            test_scores = model.predict_frames(test_frame, X_test, sequences)

            def score(frame, X):
                return model.predict_frames(frame, X, sequences)
        else:
            model = build_model(
                config.model,
                seed=config.seed,
                split_index=split_key,
                **({"n_features": X_train.shape[1]}
                   if config.model in MATRIX_WIDTH_MODELS else {}),
                **model_kwargs,
            )
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

            val_scores = model.predict_proba(X_val)
            test_scores = model.predict_proba(X_test)

            def score(frame, X):
                return model.predict_proba(X)

        # Threshold from inner validation only. Selecting it on `test_scores`
        # would inflate every threshold-dependent metric below.
        threshold, _ = best_threshold_by_mcc(y_val, val_scores)

        # One row per held-out gene. Under leave-one-gene-out a fold IS one
        # gene, so this is the original per-fold row; under the family split a
        # fold holds two genes and each is still reported on its own.
        test_genes = test_frame["gene"].to_numpy()
        for gene in fold.test_genes:
            in_gene = test_genes == gene
            report = evaluation_report(
                y_test[in_gene], test_scores[in_gene], threshold=threshold,
                n_bootstrap=config.n_bootstrap, seed=config.seed,
                check_orientation=False,   # reported, not enforced, per fold
            )
            row = {"gene": gene, "split_index": split_index} | report.as_dict()
            if config.split != "logo":
                row["fold"] = fold.name
            per_gene.append(row)
            logger.info(
                "%s | holdout=%s n=%d ROC-AUC=%.4f MCC=%.4f",
                config.slug, gene, report.n, report.roc_auc, report.mcc,
            )

        keys = variant_keys(test_frame)
        fold_predictions = pd.DataFrame({
            "cell": config.slug, "gene": test_frame["gene"].to_numpy(), "variant_key": keys,
            "label": y_test, "score": test_scores, "threshold": threshold,
        })
        if config.split != "logo":
            fold_predictions["fold"] = fold.name
        predictions.append(fold_predictions)
        fold_val = pd.DataFrame({
            "cell": config.slug, "gene": train_frame.iloc[inner_val]["gene"].to_numpy()
            if config.split != "logo" else fold.name,
            "variant_key": variant_keys(train_frame.iloc[inner_val]),
            "label": y_val, "score": val_scores,
        })
        if config.split != "logo":
            fold_val["fold"] = fold.name
        val_predictions.append(fold_val)

        if config.score_rows != "none" or (dl_context is not None and dl_context.export_dir):
            from vpdl.dl.runner import score_extra_rows
            extra_scores.append(score_extra_rows(
                table, fold, config, model, matrix, augment, score, functional_keys,
                dl_context, out_dir, test_frame, X_test))

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
    if config.split != "logo" or config.embedding_blocks or dl_context is not None:
        # Recorded only for DL-branch cells, so original cells' summaries are
        # unchanged. feature_version pins the feature-store entries read.
        provenance |= {
            "split_scheme": config.split,
            "embedding_blocks": list(config.embedding_blocks),
            "feature_version": (dl_context.feature_version(config.embedding_blocks)
                                if dl_context is not None else None),
            "leakage": {"critical": len(leakage.critical),
                        "warnings": sum(f.severity == "warning" for f in leakage.findings)},
        }
    if extra_scores:
        pd.concat(extra_scores, ignore_index=True).to_csv(
            out_dir / f"scores_{config.slug}.csv", index=False)

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
