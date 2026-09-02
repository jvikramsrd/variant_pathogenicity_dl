"""Two-stage transfer learning with ESM-embedding pretraining (Phases 0–3 DL).

Stage 1 — **pretrain**: fit the classification head on the broad multi-source
dataset covering **all 80 panel genes**, using their frozen ESM-2 embedding
features (+ published-model priors, never DMS-derived supervision columns).
In ``leave_gene_out`` mode every MMR gene is excluded from pretraining so no
MMR variant contaminates the general representation before transfer.

Stage 2 — **finetune**: warm-start the same architecture from the pretrained
checkpoint on dedicated MMR data (ClinVar/PG-clinical labels only; PMS2
homology gate already applied upstream) and evaluate leave-one-MMR-gene-out
with bootstrap CIs, validation-tuned thresholds, and mandatory ablations:

* ESM branch only          (branch baseline)
* prior branch only        (branch baseline)
* concat fusion            (plan-default head)
* GateWave gated fusion    (MVmamba base-paper head)

The checkpoint stores its exact feature-column order; fine-tuning refuses to
load a checkpoint whose schema differs (a different order invalidates transfer
even when dimensions happen to match).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .esm_extractor import extract_features_cached
from .fusion import BranchHead, ConcatFusionHead, GateWaveFusionHead
from .train import set_global_seed

logger = logging.getLogger(__name__)

#: Leakage-safe prior columns shared by both stages.  DMS-derived columns and
#: the functional-assay validation-only columns
#: (src.mmr_dataset.FUNCTIONAL_ASSAY_VALIDATION_ONLY_COLS: CIMRA OddsPath,
#: MaveDB scores) are deliberately absent: they are supervision or held-out
#: orthogonal evidence, never training features.
#:
#: Every other source this project acquires (src/extended_builder.py) IS
#: wired in here deliberately -- a column joined onto the table but missing
#: from this tuple is silently invisible to the model (this bit us once
#: already with gnomAD AF; fixed below, and every new source added since is
#: included from the start):
#:   * gnomad_log10_af, acmg_ba1/bs1/pm2   -- variant-level AF (opt-in join)
#:   * gnomad_pli, gnomad_oe_lof/mis,
#:     gnomad_mis_z, gnomad_syn_z          -- gene-level constraint (opt-in)
#:   * af_plddt, af_disordered             -- AlphaFold structural confidence (opt-in)
#:   * in_interpro_domain                  -- InterPro domain/family call (opt-in)
#:   * is_functional_site                  -- UniProt active/binding site, PTM, ... (opt-in)
TRANSFER_PRIOR_COLS: Tuple[str, ...] = (
    "am_pathogenicity", "in_domain",
    "gnomad_log10_af", "acmg_ba1", "acmg_bs1", "acmg_pm2",
    "gnomad_pli", "gnomad_oe_lof", "gnomad_oe_mis", "gnomad_mis_z", "gnomad_syn_z",
    "af_plddt", "af_disordered",
    "in_interpro_domain",
    "is_functional_site",
)
ZS_PREFIX = "zs_"
RANK_PREFIX = "rank_"
CONSENSUS_COL = "consensus_rank"

#: gnomAD **gene-level** constraint metrics. These are constant across every
#: variant of a gene, so under leave-one-gene-out they are not features at all
#: -- they are a 5-dimensional gene identifier. With three training genes a
#: head can memorise "this constraint vector -> this gene's base rate" (MLH1 is
#: 82% pathogenic, MSH2 40%) and exploit prevalence instead of variant-level
#: evidence; on the held-out gene the vector takes a value never seen in
#: training, so whatever was memorised maps arbitrarily. Drop them for
#: cross-gene evaluation; they remain legitimate for within-gene work and for
#: the broad panel, where they genuinely vary.
GENE_CONSTANT_PRIOR_COLS: Tuple[str, ...] = (
    "gnomad_pli", "gnomad_oe_lof", "gnomad_oe_mis", "gnomad_mis_z",
    "gnomad_syn_z",
)

#: gnomAD allele-frequency columns and the ACMG flags derived from them.
#: :func:`src.gnomad.add_frequency_flags` defines acmg_ba1 as ``AF > 0.05``,
#: acmg_bs1 as ``AF > bs1_af`` and acmg_pm2 as ``AF < pm2_af``, so any label
#: minted from allele frequency is reproduced exactly by these columns.
AF_DERIVED_PRIOR_COLS: Tuple[str, ...] = (
    "gnomad_log10_af", "acmg_ba1", "acmg_bs1", "acmg_pm2",
)


def assert_af_quarantine(prior_cols: Sequence[str],
                         af_labels_active: bool) -> None:
    """Refuse a configuration that both labels *and* features on frequency.

    Minting benign labels from gnomAD allele frequency while feeding allele
    frequency as a feature makes ``acmg_bs1 == label`` by construction on
    every minted row -- the same target leakage as ``dms_bin_median ==
    1 - label`` (docs/PAPER.md Finding 2). Enforced here rather than left to
    convention, because the failure is invisible in every metric.
    """
    if not af_labels_active:
        return
    offenders = sorted(set(prior_cols) & set(AF_DERIVED_PRIOR_COLS))
    if offenders:
        raise ValueError(
            "AF quarantine violated: allele-frequency-derived labels are "
            f"active while these AF-derived features are in the feature set: "
            f"{offenders}. Drop them, or disable the AF labels.")


#: Sign applied before ranking so that every score reads "higher = more
#: pathogenic". The four negated columns are likelihood-style scores where a
#: higher value means *more fit*. Signs were fitted empirically on the broad
#: panel's clinical labels with all four MMR genes removed (see
#: ``scripts/benchmark_published_predictors.py::fit_orientations``); no
#: column's calibration ROC-AUC sat closer to 0.5 than 0.21, so none of these
#: is a coin flip. Sign only affects :data:`CONSENSUS_COL` -- for the
#: individual ``rank_*`` features a monotone flip is irrelevant to the head.
PRIOR_SCORE_SIGN: Dict[str, int] = {
    "zs_eve": -1, "zs_gemme": -1, "zs_trancepteve_l": -1, "zs_esm1b": -1,
}

MMR_GENES: Tuple[str, ...] = ("MLH1", "MSH2", "MSH6", "PMS2")


# --------------------------------------------------------------------------- #
# Feature construction
# --------------------------------------------------------------------------- #
def rankable_score_columns(df: pd.DataFrame) -> List[str]:
    """Continuous pathogenicity-score columns worth rank-normalising.

    The published per-variant scores only: every ``zs_*`` column plus
    AlphaMissense. Binary flags, gene-level constants and structural
    annotations are excluded -- a percentile rank of a 0/1 flag is noise, and
    ranking a gene-constant column collapses it to 0.5 everywhere.
    """
    return [c for c in df.columns
            if (c.startswith(ZS_PREFIX) or c == "am_pathogenicity")
            and not c.startswith(RANK_PREFIX)]


def add_within_gene_rank_features(
    df: pd.DataFrame,
    score_cols: Optional[Sequence[str]] = None,
    add_consensus: bool = True,
) -> pd.DataFrame:
    """Add within-gene percentile ranks of each published score, + a consensus.

    Why this exists
    ---------------
    Every published predictor has its own score scale, and — the part that
    actually breaks leave-one-gene-out — *the same predictor has a different
    score distribution in every gene*. A head fitted on two or three genes
    therefore learns decision boundaries in units that do not exist in the
    fourth. Measured on this panel, a plain skip-NaN mean of the ranked
    scores, with no training whatsoever, out-scores the trained head on every
    MMR gene; on MSH2 it also beats the best single published predictor
    (0.916 vs 0.896 ROC-AUC). Converting each score to a within-gene
    percentile puts every gene on one comparable scale and hands the head a
    representation that transfers.

    ``consensus_rank`` is the skip-NaN mean of the ranked scores. Skipping
    rather than imputing matters: coverage runs 68-89% per predictor per gene,
    and averaging only what was actually scored beats filling a median into
    the gap.

    Leakage contract
    ----------------
    Ranks are computed over **every variant of the gene in *df*** — labelled
    and VUS alike — and never touch the label column. No label information
    crosses the split. It is, deliberately, *transductive* in the score
    distribution: scoring a held-out gene requires that gene's variant set to
    rank against. That matches how the tool is actually used (a gene's VUS are
    scored as a batch) and it is the same assumption gene-specific calibration
    makes, but it must be stated in any write-up — pass the full per-gene
    variant table, not just the labelled rows, or the ranks shift between
    training and inference.
    """
    cols = list(score_cols) if score_cols is not None else rankable_score_columns(df)
    cols = [c for c in cols if c in df.columns]
    if not cols:
        logger.warning("No rankable score columns found; returning unchanged.")
        return df
    out = df.copy()
    ranked: Dict[str, pd.Series] = {}
    for c in cols:
        signed = PRIOR_SCORE_SIGN.get(c, 1) * pd.to_numeric(out[c], errors="coerce")
        ranked[c] = signed.groupby(out["gene"]).rank(pct=True)
        out[f"{RANK_PREFIX}{c}"] = ranked[c]
    if add_consensus:
        out[CONSENSUS_COL] = pd.DataFrame(ranked).mean(axis=1, skipna=True)
    logger.info("Within-gene rank features: %d columns%s over genes %s.",
                len(cols), " + consensus" if add_consensus else "",
                sorted(out["gene"].unique()))
    return out


def prior_columns_of(df: pd.DataFrame,
                     drop_gene_constant: bool = False) -> List[str]:
    """All leakage-safe prior columns present in *df*.

    *drop_gene_constant* removes :data:`GENE_CONSTANT_PRIOR_COLS` — set it for
    any cross-gene (leave-one-gene-out) evaluation, where those columns encode
    gene identity rather than variant evidence.
    """
    cols = [c for c in df.columns
            if c in TRANSFER_PRIOR_COLS
            or c.startswith(ZS_PREFIX)
            or c.startswith(RANK_PREFIX)
            or c == CONSENSUS_COL]
    if drop_gene_constant:
        cols = [c for c in cols if c not in GENE_CONSTANT_PRIOR_COLS]
    return cols


def prior_impute_values(df: pd.DataFrame,
                        columns: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Per-column median fill values for the prior branch of *df*.

    Split out from :func:`prior_matrix` so a stage can compute its fill values
    once, persist them in a checkpoint, and hand the *same* values to a later
    stage. Columns that are entirely missing get 0.0, matching the historical
    ``.fillna(raw.median()).fillna(0.0)`` behaviour.
    """
    cols = list(columns) if columns is not None else prior_columns_of(df)
    base_cols = [c for c in cols if not c.startswith("is_missing_") and c in df.columns]
    raw = df[base_cols].apply(pd.to_numeric, errors="coerce")
    medians = raw.median()
    return {c: (float(medians[c]) if pd.notna(medians[c]) else 0.0) for c in base_cols}


def prior_matrix(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    impute_values: Optional[Mapping[str, float]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Median-imputed numeric prior matrix + missingness indicator flags.

    Parameters
    ----------
    columns:
        Full ordered column specification.  Entries starting with
        ``is_missing_`` name *derived* missingness flags (never source
        columns); everything else must exist in *df*.  When ``None``, all
        leakage-safe prior columns of *df* are used in values-then-flags
        order.
    impute_values:
        Fill value per source column, normally carried over from the stage
        that fitted the model (see :func:`prior_impute_values`).  Columns
        absent from the mapping fall back to *df*'s own median.

        This matters more than it looks. Most ``zs_*`` prior columns are
        missing for ~99% of rows, so the imputed constant *is* the column for
        almost every variant. Recomputing the median per call meant stage 1
        pretrained the head on the 80-gene panel's medians while stage 2
        warm-started it onto the four MMR genes' medians -- the same weights
        applied to a differently-centred feature. Passing the pretraining
        values through keeps warm-starting meaningful.

    Returns ``(X, ordered_column_names_including_missing_flags)``.
    """
    cols = list(columns) if columns is not None else prior_columns_of(df)
    if not cols:
        raise ValueError("No non-DMS prior columns available.")
    base_cols = [c for c in cols if not c.startswith("is_missing_")]
    if not base_cols:
        raise ValueError("Prior column spec contains only missingness flags.")
    missing_cols = [c for c in base_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Table lacks required prior column(s): {missing_cols[:5]} "
            f"({len(missing_cols)} total). Rebuild the table or re-pretrain "
            "with a matching feature schema.")
    raw = df[base_cols].apply(pd.to_numeric, errors="coerce")
    missing = raw.isna().astype(np.float32)
    missing.columns = [f"is_missing_{c}" for c in base_cols]
    fill = raw.median()
    if impute_values:
        supplied = {c: float(v) for c, v in impute_values.items() if c in base_cols}
        if supplied:
            fill = fill.copy()
            for c, v in supplied.items():
                fill[c] = v
            logger.info("Prior imputation: reused %d/%d fill values from the "
                        "fitting stage.", len(supplied), len(base_cols))
    values = raw.fillna(fill).fillna(0.0)
    combined = pd.concat([values, missing], axis=1)
    if columns is not None:
        # Enforce the stored ordering exactly; an unknown entry here means
        # the schema drifted between stages.
        combined = combined[cols]
    return combined.to_numpy(np.float32), list(combined.columns)


def esm_branch_matrix(
    df: pd.DataFrame,
    sequence_by_gene: Dict[str, str],
    model_name: str,
    processed_dir: Path,
    device: torch.device,
    batch_size: int = 8,
    overwrite_cache: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Stack cached ESM+PLLR feature blocks for every gene present in *df*."""
    blocks: List[np.ndarray] = []
    metas: List[pd.DataFrame] = []
    for gene in sorted(df["gene"].unique()):
        sub = df[df["gene"] == gene].sort_values(["position", "mut_aa"])
        seq = sequence_by_gene.get(gene)
        if not seq:
            logger.warning("%s: no canonical sequence available; skipped.", gene)
            continue
        feats, meta = extract_features_cached(
            sub.reset_index(drop=True), seq, gene=gene, model_name=model_name,
            processed_dir=processed_dir, batch_size=batch_size, device=device,
            overwrite=overwrite_cache, extra_features=None)
        blocks.append(feats.astype(np.float32))
        metas.append(meta)
    if not blocks:
        raise ValueError("No ESM features extracted for any requested gene.")
    return np.vstack(blocks), pd.concat(metas, ignore_index=True)


def align_rows(meta: pd.DataFrame, target_keys: pd.DataFrame) -> np.ndarray:
    """Boolean mask selecting *meta* rows present in *target_keys* frame."""
    keys = set(zip(target_keys["gene"], target_keys["position"].astype(int),
                   target_keys["wt_aa"], target_keys["mut_aa"]))
    return np.array([
        (g, int(p), w, m) in keys
        for g, p, w, m in zip(meta["gene"], meta["position"],
                              meta["wt_aa"], meta["mut_aa"])
    ], dtype=bool)


# --------------------------------------------------------------------------- #
# Training configuration / primitives
# --------------------------------------------------------------------------- #
@dataclass
class TransferConfig:
    epochs: int = 60
    patience: int = 10
    lr: float = 3e-4
    finetune_lr: float = 3e-5
    weight_decay: float = 1e-2
    batch_size: int = 256
    dropout: float = 0.15
    clinical_weight: float = 5.0
    seed: int = 42
    n_bootstrap: int = 2000          # headline CIs; plan asks 10k for final runs
    threshold_grid: int = 1001

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


ARCHITECTURES: Dict[str, type] = {
    "esm": BranchHead,           # branch-only baselines
    "priors": BranchHead,
    "concat": ConcatFusionHead,
    "gatewave": GateWaveFusionHead,
}


def build_model(arch: str, dims: Sequence[int], hidden_dim: int = 256,
                dropout: float = 0.15) -> torch.nn.Module:
    """Instantiate one of the four benchmarked architectures."""
    if arch in ("esm", "priors"):
        return BranchHead(int(dims[0]), hidden_dim=hidden_dim, dropout=dropout)
    if arch == "concat":
        return ConcatFusionHead(tuple(dims), shared_dim=min(128, hidden_dim),
                                dropout=dropout)
    if arch == "gatewave":
        return GateWaveFusionHead(tuple(dims), shared_dim=min(128, hidden_dim),
                                  dropout=dropout)
    raise ValueError(f"Unknown architecture '{arch}'.")


def predict_logits(model: torch.nn.Module,
                   inputs: Sequence[np.ndarray],
                   device: torch.device, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    tensors = [torch.as_tensor(x, dtype=torch.float32) for x in inputs]
    n = len(tensors[0])
    out = np.empty(n, dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            chunk = [t[start:start + batch_size].to(device) for t in tensors]
            out[start:start + batch_size] = model(*chunk).cpu().numpy()
    return out


def fit_head(
    model: torch.nn.Module,
    Xs_train: Sequence[np.ndarray],
    y_train: np.ndarray,
    Xs_val: Sequence[np.ndarray],
    y_val: np.ndarray,
    sample_weights: Optional[np.ndarray],
    lr: float,
    epochs: int,
    patience: int,
    weight_decay: float,
    batch_size: int,
    device: torch.device,
    monitor_mask: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Tuple[torch.nn.Module, int]:
    """Generic head trainer with early stopping on val ROC-AUC."""
    from sklearn.metrics import roc_auc_score
    from .calibration import expit

    set_global_seed(seed)
    model.to(device)
    n_pos = max(1, int((y_train == 1).sum()))
    n_neg = max(1, int((y_train == 0).sum()))
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32)

    tensors_tr = [torch.as_tensor(x, dtype=torch.float32) for x in Xs_train]
    y_t = torch.as_tensor(y_train, dtype=torch.float32)
    w_t = (torch.ones(len(y_train)) if sample_weights is None
           else torch.as_tensor(sample_weights, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)

    val_inputs = [torch.as_tensor(x, dtype=torch.float32).to(device) for x in Xs_val]
    y_val_np = np.asarray(y_val, dtype=np.float32)
    active = (np.asarray(monitor_mask, dtype=bool) if monitor_mask is not None
              else np.ones(len(y_val_np), dtype=bool))

    best_auc, best_state, best_epoch, left = -np.inf, None, -1, patience
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(y_t))
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(*(t[idx].to(device) for t in tensors_tr))
            y_dev = y_t[idx].to(device)
            cw = torch.where(y_dev > 0.5, pos_weight,
                             torch.ones_like(pos_weight))
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y_dev, reduction="none")
                * w_t[idx].to(device) * cw).mean()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            val_logits = model(*val_inputs).cpu().numpy()
        try:
            use = active if np.unique(y_val_np[active]).size > 1 \
                else np.ones(len(y_val_np), dtype=bool)
            val_auc = float(roc_auc_score(y_val_np[use], expit(val_logits)[use]))
        except ValueError:
            val_auc = float("nan")
        if not np.isfinite(val_auc):
            val_auc = -np.inf
        if val_auc > best_auc:
            best_auc, best_epoch, left = val_auc, epoch + 1, patience
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            left -= 1
            if left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, max(best_epoch, 0)


# --------------------------------------------------------------------------- #
# Stage 1 — pretraining over the broad panel
# --------------------------------------------------------------------------- #
def select_stage_rows(df: pd.DataFrame, stage: str, mode: str,
                      holdout_gene: Optional[str]) -> pd.DataFrame:
    """Anti-circularity row selection applied before any fitting."""
    if stage == "pretrain":
        out = df.copy()
        if mode == "leave_gene_out":
            out = out[~out["gene"].isin(MMR_GENES)]
        return out.reset_index(drop=True)
    # Fine-tune: clinical labels only (ClinVar / ProteinGym-clinical), MMR
    # genes only; DMS-only and functional-assay values never become targets.
    out = df[df["gene"].isin(MMR_GENES)].copy()
    if len(out):
        if "label_source" in out.columns:
            clin = out["label_source"].isin(["clinvar", "pg_clinical"]).to_numpy()
        elif {"clinvar_label", "clinical_label"} <= set(out.columns):
            clin = (out["clinvar_label"].notna()
                    | out["clinical_label"].notna()).to_numpy()
        else:
            raise ValueError("Fine-tuning table lacks label-source columns.")
        out = out.loc[clin]
    if mode == "leave_gene_out":
        if holdout_gene is None:
            raise ValueError("leave_gene_out mode requires --holdout_gene.")
        out = out[out["gene"] != holdout_gene]
    return out.reset_index(drop=True)


def save_checkpoint(path: Path, model: torch.nn.Module, scaler_mean, scaler_scale,
                    feature_columns: Sequence[str], cfg_dict: Dict,
                    extra: Optional[Dict] = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "feature_columns": list(feature_columns),
        "config": cfg_dict,
        **(extra or {}),
    }
    torch.save(payload, path)
    logger.info("Checkpoint saved -> %s", path)
    return path


def load_checkpoint(path: Path) -> Dict:
    return torch.load(path, map_location="cpu", weights_only=False)


#: Bumped if the stage-2 head payload layout changes incompatibly.
TRANSFER_HEAD_FORMAT = "mmr_transfer_head/v1"


def save_transfer_head(
    path: Path, model: torch.nn.Module, *,
    arch: str, dims: Sequence[int], hidden_dim: int, dropout: float,
    scalers: Sequence[Tuple[np.ndarray, np.ndarray]],
    prior_cols: Sequence[str], feature_mode: str, esm_model: str, esm_dim: int,
    threshold: Optional[float] = None,
    metrics: Optional[Dict[str, object]] = None,
    extra: Optional[Dict[str, object]] = None,
) -> Path:
    """Persist one benchmarked stage-2 head so it can score raw features again.

    ``scripts/run_mmr_transfer.py`` trains one head per (holdout gene x
    architecture) and, before this existed, kept only the metrics row -- the
    trained head and the per-view :class:`~sklearn.preprocessing.StandardScaler`
    fitted on its train slice were both lost on exit, so no downstream stage
    (calibration especially) could reproduce a single held-out probability.

    The payload carries the architecture config, one ``(mean, scale)`` pair per
    feature view, the prior-column schema, and the MCC threshold -- everything
    :func:`load_transfer_head` needs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {
        "format": TRANSFER_HEAD_FORMAT,
        "model_state_dict": model.state_dict(),
        "config": {
            "arch": arch, "dims": [int(d) for d in dims],
            "hidden_dim": int(hidden_dim), "dropout": float(dropout),
            "feature_mode": feature_mode, "esm_model": esm_model,
            "esm_dim": int(esm_dim),
        },
        "scalers": [
            {"mean": np.asarray(m, dtype=np.float64),
             "scale": np.asarray(s, dtype=np.float64)}
            for m, s in scalers
        ],
        "feature_columns": list(prior_cols),
    }
    if threshold is not None:
        payload["threshold"] = float(threshold)
    if metrics:
        payload["metrics"] = dict(metrics)
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    logger.info("Transfer head saved -> %s", path)
    return path


def load_transfer_head(path: Path, device: Optional[torch.device] = None):
    """Rebuild a stage-2 head written by :func:`save_transfer_head`.

    Returns ``(model, scale_views, payload)`` where ``scale_views`` maps a list
    of raw feature matrices (one per view, same order as training) to their
    standardised form using the persisted per-view statistics.
    """
    payload = load_checkpoint(Path(path))
    cfg = payload["config"]
    model = build_model(cfg["arch"], dims=cfg["dims"],
                        hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"])
    model.load_state_dict(payload["model_state_dict"])
    if device is not None:
        model.to(device)
    model.eval()

    stats = [(np.asarray(s["mean"]), np.asarray(s["scale"]))
             for s in payload["scalers"]]

    def scale_views(mats: Sequence[np.ndarray]) -> List[np.ndarray]:
        if len(mats) != len(stats):
            raise ValueError(
                f"{len(mats)} feature views given but the checkpoint stored "
                f"{len(stats)} scalers.")
        return [((np.asarray(m, dtype=np.float64) - mean) / scale).astype(np.float32)
                for m, (mean, scale) in zip(mats, stats)]

    return model, scale_views, payload


# --------------------------------------------------------------------------- #
# Shared feature-block assembly used by both stage CLIs
# --------------------------------------------------------------------------- #
@dataclass
class FeatureBundle:
    """Assembled training matrices for one stage.

    ``X_esm`` is the frozen ESM+PLLR branch (or None in priors-only mode);
    ``X_prior`` the leakage-safe external-prior branch whose column order is
    exactly ``prior_cols`` (missingness flags included).  ``meta`` is
    row-aligned with every matrix.
    """

    X_esm: Optional[np.ndarray]
    X_prior: np.ndarray
    prior_cols: List[str]
    meta: pd.DataFrame
    #: Fill value actually used per prior source column. Persist this next to
    #: ``prior_cols`` in a checkpoint and pass it back in at the next stage so
    #: the head sees the same imputed constants it was fitted on.
    impute_values: Dict[str, float] = field(default_factory=dict)

    @property
    def dims(self) -> List[int]:
        dims = []
        if self.X_esm is not None:
            dims.append(self.X_esm.shape[1])
        dims.append(self.X_prior.shape[1])
        return dims

    def stack(self) -> np.ndarray:
        """Single-matrix view (ESM | priors) for BranchHead-style models."""
        parts = [x for x in (self.X_esm, self.X_prior) if x is not None]
        return np.concatenate(parts, axis=1).astype(np.float32) if len(parts) > 1 \
            else parts[0].astype(np.float32)


def assemble_features(
    df: pd.DataFrame,
    sequence_by_gene: Dict[str, str],
    model_name: str,
    processed_dir: Path,
    device: torch.device,
    features_mode: str = "esm+priors",
    batch_size: int = 8,
    overwrite_cache: bool = False,
    fixed_prior_columns: Optional[Sequence[str]] = None,
    fixed_impute_values: Optional[Mapping[str, float]] = None,
) -> FeatureBundle:
    """Build the (ESM | prior) feature bundle for every labelled row in *df*.

    ``fixed_prior_columns`` and ``fixed_impute_values`` both come from the
    checkpoint being warm-started from: the first pins the feature *order*,
    the second pins the imputed *values*. Pinning only the order still leaves
    the head reading differently-centred columns across stages -- see
    :func:`prior_matrix`.
    """
    if features_mode == "esm+priors":
        X_esm_all, meta = esm_branch_matrix(
            df, sequence_by_gene, model_name, processed_dir, device,
            batch_size=batch_size, overwrite_cache=overwrite_cache)
        # Keep only rows that survived extraction (unknown-sequence genes drop).
        mask = align_rows(meta, df)
        df_aligned = df.loc[mask].reset_index(drop=True)
        if len(df_aligned) != len(meta):
            raise RuntimeError("Row alignment failed between extraction pool "
                               "and labelled table.")
        X_prior, prior_cols = prior_matrix(meta, columns=fixed_prior_columns,
                                           impute_values=fixed_impute_values)
        return FeatureBundle(X_esm=X_esm_all.astype(np.float32),
                             X_prior=X_prior.astype(np.float32),
                             prior_cols=prior_cols, meta=meta,
                             impute_values=_effective_impute_values(
                                 meta, prior_cols, fixed_impute_values))
    if features_mode == "priors":
        X_prior, prior_cols = prior_matrix(df, columns=fixed_prior_columns,
                                           impute_values=fixed_impute_values)
        return FeatureBundle(X_esm=None, X_prior=X_prior.astype(np.float32),
                             prior_cols=prior_cols,
                             meta=df.reset_index(drop=True),
                             impute_values=_effective_impute_values(
                                 df, prior_cols, fixed_impute_values))
    raise ValueError(f"Unknown --features mode '{features_mode}'.")


def _effective_impute_values(
    df: pd.DataFrame, prior_cols: Sequence[str],
    fixed: Optional[Mapping[str, float]],
) -> Dict[str, float]:
    """The fill values :func:`prior_matrix` just used, for persisting."""
    values = prior_impute_values(df, prior_cols)
    if fixed:
        values.update({c: float(v) for c, v in fixed.items() if c in values})
    return values


def stage_sample_weights(meta: pd.DataFrame, clinical_weight: float,
                         dms_weight: float = 1.0) -> np.ndarray:
    """Evidence-quality × source-priority loss weights for a stage table."""
    if "label_source" in meta.columns:
        is_clinical = meta["label_source"].isin(["clinvar", "pg_clinical"]).to_numpy()
    elif {"clinvar_label", "clinical_label"} <= set(meta.columns):
        is_clinical = (meta["clinvar_label"].notna()
                       | meta["clinical_label"].notna()).to_numpy()
    else:
        is_clinical = np.ones(len(meta), dtype=bool)
    weights = np.where(is_clinical, clinical_weight, dms_weight).astype(np.float32)
    if "label_weight" in meta.columns:
        quality = pd.to_numeric(meta["label_weight"], errors="coerce") \
            .fillna(0.0).to_numpy(np.float32)
        weights = weights * quality
    return weights


__all__ = [
    "RANK_PREFIX", "CONSENSUS_COL", "GENE_CONSTANT_PRIOR_COLS",
    "PRIOR_SCORE_SIGN", "rankable_score_columns",
    "add_within_gene_rank_features",
    "MMR_GENES", "TRANSFER_PRIOR_COLS", "AF_DERIVED_PRIOR_COLS",
    "assert_af_quarantine",
    "TransferConfig", "FeatureBundle",
    "prior_columns_of", "prior_matrix", "prior_impute_values",
    "esm_branch_matrix", "align_rows",
    "assemble_features", "stage_sample_weights",
    "build_model", "predict_logits", "fit_head", "select_stage_rows",
    "save_checkpoint", "load_checkpoint",
    "TRANSFER_HEAD_FORMAT", "save_transfer_head", "load_transfer_head",
]
