"""ESM-2 feature extraction: per-residue embeddings and zero-shot PLLR scores.

For every missense variant ``(i, wt -> mut)`` of a wild-type sequence ``X`` the
extractor builds the supervised feature vector

    z = [ h_wt || h_mut || (h_mut - h_wt) || |h_mut - h_wt| || delta_i ]

where

* ``h_wt``  : last-layer hidden state of ESM-2 at residue ``i`` for the
  wild-type sequence,
* ``h_mut``: last-layer hidden state at residue ``i`` after mutating residue
  ``i`` in silico to the alternate amino acid,
* ``delta_i = log P(x_mut | X_\\i) - log P(x_wt | X_\\i)`` is the pseudo-log-
  likelihood ratio (PLLR) zero-shot score. Because both pseudo-likelihoods are
  conditioned on the *same* masked context, both log-probabilities can be read
  from a single forward pass over the wild-type sequence (the standard trick of
  Meier et al., 2021).

Sequences longer than the model's positional capacity are processed with
overlapping sliding windows; per-residue hidden states are averaged across all
windows covering the residue and per-residue log-probabilities are averaged in
log space.

All extracted features are cached under ``data/processed/`` so that repeated
runs skip the expensive forward passes entirely.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .data_loader import VALID_AA

logger = logging.getLogger(__name__)

#: Longest protein stretch addressable by the ESM-2 position embeddings.
MAX_RESIDUES: int = 1022
#: Overlap between consecutive sliding windows.
WINDOW_OVERLAP: int = 256


def get_device() -> torch.device:
    """Automatic device fallback: CUDA -> Apple MPS -> CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)
    return device


def validate_and_align(df: pd.DataFrame, sequence: str) -> pd.DataFrame:
    """Drop variants whose wt/position disagree with the canonical sequence.

    This guards against ClinVar rows annotated on a different isoform than the
    UniProt canonical sequence.
    """
    seq_arr = np.array(list(sequence), dtype="U1")
    # Coerce defensively: cached CSVs may yield float/object dtypes when a
    # frame is empty or contains NaN sentinels.
    pos_num = pd.to_numeric(df["position"], errors="coerce").to_numpy(dtype="float64")
    in_range = np.isfinite(pos_num) & (pos_num >= 1) & (pos_num <= len(sequence))
    n_dropped_range = int((~in_range).sum())
    wt_ok = np.zeros(len(df), dtype=bool)
    idx = np.flatnonzero(in_range)
    positions_int = pos_num[idx].astype("int64")
    wt_ok[idx] = seq_arr[positions_int - 1] == df["wt_aa"].to_numpy()[idx]
    # A valid substitution token has exactly one amino-acid character.
    mut_ok = np.asarray([len(str(m)) == 1 and str(m) in VALID_AA
                         for m in df["mut_aa"]], dtype=bool)
    keep = in_range & wt_ok & mut_ok
    dropped = len(df) - int(keep.sum())
    if dropped:
        logger.warning(
            "Dropped %d variants inconsistent with canonical sequence "
            "(%d out-of-range/non-numeric, %d isoform/reference mismatches).",
            dropped, n_dropped_range, dropped - n_dropped_range,
        )
    out = df.loc[keep].reset_index(drop=True)
    out["position"] = out["position"].astype("int64")
    return out


def _sliding_spans(length: int, max_residues: int = MAX_RESIDUES,
                   overlap: int = WINDOW_OVERLAP) -> List[Tuple[int, int]]:
    """Return half-open ``[start, end)`` windows covering ``[0, length)``."""
    if length <= max_residues:
        return [(0, length)]
    step = max(1, max_residues - overlap)
    spans: List[Tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + max_residues, length)
        spans.append((start, end))
        if end == length:
            break
        start += step
    # Guarantee full tail coverage even after aggressive stepping.
    if spans[-1][1] != length:
        spans.append((max(0, length - max_residues), length))
    return spans


class ESM2Extractor:
    """Wraps a HuggingFace ESM-2 checkpoint for embedding / PLLR extraction."""

    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        device: Optional[torch.device] = None,
        batch_size: int = 8,
    ) -> None:
        # Imported lazily so that `import src` stays cheap without transformers.
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError(
                "transformers is required for ESM-2 extraction. "
                "pip install -r requirements.txt"
            ) from exc

        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.device = device or get_device()

        logger.info("Loading tokenizer/model for %s ...", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device).eval()
        self.hidden_dim: int = self.model.config.hidden_size

        # Map each standard amino acid to its vocabulary id once.
        self._aa_to_id: dict[str, int] = {
            aa: int(self.tokenizer.convert_tokens_to_ids(aa))
            for aa in "ACDEFGHIKLMNPQRSTVWY"
        }

    # ------------------------------------------------------------------ #
    # Low-level forward helpers
    # ------------------------------------------------------------------ #
    def _embed_spans(
        self, jobs: Sequence[Tuple[int, str]]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Forward a list of ``(job_id, sub_sequence)`` spans in mini-batches.

        Returns per-job accumulated ``(hidden [L,d], mean_log_probs [L,V])``
        arrays where residues outside the span remain zero-filled by callers.
        """
        acc_hidden: dict[int, list] = {}
        acc_logp: dict[int, list] = {}

        def flush(batch: Sequence[Tuple[int, str, int]]) -> None:
            if not batch:
                return
            seqs = [s for _, s, _ in batch]
            tokens = self.tokenizer(seqs, return_tensors="pt", padding=True)
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            with torch.inference_mode():
                # fp16 autocast on CUDA roughly doubles ESM-2 throughput;
                # hidden states / logits are cast back to fp32 below.
                #
                # Only the last layer's hidden state is used below, so we call
                # the encoder and LM head directly instead of passing
                # output_hidden_states=True through the full MaskedLM model:
                # that flag makes HF retain *every* transformer layer's
                # hidden state (33 layers + embeddings for esm2_t33_650M),
                # a ~34x peak-memory spike per batch for the 33 layers we
                # never look at.
                if self.device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        sequence_output = self.model.esm(**tokens).last_hidden_state
                        logits = self.model.lm_head(sequence_output)
                else:
                    sequence_output = self.model.esm(**tokens).last_hidden_state
                    logits = self.model.lm_head(sequence_output)
                hidden = sequence_output.float()                     # [B,T,d]
                log_probs = torch.log_softmax(logits.float(), dim=-1)

            lengths = tokens["attention_mask"].sum(dim=1)
            for row, (job_id, _, offset) in enumerate(batch):
                # Strip <cls> at index 0; keep residues up to span length.
                n_res = min(int(lengths[row].item()) - 2, len(seqs[row]))
                h = hidden[row, 1:1 + n_res].cpu().numpy()
                lp = log_probs[row, 1:1 + n_res].cpu().numpy()
                acc_hidden.setdefault(job_id, []).append((offset, h))
                acc_logp.setdefault(job_id, []).append((offset, lp))

        pending: list[Tuple[int, str, int]] = []
        for job_id, subseq in jobs:
            pending.append((job_id, subseq, 0))
            if len(pending) >= self.batch_size:
                flush(pending)
                pending = []
        flush(pending)

        hidden_out: List[np.ndarray] = []
        logp_out: List[np.ndarray] = []
        for job_id in sorted(acc_hidden):
            pieces_h, pieces_lp = acc_hidden[job_id], acc_logp[job_id]
            total_len = sum(p[1].shape[0] for p in pieces_h)
            h_cat = np.concatenate([p[1] for p in pieces_h], axis=0)
            assert h_cat.shape[0] == total_len
            hidden_out.append(h_cat)
            logp_out.append(np.concatenate([p[1] for p in pieces_lp], axis=0))
        return hidden_out, logp_out

    def embed_sequences(
        self, sequences: Sequence[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Embed full-length sequences with sliding-window aggregation.

        Returns ``(hidden [N,L,d], mean_log_probs [N,L,V])``. Hidden states of
        residues covered by several windows are averaged; per-residue log-probs
        are averaged in log space (a geometric-mean approximation).
        """
        n_seq = len(sequences)
        lengths = [len(s) for s in sequences]

        # Build one job per (sequence, window).  Job ids must be unique PER
        # WINDOW (not per sequence): _embed_spans accumulates pieces keyed by
        # job id, so reusing the sequence index would concatenate overlapping
        # windows of long chains (e.g. MSH6, 1360 aa) into one oversized block.
        jobs: List[Tuple[int, str]] = []
        job_spans: List[Tuple[int, Tuple[int, int]]] = []   # (si, (start, end))
        for si, seq in enumerate(sequences):
            spans = _sliding_spans(len(seq))
            for start, end in spans:
                job_spans.append((si, (start, end)))
                jobs.append((len(jobs), seq[start:end]))

        span_hidden, span_logp = self._embed_spans(jobs)
        if len(span_hidden) != len(jobs) or len(span_logp) != len(jobs):
            raise RuntimeError(
                f"ESM span extraction returned hidden={len(span_hidden)} and "
                f"logp={len(span_logp)} blocks for "
                f"{len(jobs)} windows."
            )

        hidden = np.zeros((n_seq, max(lengths), self.hidden_dim), dtype=np.float32)
        logp = np.zeros(
            (n_seq, max(lengths), span_logp[0].shape[1]), dtype=np.float32
        )
        counts = np.zeros((n_seq, max(lengths)), dtype=np.float32)

        for cursor, (si, (_start, _end)) in enumerate(job_spans):
            h = span_hidden[cursor]
            lp = span_logp[cursor]
            span_len = _end - _start
            if h.shape[0] != span_len or lp.shape[0] != span_len:
                raise RuntimeError(
                    "ESM span length mismatch: "
                    f"window [{_start}, {_end}) expected {span_len} residues, "
                    f"got hidden={h.shape[0]}, logp={lp.shape[0]}."
                )
            hidden[si, _start:_start + h.shape[0]] += h
            logp[si, _start:_start + lp.shape[0]] += lp
            counts[si, _start:_start + h.shape[0]] += 1.0
        counts_safe = np.maximum(counts, 1.0)[..., None]
        return hidden / counts_safe, logp / counts_safe

    # ------------------------------------------------------------------ #
    # High-level variant extraction
    # ------------------------------------------------------------------ #
    def extract(self, df: pd.DataFrame, sequence: str) -> Tuple[np.ndarray, pd.DataFrame]:
        """Compute the stacked feature matrix for every variant in *df*.

        Parameters
        ----------
        df:
            Dataframe with columns ``position`` (1-based), ``wt_aa``, ``mut_aa``.
        sequence:
            Canonical wild-type amino-acid sequence.

        Returns
        -------
        features : ndarray of shape ``[n_variants, 4*d + 1]``
        meta     : copy of *df* with an added ``pllr`` column.
        """
        positions = df["position"].to_numpy()
        wt_ids = df["wt_aa"].map(self._aa_to_id).to_numpy()
        mut_ids = df["mut_aa"].map(self._aa_to_id).to_numpy()

        # ---- Wild-type pass: one sequence covers every variant -------------
        logger.info("Forward pass on wild-type sequence (%d aa) ...", len(sequence))
        h_wt_full, logp_wt_full = self.embed_sequences([sequence])
        h_wt_all = h_wt_full[0][positions - 1]                     # [n, d]
        site_logp = logp_wt_full[0][positions - 1]                 # [n, V]
        pllr = site_logp[np.arange(len(df)), mut_ids] \
            - site_logp[np.arange(len(df)), wt_ids]                # [n]

        # ---- Mutant passes -------------------------------------------------
        unique_pairs = (
            df[["position", "mut_aa"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        logger.info("Forward passes on %d unique mutant sequences ...", len(unique_pairs))
        mut_seqs = [
            sequence[:pos - 1] + mut + sequence[pos:]
            for pos, mut in zip(unique_pairs["position"], unique_pairs["mut_aa"])
        ]
        h_mut_unique, _ = self.embed_sequences(mut_seqs)

        pair_key = {
            (int(pos), mut): k
            for k, (pos, mut) in enumerate(zip(unique_pairs["position"],
                                               unique_pairs["mut_aa"]))
        }
        rows = [pair_key[(int(p), m)] for p, m in zip(df["position"], df["mut_aa"])]
        # Gather each unique mutant's hidden state at its own mutated residue.
        mut_pos_idx = unique_pairs["position"].to_numpy() - 1
        h_mut_at_site = h_mut_unique[np.arange(len(unique_pairs)), mut_pos_idx]  # [n_mut, d]
        h_mut = h_mut_at_site[np.asarray(rows)]                    # [n, d]

        # ---- Feature stacking ----------------------------------------------
        diff = h_mut - h_wt_all
        features = np.concatenate(
            [h_wt_all, h_mut, diff, np.abs(diff), pllr[:, None].astype(np.float32)],
            axis=1,
        ).astype(np.float32)

        meta = df.copy()
        meta["pllr"] = pllr.astype(np.float64)
        logger.info("Feature matrix: %s (d=%d)", features.shape, self.hidden_dim)
        return features, meta


def extract_features_cached(
    df: pd.DataFrame,
    sequence: str,
    gene: str,
    model_name: str,
    processed_dir: Path,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
    overwrite: bool = False,
    extra_features: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Extract (or load cached) ESM-2 features for all variants in *df*.

    The cache key combines gene, checkpoint name and (when external prior
    features are attached) an ``+extras`` suffix; the returned arrays are
    row-aligned with *df* (validated on cache hits).

    Parameters
    ----------
    extra_features:
        Optional dataframe aligned 1:1 with *df* whose numeric columns are
        appended to the ESM embedding features along axis 1.  Used to inject
        AlphaMissense / published-model priors from the extended dataset.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    tag = model_name.split("/")[-1]
    if extra_features is not None:
        tag += f"+extras{extra_features.shape[1]}"
    feat_path = processed_dir / f"{gene}_{tag}_features.npz"
    meta_path = processed_dir / f"{gene}_{tag}_features_meta.csv"

    if feat_path.exists() and meta_path.exists() and not overwrite:
        logger.info("Loading cached ESM-2 features from %s", feat_path)
        blob = np.load(feat_path)
        meta = pd.read_csv(meta_path)
        if len(meta) != len(df):
            raise RuntimeError(
                f"Cached metadata ({len(meta)} rows) does not match current "
                f"variant table ({len(df)} rows); re-run with --overwrite_cache."
            )
        return blob["features"].astype(np.float32), meta

    extractor = ESM2Extractor(model_name=model_name, device=device, batch_size=batch_size)
    features, meta = extractor.extract(df, sequence)
    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if extra_features is not None:
        if len(extra_features) != len(features):
            raise RuntimeError(
                "extra_features rows (%d) do not match variants (%d)."
                % (len(extra_features), len(features)))
        numeric = extra_features.select_dtypes(include=[np.number]).to_numpy()
        features = np.concatenate(
            [features, numeric.astype(np.float32)], axis=1).astype(np.float32)
        logger.info("Appended %d external prior features -> d=%d",
                    numeric.shape[1], features.shape[1])

    np.savez_compressed(feat_path, features=features)
    meta.to_csv(meta_path, index=False)
    logger.info("Cached features -> %s", feat_path)
    return features, meta
