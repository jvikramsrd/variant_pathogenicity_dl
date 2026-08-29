"""MVmamba-style structural-PLM feature extraction + zero-shot baselines.

Implements PROJECT_PLAN.md Phase 3 step 1 — the primary base-paper recipe:

* Embed **wild-type (WT)** and **variant-type (VT)** sequences through the
  protein-language-model separately (structure-informed encodings in the base
  paper; ESM-2 is the closest available substitute here).
* Extract **global** features (mean-pool over the full sequence) and **local**
  features (mean-pool inside ``mutation +/- w`` residues, optimal ``w = 3``
  per MVmamba's own window-size sweep, Table I) for BOTH WT and VT — four
  feature vectors per variant: ``[g_wt || g_vt || l_wt || l_vt]``, optionally
  stacked with their differences.
* gnomAD allele frequency enters downstream as an explicit input feature
  (:mod:`src.gnomad`), mirroring the paper's ablation gain (AUC 0.895->0.901).

Sequence handling follows Phase 3 step 3 (VariPred recipe): proteins longer
than the positional capacity (MSH6, 1360 aa) get their *local* features from a
window centred on the mutation (510/511 split, nearest-terminus fallback);
global features always cover the whole chain via overlap-averaged windows.
The other three MMR genes fit natively.

Phase 3 step 2 additionally requires true **masked-marginal zero-shot**
baselines for ESM-1b and ESM2-650M:
``delta_i = log P(x_mut | X_\\i) - log P(x_wt | X_\\i)``
read off a single forward pass over the wild-type sequence (the standard
Meier et al. 2021 trick — both pseudo-likelihoods share the masked context).
:class:`MaskedMarginalScorer` provides this for any HF masked-LM checkpoint,
so the two backbones are compared on identical footing ("don't assume ESM2
wins").
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .esm_extractor import (ESM2Extractor, MAX_RESIDUES, _assert_cache_matches,
                             get_device, validate_and_align)

logger = logging.getLogger(__name__)

#: Default half-width for the local window (MVmamba Table I optimum: +-3).
DEFAULT_LOCAL_WINDOW = 3

#: Variant-type sequences embedded per batch before being reduced to pooled
#: vectors. Bounds peak memory at ``chunk x L x hidden_dim`` floats rather than
#: the whole variant set at once -- see the note in
#: :meth:`MVmambaFeatureExtractor.extract`.
DEFAULT_VT_CHUNK = 64


def centered_window_bounds(seq_len: int, pos0: int,
                           max_residues: int = MAX_RESIDUES) -> Tuple[int, int]:
    """Half-open ``[start, end)`` residue bounds of a mutation-centred window.

    Follows the VariPred asymmetric-window recipe quoted in the plan: a
    510/511 split centred on the mutated residue, falling back to the nearest
    terminus when the centred window would leave the chain.
    """
    span = min(max_residues, seq_len)
    left_capacity = span // 2          # 510/511 style split
    start = int(np.clip(pos0 - left_capacity, 0, max(0, seq_len - span)))
    end = start + span
    # Guarantee the mutated residue is inside the window.
    if pos0 < start:
        start, end = pos0, min(seq_len, pos0 + span)
    elif pos0 >= end:
        end = min(seq_len, pos0 + 1)
        start = max(0, end - span)
    return start, end


class MVmambaFeatureExtractor:
    """Four-vector WT/VT global/local feature extraction on an ESM backbone."""

    def __init__(self, model_name: str = "facebook/esm2_t33_650M_UR50D",
                 device: Optional[torch.device] = None, batch_size: int = 8,
                 local_window: int = DEFAULT_LOCAL_WINDOW,
                 include_deltas: bool = True,
                 vt_chunk_size: int = DEFAULT_VT_CHUNK) -> None:
        self.backend = ESM2Extractor(model_name=model_name, device=device,
                                     batch_size=batch_size)
        self.local_window = int(local_window)
        self.include_deltas = bool(include_deltas)
        self.vt_chunk_size = int(vt_chunk_size)

    # ------------------------------------------------------------------ #
    def _pool(self, hidden: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Mean-pool along the residue axis (mask shape [L])."""
        if mask is None:
            return hidden.mean(axis=0)
        m = mask[:, None].astype(np.float32)
        return (hidden * m).sum(axis=0) / max(1.0, float(mask.sum()))

    def _local_pool(self, hidden: np.ndarray, pos0: int, window_span: Tuple[int, int]) -> np.ndarray:
        """Mean-pool inside ``pos0 +/- w`` clipped to the encoded span."""
        w = self.local_window
        start_enc, end_enc = window_span
        lo = max(start_enc, pos0 - w)
        hi = min(end_enc, pos0 + w + 1)
        if hi <= lo:
            hi = lo + 1
        return hidden[lo - start_enc:hi - start_enc].mean(axis=0)

    # ------------------------------------------------------------------ #
    def extract(self, df: pd.DataFrame,
                sequence: str) -> Tuple[np.ndarray, pd.DataFrame]:
        """Compute MVmamba feature matrix for every missense variant in *df*.

        Returns ``(features [n, 4d (+4d deltas)], meta with pllr column)``.
        """
        df = validate_and_align(df.reset_index(drop=True), sequence)
        positions = df["position"].to_numpy()
        seq_len = len(sequence)

        # ---- Wild-type pass (one sequence covers every variant) ----------- #
        logger.info("MVmamba: WT forward pass (%d aa) ...", seq_len)
        h_wt_full, logp_wt_full = self.backend.embed_sequences([sequence])
        h_wt_seq = h_wt_full[0][:seq_len]                       # [L, d]
        g_wt = self._pool(h_wt_seq)
        site_logp = logp_wt_full[0][positions - 1]              # [n, V]

        wt_ids = df["wt_aa"].map(self.backend._aa_to_id).to_numpy()
        mut_ids = df["mut_aa"].map(self.backend._aa_to_id).to_numpy()
        pllr = site_logp[np.arange(len(df)), mut_ids] \
            - site_logp[np.arange(len(df)), wt_ids]

        # ---- Variant-type passes ------------------------------------------ #
        uniq = df[["position", "mut_aa"]].drop_duplicates().reset_index(drop=True)
        logger.info("MVmamba: %d unique VT sequences ...", len(uniq))
        g_vt_by_key: Dict[Tuple[int, str], np.ndarray] = {}
        l_vt_by_key: Dict[Tuple[int, str], np.ndarray] = {}
        l_wt_by_pos: Dict[int, np.ndarray] = {}
        l_wt_by_span: Dict[Tuple[int, int], np.ndarray] = {}

        mut_seqs = [
            sequence[:p - 1] + m + sequence[p:]
            for p, m in zip(uniq["position"], uniq["mut_aa"])
        ]
        long_rows = {k for k, s in enumerate(mut_seqs) if len(s) > MAX_RESIDUES}

        # --- Full-length VT passes, CHUNKED -------------------------------- #
        # Both outputs wanted from the variant-type pass are reductions (a
        # full-sequence mean and a +/-w window mean), so the [N, L, d] hidden
        # tensor never needs to exist all at once. Building it did: scoring
        # MSH6's 3,886 VUS at 1360 aa x 1280 dims is a single 25 GiB float32
        # allocation, which is an immediate OOM on any ordinary machine -- and
        # VUS scoring is the task this pipeline exists to perform. Reducing
        # each chunk on arrival caps the peak at chunk_size x L x d instead
        # (~0.4 GiB at the default 64), and leaves the arithmetic unchanged.
        chunk = max(1, int(self.vt_chunk_size))
        n_uniq = len(mut_seqs)
        if n_uniq > chunk:
            logger.info("MVmamba: %d VT sequences in chunks of %d "
                        "(peak ~%.2f GiB per chunk).", n_uniq, chunk,
                        chunk * seq_len * self.backend.hidden_dim * 4 / 2**30)
        for start in range(0, n_uniq, chunk):
            stop = min(start + chunk, n_uniq)
            h_chunk, _ = self.backend.embed_sequences(mut_seqs[start:stop])
            for offset, k in enumerate(range(start, stop)):
                p, m = uniq["position"].iloc[k], uniq["mut_aa"].iloc[k]
                key = (int(p), str(m))
                p0 = int(p) - 1
                h_mut = h_chunk[offset][:len(mut_seqs[k])]
                g_vt_by_key[key] = self._pool(h_mut)
                if k not in long_rows:
                    l_vt_by_key[key] = self._local_pool(h_mut, p0, (0, seq_len))
            del h_chunk

        # --- Mutation-centred windows for over-length chains --------------- #
        # Local features for chains past the positional capacity come from a
        # window centred on the mutation (VariPred recipe) rather than from the
        # overlap-averaged full-length pass.
        #
        # The variant-type window is sliced from ``mut_seqs[k]``, NOT from
        # ``sequence``. Slicing the wild-type chain here -- as this branch
        # previously did -- made the "variant-type" local vector literally the
        # wild-type one, so ``l_vt - l_wt`` and ``|l_vt - l_wt|`` were
        # identically zero for every chain over the capacity: two of the eight
        # feature blocks dead, and precisely the windowed WT/VT contrast the
        # MVmamba recipe is built around (their Table I window sweep tunes
        # exactly this term). On the MMR panel only MSH6 (1360 aa) exceeds the
        # limit, so the defect silently gave one gene a different feature
        # space from the other three -- inside a leave-one-gene-out design.
        if long_rows:
            logger.info("MVmamba: %d VT chains exceed %d aa -> extra "
                        "mutation-centred window passes (VariPred recipe).",
                        len(long_rows), MAX_RESIDUES)
            order = sorted(long_rows)
            spans = {k: centered_window_bounds(
                len(sequence), int(uniq["position"].iloc[k]) - 1)
                for k in order}
            # The wild-type window depends only on the position, so distinct
            # substitutions at one residue share it. Collect the distinct
            # windows and batch them, instead of one forward pass per variant
            # inside the loop.
            wt_spans = sorted({spans[k] for k in order})
            for wstart in range(0, len(wt_spans), chunk):
                batch = wt_spans[wstart:wstart + chunk]
                h_wc, _ = self.backend.embed_sequences(
                    [sequence[s:e] for s, e in batch])
                for i, (s, e) in enumerate(batch):
                    l_wt_by_span[(s, e)] = h_wc[i][:e - s]
                del h_wc

            for kstart in range(0, len(order), chunk):
                ks = order[kstart:kstart + chunk]
                jobs = [(k, mut_seqs[k][spans[k][0]:spans[k][1]]) for k in ks]
                span_hidden, _ = self.backend._embed_spans(jobs)
                for (k, _sub), h in zip(jobs, span_hidden):
                    p, m = uniq["position"].iloc[k], uniq["mut_aa"].iloc[k]
                    p0 = int(p) - 1
                    s, e = spans[k]
                    l_vt_by_key[(int(p), str(m))] = self._local_pool(h, p0, (s, e))
                    if p0 not in l_wt_by_pos:
                        l_wt_by_pos[p0] = self._local_pool(
                            l_wt_by_span[(s, e)], p0, (s, e))
                del span_hidden

        # ---- Assemble per-variant rows ------------------------------------- #
        rows_g_vt, rows_l_vt, rows_l_wt = [], [], []
        for p, m in zip(positions, df["mut_aa"]):
            key = (int(p), str(m))
            rows_g_vt.append(g_vt_by_key[key])
            rows_l_vt.append(l_vt_by_key[key])
            p0 = int(p) - 1
            if p0 in l_wt_by_pos:
                rows_l_wt.append(l_wt_by_pos[p0])
            else:
                rows_l_wt.append(self._local_pool(h_wt_seq, p0, (0, seq_len)))

        g_vt_arr = np.vstack(rows_g_vt)
        l_vt_arr = np.vstack(rows_l_vt)
        l_wt_arr = np.vstack(rows_l_wt)
        g_wt_arr = np.broadcast_to(g_wt, g_vt_arr.shape).astype(np.float32)

        blocks = [g_wt_arr, g_vt_arr, l_wt_arr.astype(np.float32),
                  l_vt_arr.astype(np.float32)]
        if self.include_deltas:
            blocks += [g_vt_arr - g_wt_arr,
                       l_vt_arr - l_wt_arr,
                       np.abs(g_vt_arr - g_wt_arr),
                       np.abs(l_vt_arr - l_wt_arr)]
        features = np.concatenate(blocks, axis=1).astype(np.float32)

        meta = df.copy()
        meta["pllr"] = pllr.astype(np.float64)
        meta["mv_local_window"] = self.local_window
        logger.info("MVmamba feature matrix: %s (d=%d, window=+%d-%d)",
                    features.shape, self.backend.hidden_dim,
                    self.local_window, self.local_window)
        return features, meta


# --------------------------------------------------------------------------- #
# Zero-shot masked-marginal scoring for arbitrary HF protein LMs
# --------------------------------------------------------------------------- #
class MaskedMarginalScorer:
    """``log P(mut|masked) - log P(wt|masked)`` zero-shot baseline scorer.

    Loads any HuggingFace masked-LM checkpoint (``facebook/esm1b_t33_650M_UR50S``
    and ``facebook/esm2_t33_650M_UR50D`` are the plan-mandated pair) once and
    scores every variant from a single forward pass per sequence.
    """

    def __init__(self, model_name: str,
                 device: Optional[torch.device] = None, batch_size: int = 8) -> None:
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError("transformers required: pip install -r requirements.txt") from exc
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.device = device or get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device).eval()
        self._aa_to_id = {aa: int(self.tokenizer.convert_tokens_to_ids(aa))
                          for aa in "ACDEFGHIKLMNPQRSTVWY"}

    @torch.inference_mode()
    def score(self, df: pd.DataFrame, sequence: str) -> np.ndarray:
        """Per-variant masked-marginal scores, row-aligned with *df*."""
        df = validate_and_align(df.reset_index(drop=True), sequence)
        positions = df["position"].to_numpy()
        logp = self.model_logprobs(sequence)
        site = logp[positions - 1]                                  # [n, V]
        wt_ids = df["wt_aa"].map(self._aa_to_id).to_numpy()
        mut_ids = df["mut_aa"].map(self._aa_to_id).to_numpy()
        return (site[np.arange(len(df)), mut_ids]
                - site[np.arange(len(df)), wt_ids]).astype(np.float64)

    def pathogenicity_score(self, df: pd.DataFrame, sequence: str) -> np.ndarray:
        """:meth:`score` negated, so that **higher = more pathogenic**.

        :meth:`score` returns the raw PLLR, ``log P(mut) - log P(wt)``, which
        is *negative* when the substitution is unlikely under the language
        model -- i.e. the more damaging a variant, the lower its PLLR. That
        orientation is fine when the value is consumed as a learned feature
        (the head fits its own sign), but it is backwards for anything that
        compares the score directly against a pathogenic=1 label: ROC-AUC
        silently comes out as ``1 - AUC``, so a genuinely strong backbone
        reports as far worse than random. Use this method for any direct
        evaluation, ranking, or threshold; use :meth:`score` for features.
        """
        return -self.score(df, sequence)

    def model_logprobs(self, sequence: str) -> np.ndarray:
        """Sliding-window per-residue log-probs ``[L, V]`` for one sequence."""
        from .esm_extractor import _sliding_spans
        spans = _sliding_spans(len(sequence))
        jobs: List[Tuple[int, int]] = list(enumerate(spans))   # (window_idx, (s,e))
        vocab_size = int(self.model.config.vocab_size)
        logp = np.zeros((len(sequence), vocab_size), dtype=np.float32)
        counts = np.zeros(len(sequence), dtype=np.float32)

        def run_batch(batch: Sequence[Tuple[int, Tuple[int, int]]]) -> None:
            seqs = [sequence[s:e] for _, (s, e) in batch]
            tokens = self.tokenizer(seqs, return_tensors="pt", padding=True)
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(**tokens).logits
            else:
                logits = self.model(**tokens).logits
            log_probs = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
            for row, (_, (s, e)) in enumerate(batch):
                n_res = e - s
                lp = log_probs[row, 1:1 + n_res]
                logp[s:s + n_res] += lp
                counts[s:s + n_res] += 1.0

        for start in range(0, len(jobs), max(1, self.batch_size)):
            run_batch(jobs[start:start + max(1, self.batch_size)])
        return logp / np.maximum(counts, 1.0)[..., None]


# --------------------------------------------------------------------------- #
# Cached driver
# --------------------------------------------------------------------------- #
def extract_mvmamba_cached(df: pd.DataFrame, sequence: str, gene: str,
                           model_name: str, processed_dir: Path,
                           local_window: int = DEFAULT_LOCAL_WINDOW,
                           batch_size: int = 8,
                           device: Optional[torch.device] = None,
                           overwrite: bool = False,
                           ) -> Tuple[np.ndarray, pd.DataFrame]:
    """Disk-cached :meth:`MVmambaFeatureExtractor.extract`."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{model_name.split('/')[-1]}_mv_w{local_window}"
    feat_path = processed_dir / f"{gene}_{tag}_features.npz"
    meta_path = processed_dir / f"{gene}_{tag}_features_meta.csv"
    if feat_path.exists() and meta_path.exists() and not overwrite:
        blob = np.load(feat_path)
        meta = pd.read_csv(meta_path)
        # Compare against the *aligned* table, not the raw one.
        # :meth:`MVmambaFeatureExtractor.extract` runs ``validate_and_align``
        # first, so the cached meta holds only the rows that survived it. A
        # plain ``len(meta) != len(df)`` check against the raw table therefore
        # rejected its own cache forever the moment a single variant failed
        # alignment -- every later call raised "use overwrite=True" and
        # re-ran the whole extraction. Re-aligning here reproduces exactly
        # what was cached, and the shared key check catches the case a row
        # count alone cannot: a table rebuilt or reordered underneath the
        # cache. (:mod:`src.esm_extractor` does not align inside ``extract``,
        # which is why only this cache needed the extra step.)
        _assert_cache_matches(meta, validate_and_align(df.reset_index(drop=True),
                                                       sequence), feat_path)
        return blob["features"].astype(np.float32), meta

    ext = MVmambaFeatureExtractor(model_name=model_name, device=device,
                                  batch_size=batch_size, local_window=local_window)
    features, meta = ext.extract(df, sequence)
    del ext
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    np.savez_compressed(feat_path, features=features)
    meta.to_csv(meta_path, index=False)
    logger.info("Cached MVmamba features -> %s", feat_path)
    return features, meta


__all__ = [
    "DEFAULT_LOCAL_WINDOW",
    "centered_window_bounds",
    "MVmambaFeatureExtractor",
    "MaskedMarginalScorer",
    "extract_mvmamba_cached",
]
