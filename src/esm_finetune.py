"""End-to-end ESM-2 fine-tuning (PROJECT_PLAN.md Phase 3 step 4).

Every model trained elsewhere in this codebase (:mod:`src.esm_extractor`,
:mod:`src.mvmamba_features`, :mod:`src.transfer`) uses **frozen** ESM-2
embeddings feeding a shallow MLP head -- a linear probe (VariPred's recipe).
That is one of the four fine-tuning strategies the plan asks to benchmark,
not automatically the best one. This module implements the other two that
require gradient flow through the backbone itself:

* ``mode="siamese"`` -- **ProPath's recipe**: separate wild-type (WT) and
  variant-type (VT) forward passes through the (partially or fully unfrozen)
  backbone, pool the hidden state at the mutated residue from each, and feed
  ``[h_wt || h_vt || h_vt-h_wt || |h_vt-h_wt|]`` to a small head. Suggested
  hyperparameters per the plan: backbone LR 1e-5, batch size 8, 10 epochs.
* ``mode="wt_site"`` -- **CSBJ-style per-residue token classifier**: a single
  WT-sequence forward pass; classify directly from the (now *gradient-updated*,
  not frozen) token embedding at the mutated position. Cheaper (one forward
  pass per example) and closer to a true per-residue token classification head
  than the frozen linear probe.

(MVmamba's own recipe -- global+local WT/VT pooled *frozen* features -- is
already implemented in :mod:`src.mvmamba_features`; VariPred's frozen linear
probe is already implemented via :mod:`src.esm_extractor` +
:class:`src.fusion.BranchHead`. Together with the two modes here, all four
strategies the plan asks to benchmark (Part 1 / Phase 3 step 4) exist in this
codebase -- see ``scripts/compare_finetune_strategies.py``.)

Sequences longer than the model's positional capacity are cropped to a
mutation-centred window (:func:`src.mvmamba_features.centered_window_bounds`,
VariPred's asymmetric-window recipe) rather than processed as sliding
sub-windows: gradient fine-tuning needs one contiguous forward pass per
example, and this is standard practice for PLM fine-tuning on long chains.
"""

from __future__ import annotations

import contextlib
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .mvmamba_features import centered_window_bounds

logger = logging.getLogger(__name__)

FINETUNE_MODES = ("wt_site", "siamese")

#: How the zero-shot term log P(mut|X) - log P(wt|X) reaches the classifier.
#: "residual" adds it as a learnable-gain skip connection past a
#: zero-initialised output layer, so an untrained model reproduces the
#: zero-shot ESM ranking (ROC-AUC ~0.834 on the MMR panel) and training learns
#: a correction on top of a working predictor. "concat" is the historical
#: behaviour: one more input dimension with a random init weight. "off" is the
#: ablation.
PLLR_MODES = ("residual", "concat", "off")

#: PROJECT_PLAN.md Phase 3 step 4's ProPath-recipe defaults.
PROPATH_DEFAULTS: Dict[str, object] = {
    "backbone_lr": 1e-5, "batch_size": 8, "epochs": 10,
}


# --------------------------------------------------------------------------- #
# Mixed precision
# --------------------------------------------------------------------------- #
def amp_dtype_for(device: torch.device) -> Optional[torch.dtype]:
    """Autocast dtype for *device*, or ``None`` when AMP does not apply.

    Backbone gradient fine-tuning stores an activation per saved tensor per
    layer, so fp32 activations -- not the weights -- are what blows up VRAM
    here (esm2_t33_650M, siamese, batch 8 x 1024 tokens: ~41 GB of fp32
    activations against ~10 GB of weights/grads/AdamW state). Halving them is
    the single largest lever, so this mirrors the fp16 autocast that
    :mod:`src.esm_extractor` already uses for frozen inference -- but prefers
    bf16 when the card supports it, because bf16 keeps fp32's exponent range
    and so needs no loss scaler to train stably.
    """
    if device.type != "cuda":
        return None                       # MPS/CPU autocast buys us nothing here
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _autocast(device: torch.device, dtype: Optional[torch.dtype]):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class ESMFineTuneClassifier(nn.Module):
    """ESM-2 backbone (partially/fully unfrozen) + pathogenicity head.

    Parameters
    ----------
    n_unfrozen_layers:
        ``0`` freezes the whole backbone (equivalent in spirit to the linear
        probe elsewhere in this repo, but routed through this module's
        siamese/token-site pooling instead of :mod:`src.esm_extractor`'s
        sliding-window aggregation -- mainly useful as a same-code-path
        ablation floor). ``-1`` unfreezes every backbone parameter (full
        fine-tune). A positive integer unfreezes that many trailing
        transformer layers (a common, cheaper middle ground).
    """

    def __init__(self, model_name: str = "facebook/esm2_t12_35M_UR50D",
                 mode: str = "wt_site", n_unfrozen_layers: int = -1,
                 hidden_dim: int = 256, dropout: float = 0.15,
                 gradient_checkpointing: bool = False,
                 pllr_mode: Optional[str] = None,
                 use_pllr: Optional[bool] = None) -> None:
        super().__init__()
        if mode not in FINETUNE_MODES:
            raise ValueError(f"mode must be one of {FINETUNE_MODES}; got {mode!r}")
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError(
                "transformers is required for ESM-2 fine-tuning. "
                "pip install -r requirements.txt") from exc

        self.model_name = model_name
        self.mode = mode
        if use_pllr is not None:
            # Legacy callers passed a bool. Honour it, but not alongside the
            # richer flag -- silently preferring one would make an ablation
            # cell run a configuration its name does not describe. Both
            # parameters default to None so that an *explicitly* passed
            # pllr_mode is distinguishable from the default.
            if pllr_mode is not None:
                raise ValueError(
                    "pass either pllr_mode or use_pllr, not both "
                    f"(got pllr_mode={pllr_mode!r}, use_pllr={use_pllr!r})")
            pllr_mode = "concat" if use_pllr else "off"
        if pllr_mode is None:
            pllr_mode = "residual"
        if pllr_mode not in PLLR_MODES:
            raise ValueError(f"pllr_mode must be one of {PLLR_MODES}; got {pllr_mode!r}")
        self.pllr_mode = pllr_mode
        #: Retained for backwards compatibility with existing callers.
        self.use_pllr = pllr_mode != "off"
        # Kept as attributes (not just consumed below) so a fine-tuned model
        # can be rebuilt from a checkpoint without the original CLI args --
        # see config() / save_finetuned() / load_finetuned_model().
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # AutoModelForMaskedLM, not AutoModel: the masked-LM head is what makes
        # the PLLR term below computable. Loading the encoder alone was the
        # reason this module could not see the one ESM-derived signal that
        # already works -- see the class docstring.
        lm = AutoModelForMaskedLM.from_pretrained(model_name)
        self.backbone = lm.esm
        self.lm_head = lm.lm_head
        self.hidden_size = int(self.backbone.config.hidden_size)

        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.lm_head.parameters():
            p.requires_grad = False
        if n_unfrozen_layers == -1:
            for p in self.backbone.parameters():
                p.requires_grad = True
            for p in self.lm_head.parameters():
                p.requires_grad = True
        elif n_unfrozen_layers > 0:
            layers = self.backbone.encoder.layer
            for layer in layers[-n_unfrozen_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
            # The LM head reads the last layer, so it moves with it.
            for p in self.lm_head.parameters():
                p.requires_grad = True
        self.n_unfrozen_layers = n_unfrozen_layers
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        emb_dim = self.esm_block_dim
        # LayerNorm BEFORE the first Linear. Raw ESM hidden states have large,
        # position-dependent norms, and in siamese mode half the concatenated
        # vector (site_wt, site_mut) is near-duplicate while the informative
        # part (their difference at a single substituted residue) is small --
        # so an unnormalised input made the head spend its capacity on scale
        # rather than on the signal.
        self.feat_norm = nn.LayerNorm(emb_dim)
        in_dim = emb_dim + (1 if self.pllr_mode == "concat" else 0)
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden_dim, 1)
        #: PLLR is a log-ratio, typically within about [-20, +5]. Dividing by a
        #: fixed constant puts it on roughly the same scale as the LayerNorm'd
        #: embedding block so neither term dominates the first Linear at
        #: initialisation. A constant (not a learned or batch statistic) keeps
        #: the feature identical between training and single-variant
        #: inference, and keeps it comparable across genes and batch sizes.
        self.register_buffer("pllr_scale", torch.tensor(10.0))
        if self.pllr_mode == "residual":
            #: Learnable gain on the zero-shot term. Negative at init because
            #: raw PLLR is negative for damaging variants while pathogenic is
            #: the positive class.
            self.pllr_gain = nn.Parameter(torch.tensor(-1.0))
            # Zero-init the scalar projection so the untrained model is exactly
            # the zero-shot predictor. out.weight still receives a non-zero
            # gradient on the first backward, so it leaves zero immediately --
            # nothing is frozen out.
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def trainable_backbone_params(self) -> List[torch.nn.Parameter]:
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def head_params(self) -> List[torch.nn.Parameter]:
        params = list(self.head.parameters()) + list(self.out.parameters())
        if self.pllr_mode == "residual":
            params.append(self.pllr_gain)
        return params

    @property
    def esm_block_dim(self) -> int:
        """Width of the ESM feature block that :meth:`encode` returns."""
        return self.hidden_size * (4 if self.mode == "siamese" else 1)

    def config(self) -> Dict[str, object]:
        """Constructor kwargs needed to rebuild this model before loading a
        fine-tuned state dict back into it (see :func:`load_finetuned_model`)."""
        return {
            "model_name": self.model_name,
            "mode": self.mode,
            "n_unfrozen_layers": self.n_unfrozen_layers,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "pllr_mode": self.pllr_mode,
        }

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def encode(self, wt_ids: torch.Tensor, wt_mask: torch.Tensor, wt_pos: torch.Tensor,
               mut_ids: Optional[torch.Tensor] = None,
               mut_mask: Optional[torch.Tensor] = None,
               mut_pos: Optional[torch.Tensor] = None,
               wt_tok_id: Optional[torch.Tensor] = None,
               mut_tok_id: Optional[torch.Tensor] = None,
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backbone pass -> ``(esm_block [B, esm_block_dim], pllr [B])``.

        Separated from :meth:`classify` so that (a) the prior branch and the
        PLLR residual are head-local concerns, and (b) when the backbone is
        frozen this output is constant across epochs and can be computed once
        (see :func:`fit_esm_finetune`'s precompute path).
        """
        h_wt = self._encode(wt_ids, wt_mask)
        batch_idx = torch.arange(h_wt.shape[0], device=h_wt.device)
        # +1: token index 0 is <cls>; wt_pos is 0-based within the raw window.
        site_wt = h_wt[batch_idx, wt_pos + 1]
        if self.mode == "siamese":
            if mut_ids is None or mut_mask is None or mut_pos is None:
                raise ValueError("mode='siamese' requires mut_ids/mut_mask/mut_pos.")
            h_mut = self._encode(mut_ids, mut_mask)
            site_mut = h_mut[batch_idx, mut_pos + 1]
            diff = site_mut - site_wt
            block = torch.cat([site_wt, site_mut, diff, diff.abs()], dim=-1)
        else:
            block = site_wt

        if self.pllr_mode == "off":
            return block, torch.zeros(block.shape[0], device=block.device,
                                      dtype=torch.float32)
        if wt_tok_id is None or mut_tok_id is None:
            raise ValueError(
                "PLLR is enabled but wt_tok_id/mut_tok_id are missing. Pass "
                "tokenizer=model.tokenizer to build_examples().")
        if bool((wt_tok_id == mut_tok_id).any()):
            # Synonymous substitutions are dropped upstream, so wt and mut ids
            # can never legitimately coincide. Equal ids mean build_examples()
            # ran without a tokenizer and left them at the default 0, which
            # would make PLLR a constant zero read off a special token --
            # silently disabling the very term this path exists to supply.
            raise ValueError(
                "wt_tok_id == mut_tok_id for at least one example; "
                "build_examples() was called without tokenizer=..., so "
                "residue ids were never resolved.")
        # delta = log P(mut | X) - log P(wt | X), read off the SAME wild-type
        # forward pass that produced site_wt (Meier et al. 2021: both
        # pseudo-likelihoods share one context, so no extra pass is needed).
        # This is the zero-shot ESM score -- on its own it reaches ROC-AUC
        # 0.834 pooled across these genes.
        site_logits = self.lm_head(h_wt[batch_idx, wt_pos + 1].unsqueeze(1))
        log_probs = torch.log_softmax(site_logits.float().squeeze(1), dim=-1)
        pllr = (log_probs.gather(1, mut_tok_id.view(-1, 1))
                - log_probs.gather(1, wt_tok_id.view(-1, 1))).squeeze(-1)
        return block, pllr

    def classify(self, esm_block: torch.Tensor, pllr: torch.Tensor,
                 priors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``(esm_block, pllr[, priors])`` -> logits ``[B]``."""
        feat = self.feat_norm(esm_block)
        if self.pllr_mode == "concat":
            feat = torch.cat([feat, (pllr / self.pllr_scale).unsqueeze(-1)], dim=-1)
        logit = self.out(self.head(feat)).squeeze(-1)
        if self.pllr_mode == "residual":
            logit = logit + self.pllr_gain * (pllr / self.pllr_scale)
        return logit

    def forward(self, wt_ids: torch.Tensor, wt_mask: torch.Tensor, wt_pos: torch.Tensor,
                mut_ids: Optional[torch.Tensor] = None,
                mut_mask: Optional[torch.Tensor] = None,
                mut_pos: Optional[torch.Tensor] = None,
                wt_tok_id: Optional[torch.Tensor] = None,
                mut_tok_id: Optional[torch.Tensor] = None,
                priors: Optional[torch.Tensor] = None) -> torch.Tensor:
        block, pllr = self.encode(wt_ids, wt_mask, wt_pos, mut_ids, mut_mask,
                                  mut_pos, wt_tok_id, mut_tok_id)
        return self.classify(block, pllr, priors)


# --------------------------------------------------------------------------- #
# Data: windowed examples + collate
# --------------------------------------------------------------------------- #
@dataclass
class FineTuneExample:
    wt_seq: str
    wt_pos0: int
    mut_seq: Optional[str]
    mut_pos0: Optional[int]
    label: float
    weight: float
    #: Vocabulary ids of the wild-type and substituted residues, needed for the
    #: PLLR term. Resolved once at example-build time rather than per batch.
    wt_tok_id: int = 0
    mut_tok_id: int = 0


def build_examples(
    df: pd.DataFrame, sequence_by_gene: Dict[str, str], mode: str,
    max_residues: int = 1022, tokenizer=None,
) -> List[FineTuneExample]:
    """Crop each variant to a mutation-centred window and prepare WT/VT text.

    ``df`` needs ``gene, position (1-based), mut_aa, label``; ``label_weight``
    is optional (defaults to 1.0). Rows whose gene has no entry in
    *sequence_by_gene*, or whose ``wt_aa`` disagrees with the canonical
    sequence at that position, are dropped with a warning (same contract as
    :func:`src.esm_extractor.validate_and_align`).
    """
    examples: List[FineTuneExample] = []
    n_dropped = 0
    has_weight = "label_weight" in df.columns
    # Residue -> vocabulary id, resolved once. Without a tokenizer the ids stay
    # 0 and the caller must not enable the PLLR term.
    aa_to_id: Dict[str, int] = {}
    if tokenizer is not None:
        aa_to_id = {aa: int(tokenizer.convert_tokens_to_ids(aa))
                    for aa in "ACDEFGHIKLMNPQRSTVWY"}
    for row in df.itertuples(index=False):
        gene = str(getattr(row, "gene"))
        seq = sequence_by_gene.get(gene)
        pos = int(getattr(row, "position"))
        wt_aa = str(getattr(row, "wt_aa"))
        mut_aa = str(getattr(row, "mut_aa"))
        pos0 = pos - 1
        if seq is None or not (0 <= pos0 < len(seq)) or seq[pos0] != wt_aa:
            n_dropped += 1
            continue
        start, end = centered_window_bounds(len(seq), pos0, max_residues=max_residues)
        wt_window = seq[start:end]
        local_pos = pos0 - start
        mut_seq = mut_pos0 = None
        if mode == "siamese":
            mut_seq = wt_window[:local_pos] + mut_aa + wt_window[local_pos + 1:]
            mut_pos0 = local_pos
        weight = float(getattr(row, "label_weight")) if has_weight else 1.0
        examples.append(FineTuneExample(
            wt_seq=wt_window, wt_pos0=local_pos, mut_seq=mut_seq, mut_pos0=mut_pos0,
            label=float(getattr(row, "label")), weight=weight,
            wt_tok_id=aa_to_id.get(wt_aa, 0), mut_tok_id=aa_to_id.get(mut_aa, 0)))
    if n_dropped:
        logger.warning("build_examples: dropped %d/%d rows (unknown gene or "
                       "wt/position mismatch against the canonical sequence).",
                       n_dropped, len(df))
    return examples


class FineTuneDataset(Dataset):
    def __init__(self, examples: Sequence[FineTuneExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> FineTuneExample:
        return self.examples[idx]


def make_collate_fn(tokenizer, mode: str):
    def collate(batch: List[FineTuneExample]) -> Dict[str, torch.Tensor]:
        wt_tok = tokenizer([b.wt_seq for b in batch], return_tensors="pt", padding=True)
        out = {
            "wt_ids": wt_tok["input_ids"],
            "wt_mask": wt_tok["attention_mask"],
            "wt_pos": torch.tensor([b.wt_pos0 for b in batch], dtype=torch.long),
            "labels": torch.tensor([b.label for b in batch], dtype=torch.float32),
            "weights": torch.tensor([b.weight for b in batch], dtype=torch.float32),
            "wt_tok_id": torch.tensor([b.wt_tok_id for b in batch], dtype=torch.long),
            "mut_tok_id": torch.tensor([b.mut_tok_id for b in batch], dtype=torch.long),
        }
        if mode == "siamese":
            mut_tok = tokenizer([b.mut_seq for b in batch], return_tensors="pt", padding=True)
            out["mut_ids"] = mut_tok["input_ids"]
            out["mut_mask"] = mut_tok["attention_mask"]
            out["mut_pos"] = torch.tensor([b.mut_pos0 for b in batch], dtype=torch.long)
        return out
    return collate


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _forward(model: ESMFineTuneClassifier, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    tok = {"wt_tok_id": batch.get("wt_tok_id"),
           "mut_tok_id": batch.get("mut_tok_id")}
    if model.mode == "siamese":
        return model(batch["wt_ids"], batch["wt_mask"], batch["wt_pos"],
                     batch["mut_ids"], batch["mut_mask"], batch["mut_pos"], **tok)
    return model(batch["wt_ids"], batch["wt_mask"], batch["wt_pos"], **tok)


# --------------------------------------------------------------------------- #
# Training / inference
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict_proba(model: ESMFineTuneClassifier, examples: Sequence[FineTuneExample],
                  device: torch.device, batch_size: int = 16,
                  amp: bool = True) -> np.ndarray:
    """Batched, no-grad calibrated-free probability prediction (sigmoid of logit)."""
    model.eval()
    amp_dtype = amp_dtype_for(device) if amp else None
    loader = DataLoader(FineTuneDataset(examples), batch_size=batch_size, shuffle=False,
                        collate_fn=make_collate_fn(model.tokenizer, model.mode))
    out: List[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            batch = _to_device(batch, device)
            with _autocast(device, amp_dtype):
                logits = _forward(model, batch)
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def fit_esm_finetune(
    model: ESMFineTuneClassifier,
    train_examples: Sequence[FineTuneExample],
    val_examples: Sequence[FineTuneExample],
    device: torch.device,
    backbone_lr: float = 1e-5,
    head_lr: float = 3e-4,
    epochs: int = 10,
    patience: int = 3,
    batch_size: int = 8,
    weight_decay: float = 1e-2,
    seed: int = 42,
    max_grad_norm: float = 1.0,
    grad_accum_steps: int = 1,
    amp: bool = True,
) -> Tuple[ESMFineTuneClassifier, int, float]:
    """Fine-tune with two AdamW parameter groups (ProPath's LR split).

    Early-stops on validation ROC-AUC (best-weight restoration), mirroring
    :func:`src.transfer.fit_head`'s contract so both are drop-in comparable
    inside ``scripts/compare_finetune_strategies.py``.

    ``batch_size`` is the *micro*-batch that has to fit in VRAM;
    ``grad_accum_steps`` multiplies it into the effective batch the optimizer
    sees, so a card that can only hold 2 examples at a time still trains at
    ProPath's effective batch of 8 (``batch_size=2, grad_accum_steps=4``).
    ``amp`` runs the forward/backward under autocast (see
    :func:`amp_dtype_for`); with fp16 a :class:`torch.amp.GradScaler` is used,
    with bf16 none is needed.
    """
    from sklearn.metrics import roc_auc_score

    if grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1; got {grad_accum_steps}")

    set_seed(seed)
    model.to(device)

    amp_dtype = amp_dtype_for(device) if amp else None
    # bf16 has fp32's exponent range, so gradients cannot underflow the way
    # fp16's can -- a scaler is only needed for the fp16 path.
    scaler = (torch.amp.GradScaler(device.type)
              if amp_dtype is torch.float16 else None)
    if amp_dtype is not None:
        logger.info("Mixed precision: autocast %s%s | micro-batch %d x %d "
                    "accumulation steps = effective batch %d",
                    str(amp_dtype).replace("torch.", ""),
                    " + GradScaler" if scaler is not None else "",
                    batch_size, grad_accum_steps, batch_size * grad_accum_steps)

    param_groups = []
    backbone_params = model.trainable_backbone_params()
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    param_groups.append({"params": model.head_params(), "lr": head_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    train_loader = DataLoader(
        FineTuneDataset(train_examples), batch_size=batch_size, shuffle=True,
        collate_fn=make_collate_fn(model.tokenizer, model.mode))
    val_labels = np.array([e.label for e in val_examples], dtype=np.float32)

    clip_params = [p for g in param_groups for p in g["params"]]
    n_batches = len(train_loader)
    best_auc, best_state, best_epoch, left = -np.inf, None, -1, patience
    for epoch in range(epochs):
        epoch_t0 = time.time()
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = _to_device(batch, device)
            with _autocast(device, amp_dtype):
                logits = _forward(model, batch)
            # BCE in fp32 regardless of autocast: the loss reduction is where
            # reduced precision actually costs accuracy, and it is free here.
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                logits.float(), batch["labels"], reduction="none")
                * batch["weights"]).mean()
            running += float(loss.item()) * len(batch["labels"])
            # Average over the accumulation window so the effective batch's
            # gradient matches what a single batch_size*grad_accum_steps step
            # would have produced. The divisor is the *current* window's
            # length, not grad_accum_steps: an epoch whose batch count is not
            # a multiple of grad_accum_steps ends in a short window, and
            # dividing that by the full step count would silently shrink its
            # learning rate.
            window_start = ((step - 1) // grad_accum_steps) * grad_accum_steps
            window = min(grad_accum_steps, n_batches - window_start)
            scaled = loss / window
            (scaler.scale(scaled) if scaler is not None else scaled).backward()
            # The tail of an epoch rarely divides evenly; step on it anyway
            # rather than dropping those gradients on the floor.
            if step % grad_accum_steps == 0 or step == n_batches:
                if scaler is not None:
                    scaler.unscale_(optimizer)   # clip real, not scaled, norms
                torch.nn.utils.clip_grad_norm_(clip_params, max_grad_norm)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        val_probs = predict_proba(model, val_examples, device,
                                  batch_size=batch_size, amp=amp)
        try:
            val_auc = float(roc_auc_score(val_labels, val_probs)) \
                if len(np.unique(val_labels)) > 1 else float("nan")
        except ValueError:
            val_auc = float("nan")
        if not np.isfinite(val_auc):
            val_auc = -np.inf
        # Wall clock per epoch, plus what the remaining cap would cost. At
        # micro-batch 1 with gradient checkpointing an epoch is minutes, not
        # seconds, so "how long will this take" is a question worth answering
        # from the first epoch rather than by watching the log all evening.
        # The projection is the ceiling: early stopping usually beats it.
        epoch_s = time.time() - epoch_t0
        logger.info("epoch %d/%d: train_loss=%.4f val_auc=%.4f | %.1fs "
                    "(<= %.1f min left this split at %d epochs)",
                    epoch + 1, epochs, running / max(1, len(train_examples)),
                    val_auc, epoch_s, epoch_s * (epochs - epoch - 1) / 60, epochs)

        if val_auc > best_auc:
            best_auc, best_epoch, left = val_auc, epoch + 1, patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            left -= 1
            if left <= 0:
                logger.info("Early stopping at epoch %d (best val_auc=%.4f @ epoch %d).",
                            epoch + 1, best_auc, best_epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, max(best_epoch, 0), float(best_auc)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
#: Bumped if the checkpoint payload layout changes incompatibly.
FINETUNE_CHECKPOINT_FORMAT = "esm_finetune/v1"


def save_finetuned(
    path, model: ESMFineTuneClassifier, *,
    threshold: Optional[float] = None,
    metrics: Optional[Dict[str, object]] = None,
    max_residues: Optional[int] = None,
    extra: Optional[Dict[str, object]] = None,
) -> Path:
    """Persist a fine-tuned model so its backbone weights survive the run.

    :func:`fit_esm_finetune` restores the best epoch's weights *in memory* and
    returns the model; nothing on disk changes. Without this call the
    fine-tuned backbone -- the actual output of stage 2b -- is gone the moment
    the process exits, leaving only the metrics CSV.

    The payload is self-describing: it carries every constructor argument
    (:meth:`ESMFineTuneClassifier.config`), the MCC-optimal ``threshold``, and
    the holdout ``metrics``, so :func:`load_finetuned_model` can rebuild and
    score with it without the original CLI arguments.
    """
    from .transfer import save_checkpoint

    cfg = dict(model.config())
    if max_residues is not None:
        cfg["max_residues"] = int(max_residues)
    payload_extra: Dict[str, object] = {"format": FINETUNE_CHECKPOINT_FORMAT}
    if threshold is not None:
        payload_extra["threshold"] = float(threshold)
    if metrics:
        payload_extra["metrics"] = dict(metrics)
    if extra:
        payload_extra.update(extra)
    return save_checkpoint(Path(path), model, None, None,
                           feature_columns=[], cfg_dict=cfg, extra=payload_extra)


def load_finetuned_model(path, device: Optional[torch.device] = None, *,
                         strict: bool = True) -> Tuple[ESMFineTuneClassifier, Dict]:
    """Rebuild an :class:`ESMFineTuneClassifier` from :func:`save_finetuned`
    output and load its fine-tuned weights.

    Returns ``(model, payload)``; ``payload`` is the full checkpoint dict
    (``config``, ``threshold``, ``metrics``). The base ESM checkpoint named in
    the config is loaded fresh for its architecture, then every weight is
    overwritten by the fine-tuned state dict, so the base download only has to
    still be resolvable -- it does not affect the result.
    """
    from .transfer import load_checkpoint

    payload = load_checkpoint(Path(path))
    cfg = payload["config"]
    # Checkpoints written before pllr_mode existed carry a use_pllr bool. The
    # behaviour it described is exactly "concat", never "residual".
    pllr_mode = cfg.get("pllr_mode")
    if pllr_mode is None:
        pllr_mode = "concat" if cfg.get("use_pllr", True) else "off"
    model = ESMFineTuneClassifier(
        model_name=cfg["model_name"], mode=cfg["mode"],
        n_unfrozen_layers=cfg.get("n_unfrozen_layers", -1),
        hidden_dim=cfg.get("hidden_dim", 256),
        dropout=cfg.get("dropout", 0.15),
        pllr_mode=pllr_mode)
    model.load_state_dict(payload["model_state_dict"], strict=strict)
    if device is not None:
        model.to(device)
    model.eval()
    return model, payload


__all__ = [
    "FINETUNE_MODES", "PLLR_MODES", "PROPATH_DEFAULTS", "FINETUNE_CHECKPOINT_FORMAT",
    "ESMFineTuneClassifier", "FineTuneExample", "FineTuneDataset",
    "build_examples", "make_collate_fn", "predict_proba", "fit_esm_finetune",
    "set_seed", "amp_dtype_for", "save_finetuned", "load_finetuned_model",
]
