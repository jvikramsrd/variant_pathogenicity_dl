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

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .mvmamba_features import centered_window_bounds

logger = logging.getLogger(__name__)

FINETUNE_MODES = ("wt_site", "siamese")

#: PROJECT_PLAN.md Phase 3 step 4's ProPath-recipe defaults.
PROPATH_DEFAULTS: Dict[str, object] = {
    "backbone_lr": 1e-5, "batch_size": 8, "epochs": 10,
}


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
                 gradient_checkpointing: bool = False) -> None:
        super().__init__()
        if mode not in FINETUNE_MODES:
            raise ValueError(f"mode must be one of {FINETUNE_MODES}; got {mode!r}")
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError(
                "transformers is required for ESM-2 fine-tuning. "
                "pip install -r requirements.txt") from exc

        self.model_name = model_name
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        self.hidden_size = int(self.backbone.config.hidden_size)

        for p in self.backbone.parameters():
            p.requires_grad = False
        if n_unfrozen_layers == -1:
            for p in self.backbone.parameters():
                p.requires_grad = True
        elif n_unfrozen_layers > 0:
            layers = self.backbone.encoder.layer
            for layer in layers[-n_unfrozen_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
        self.n_unfrozen_layers = n_unfrozen_layers
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        in_dim = self.hidden_size * (4 if mode == "siamese" else 1)
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden_dim, 1)

    def trainable_backbone_params(self) -> List[torch.nn.Parameter]:
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def head_params(self) -> List[torch.nn.Parameter]:
        return list(self.head.parameters()) + list(self.out.parameters())

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def forward(self, wt_ids: torch.Tensor, wt_mask: torch.Tensor, wt_pos: torch.Tensor,
                mut_ids: Optional[torch.Tensor] = None,
                mut_mask: Optional[torch.Tensor] = None,
                mut_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
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
            feat = torch.cat([site_wt, site_mut, diff, diff.abs()], dim=-1)
        else:
            feat = site_wt
        return self.out(self.head(feat)).squeeze(-1)


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


def build_examples(
    df: pd.DataFrame, sequence_by_gene: Dict[str, str], mode: str,
    max_residues: int = 1022,
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
            label=float(getattr(row, "label")), weight=weight))
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
    if model.mode == "siamese":
        return model(batch["wt_ids"], batch["wt_mask"], batch["wt_pos"],
                     batch["mut_ids"], batch["mut_mask"], batch["mut_pos"])
    return model(batch["wt_ids"], batch["wt_mask"], batch["wt_pos"])


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
                  device: torch.device, batch_size: int = 16) -> np.ndarray:
    """Batched, no-grad calibrated-free probability prediction (sigmoid of logit)."""
    model.eval()
    loader = DataLoader(FineTuneDataset(examples), batch_size=batch_size, shuffle=False,
                        collate_fn=make_collate_fn(model.tokenizer, model.mode))
    out: List[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            batch = _to_device(batch, device)
            logits = _forward(model, batch)
            out.append(torch.sigmoid(logits).cpu().numpy())
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
) -> Tuple[ESMFineTuneClassifier, int, float]:
    """Fine-tune with two AdamW parameter groups (ProPath's LR split).

    Early-stops on validation ROC-AUC (best-weight restoration), mirroring
    :func:`src.transfer.fit_head`'s contract so both are drop-in comparable
    inside ``scripts/compare_finetune_strategies.py``.
    """
    from sklearn.metrics import roc_auc_score

    set_seed(seed)
    model.to(device)

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

    best_auc, best_state, best_epoch, left = -np.inf, None, -1, patience
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = _forward(model, batch)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                logits, batch["labels"], reduction="none") * batch["weights"]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in param_groups for p in g["params"]], max_grad_norm)
            optimizer.step()
            running += float(loss.item()) * len(batch["labels"])

        val_probs = predict_proba(model, val_examples, device, batch_size=max(16, batch_size))
        try:
            val_auc = float(roc_auc_score(val_labels, val_probs)) \
                if len(np.unique(val_labels)) > 1 else float("nan")
        except ValueError:
            val_auc = float("nan")
        if not np.isfinite(val_auc):
            val_auc = -np.inf
        logger.info("epoch %d/%d: train_loss=%.4f val_auc=%.4f",
                    epoch + 1, epochs, running / max(1, len(train_examples)), val_auc)

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


__all__ = [
    "FINETUNE_MODES", "PROPATH_DEFAULTS",
    "ESMFineTuneClassifier", "FineTuneExample", "FineTuneDataset",
    "build_examples", "make_collate_fn", "predict_proba", "fit_esm_finetune",
    "set_seed",
]
