"""End-to-end PLM fine-tuning for variant classification (siamese WT / VT).

Ported from v1's ``src/esm_finetune.py`` (ProPath's siamese recipe), keeping
what its RUNLOG showed mattered and dropping the 15 GiB-card contortions:

* WT and VT windows go through the SAME backbone; the head reads
  ``[h_wt, h_vt, h_vt - h_wt, |h_vt - h_wt|]`` at the mutated residue, after a
  LayerNorm (raw ESM hidden states have large, position-dependent norms — the
  unnormalised head spent its capacity on scale, RUNLOG 2026-08-28 item 2).
* The zero-shot term enters as a skip connection past a zero-initialised head,
  so the untrained model IS the zero-shot predictor and training learns a
  correction. It is the wild-type marginal read from the WT pass (free), and
  is named that — v1 called it masked (CODEBASE_AUDIT finding 1).
* PEFT (:mod:`vpdl.dl.plm.peft`), default LoRA. Full fine-tuning of a 650M
  model is refused unless explicitly allowed.
* Windows follow a context policy; ``sliding`` is not trainable end to end (no
  single window), so fine-tuning uses ``centered``, ``asymmetric`` or ``full``.

The model is a *frame model* for :func:`vpdl.experiment.run_cell`: it takes
variant rows plus sequences, not a feature matrix.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vpdl.dl.context import site_span
from vpdl.dl.plm.peft import FinetuneStrategy, apply_strategy, trainable_state_dict
from vpdl.models.base import derive_seed

logger = logging.getLogger(__name__)

__all__ = ["PLMFinetuneClassifier", "load_adapted_backbone"]

_LLR_SCALE = 10.0


def load_adapted_backbone(backbone, adapted_dir: Path | str) -> dict[str, Any]:
    """Apply a continued-pretraining result (vpdl.dl.pretrain) to `backbone`.

    LoRA updates are merged into the base weights and the wrappers removed, so
    the adapted model is a plain backbone again and downstream fine-tuning can
    apply its own strategy on top.
    """
    import torch

    from vpdl.dl.plm.peft import merge_lora

    payload = torch.load(Path(adapted_dir) / "backbone_delta.pt", map_location="cpu",
                         weights_only=False)
    if payload.get("format") != "vpdl-dl-backbone-delta-v1":
        raise ValueError(f"{adapted_dir}: not a backbone delta (format {payload.get('format')!r})")
    if payload["backbone"] != backbone.spec.name:
        raise ValueError(f"delta was trained on {payload['backbone']}, not {backbone.spec.name}")
    strategy = FinetuneStrategy.from_dict(payload["strategy"])
    if strategy.kind == "adapters":
        raise ValueError("adapter deltas cannot be merged; pretrain with lora/last_n/partial")
    apply_strategy(backbone.model, strategy)
    missing, unexpected = backbone.model.load_state_dict(payload["state"], strict=False)
    if unexpected:
        raise ValueError(f"delta carries parameters the backbone lacks: {unexpected[:5]}")
    merge_lora(backbone.model)
    _unwrap_lora(backbone.model)
    for p in backbone.model.parameters():
        p.requires_grad = False
    corpus = payload.get("corpus_manifest") or {}
    return {"adapted_from": str(adapted_dir), "pretrain_arm": payload.get("arm"),
            "pretrain_strategy": strategy.as_dict(),
            # What the pretraining corpus held out — checked per fold by
            # vpdl.dl.runner so a strict-mode arm cannot silently use a
            # backbone that saw the held-out gene's family.
            "pretrain_corpus_mode": corpus.get("mode"),
            "pretrain_holdout": corpus.get("holdout")}


def _unwrap_lora(model) -> None:
    for name, module in list(model.named_modules()):
        if hasattr(module, "lora_a") and getattr(module, "merged", False):
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, module.base)


class PLMFinetuneClassifier:
    """Siamese fine-tuned PLM + optional tabular features -> probability."""

    def __init__(self, backbone: Any = "esm2_650m", strategy: Any = "lora",
                 context: str = "centered", n_tabular_features: int = 0, seed: int = 42,
                 split_index: int | str = 0, head_hidden: int = 256, dropout: float = 0.15,
                 lr_backbone: float = 1e-4, lr_head: float = 3e-4, weight_decay: float = 1e-2,
                 epochs: int = 10, patience: int = 3, batch_size: int = 8, grad_accum: int = 1,
                 warmup_frac: float = 0.1, precision: str = "auto",
                 gradient_checkpointing: bool = True, wt_marginal_residual: bool = True,
                 adapted_from: str | None = None, checkpoint_dir: str | None = None,
                 resume: bool = False, compile: bool = False) -> None:
        if context == "sliding":
            raise ValueError("fine-tuning needs one window per variant: use centered, "
                             "asymmetric or full (hierarchical reads its centred window)")
        self.seed = derive_seed(seed, split_index)
        from vpdl.dl.trainer import seed_everything
        seed_everything(self.seed)                      # before LoRA/head init (L7)

        import torch
        from torch import nn

        from vpdl.dl.plm.backbones import LoadedBackbone, load_backbone

        self._torch = torch
        self.backbone = backbone if isinstance(backbone, LoadedBackbone) else load_backbone(
            backbone, gradient_checkpointing=gradient_checkpointing)
        self.adaptation = (load_adapted_backbone(self.backbone, adapted_from)
                           if adapted_from else None)
        seed_everything(self.seed)
        self.peft_summary = apply_strategy(self.backbone.model, strategy)
        self.context = "centered" if context == "hierarchical" else context
        self.n_tabular = int(n_tabular_features)
        self.residual = bool(wt_marginal_residual)
        self.hyper = dict(lr_backbone=lr_backbone, lr_head=lr_head, weight_decay=weight_decay,
                          epochs=epochs, patience=patience, batch_size=batch_size,
                          grad_accum=grad_accum, warmup_frac=warmup_frac, precision=precision,
                          checkpoint_dir=checkpoint_dir, resume=resume, compile=compile)
        d = self.backbone.model.config.hidden_size
        residual = self.residual
        lm = self.backbone.model

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lm = lm
                self.norm = nn.LayerNorm(4 * d)
                self.hidden = nn.Sequential(nn.Linear(4 * d + n_tabular_features, head_hidden),
                                            nn.LayerNorm(head_hidden), nn.GELU(),
                                            nn.Dropout(dropout))
                self.out = nn.Linear(head_hidden, 1)
                if residual:
                    nn.init.zeros_(self.out.weight)
                    nn.init.zeros_(self.out.bias)
                    # Negative: raw log-ratio is negative for damaging variants.
                    self.gain = nn.Parameter(torch.tensor(-1.0))

            def features(self, batch):
                encoder = self.lm.esm
                rows = torch.arange(batch["site"].shape[0], device=batch["site"].device)
                h_wt = encoder(input_ids=batch["wt_ids"],
                               attention_mask=batch["wt_mask"]).last_hidden_state
                h_vt = encoder(input_ids=batch["vt_ids"],
                               attention_mask=batch["vt_mask"]).last_hidden_state
                site_wt, site_vt = h_wt[rows, batch["site"] + 1], h_vt[rows, batch["site"] + 1]
                diff = site_vt - site_wt
                block = self.norm(torch.cat([site_wt, site_vt, diff, diff.abs()], -1).float())
                if batch["tabular"].shape[1]:
                    block = torch.cat([block, batch["tabular"]], -1)
                return self.hidden(block), site_wt

            def forward(self, batch):
                hidden, site_wt = self.features(batch)
                logit = self.out(hidden).squeeze(-1)
                if residual:
                    log_probs = torch.log_softmax(self.lm.lm_head(site_wt).float(), -1)
                    llr = (log_probs.gather(1, batch["mut_tok"][:, None])
                           - log_probs.gather(1, batch["wt_tok"][:, None])).squeeze(-1)
                    logit = logit + self.gain * llr / _LLR_SCALE
                return logit

        self.model = _Model()

    # -- data -----------------------------------------------------------------
    def _examples(self, frame: pd.DataFrame, sequences: Mapping[str, str]) -> list[tuple]:
        spec = self.backbone.spec
        examples = []
        for row in frame.itertuples():
            sequence = sequences[row.uniprot_id]
            p0 = int(row.position) - 1
            if sequence[p0] != row.wt_aa:
                raise ValueError(f"{row.uniprot_id}:{row.position} wild-type mismatch")
            start, end = site_span(self.context, len(sequence), p0, spec.max_residues,
                                   spec.supports_full_length)
            window = sequence[start:end]
            local = p0 - start
            examples.append((window, window[:local] + row.mut_aa + window[local + 1:], local,
                             row.wt_aa, row.mut_aa))
        return examples

    def _batch(self, examples: Sequence[tuple], tabular: np.ndarray | None, index, device):
        torch = self._torch
        alphabet = self.backbone.alphabet
        chosen = [examples[i] for i in index]
        wt_ids, wt_mask = alphabet.encode([e[0] for e in chosen])
        vt_ids, vt_mask = alphabet.encode([e[1] for e in chosen])
        tab = (torch.tensor(np.asarray(tabular, dtype=np.float32)[index])
               if tabular is not None and self.n_tabular else torch.zeros((len(chosen), 0)))
        return {"wt_ids": wt_ids.to(device), "wt_mask": wt_mask.to(device),
                "vt_ids": vt_ids.to(device), "vt_mask": vt_mask.to(device),
                "site": torch.tensor([e[2] for e in chosen], device=device),
                "wt_tok": torch.tensor([alphabet.index[e[3]] for e in chosen], device=device),
                "mut_tok": torch.tensor([alphabet.index[e[4]] for e in chosen], device=device),
                "tabular": tab.to(device)}

    # -- fit / predict ------------------------------------------------------------
    def fit_frames(self, frame, y, val_frame=None, y_val=None, tabular=None,
                   tabular_val=None, sequences=None):
        torch = self._torch
        from vpdl.dl.trainer import TrainConfig, Trainer
        from vpdl.evaluate import roc_auc

        examples = self._examples(frame, sequences)
        target = torch.tensor(np.asarray(y, dtype=np.float32))
        positive = float(target.sum())
        pos_weight = (len(target) - positive) / positive if positive else 1.0
        h = self.hyper
        backbone_params = [p for p in self.model.lm.parameters() if p.requires_grad]
        head_params = [p for n, p in self.model.named_parameters()
                       if not n.startswith("lm.") and p.requires_grad]
        groups = ([{"params": backbone_params, "lr": h["lr_backbone"]}] if backbone_params
                  else []) + [{"params": head_params, "lr": h["lr_head"]}]
        trainer = Trainer(self.model, TrainConfig(
            epochs=h["epochs"], patience=h["patience"], lr=h["lr_head"],
            weight_decay=h["weight_decay"], batch_size=h["batch_size"],
            grad_accum=h["grad_accum"], warmup_frac=h["warmup_frac"], schedule="cosine",
            precision=h["precision"], compile=h["compile"],
            checkpoint_dir=h["checkpoint_dir"], resume=h["resume"],
            # As in vpdl.dl.pretrain.run: store what trains (LoRA/adapters/unfrozen layers/head),
            # not two copies of the frozen backbone per epoch.
            checkpoint_trainable_only=True,
            extra={"peft": self.peft_summary["strategy"], "context": self.context,
                   "backbone": self.backbone.spec.name}),
            seed=self.seed, param_groups=groups, run_name=f"plm-{self.backbone.spec.name}")
        device = trainer.device
        weight = torch.tensor([pos_weight], device=device)

        def loss_fn(model, batch):
            logits = model(batch)
            return torch.nn.functional.binary_cross_entropy_with_logits(
                logits.float(), batch["target"], pos_weight=weight)

        def batch_fn(index):
            batch = self._batch(examples, tabular, index, device)
            batch["target"] = target[torch.as_tensor(index, dtype=torch.long)].to(device)
            return batch

        score_fn = None
        if val_frame is not None and y_val is not None and len(y_val):
            val_examples = self._examples(val_frame, sequences)

            def score_fn(model):
                return roc_auc(np.asarray(y_val),
                               self._predict_examples(val_examples, tabular_val, model))
        self.fit_result = trainer.fit(len(examples), batch_fn, loss_fn, score_fn)
        return self

    def _predict_examples(self, examples, tabular, model=None, batch_size: int = 32,
                          represent: bool = False) -> np.ndarray:
        torch = self._torch
        from vpdl.dl.plm.forward import inference_autocast

        model = model or self.model
        device = next(self.model.parameters()).device
        model.eval()
        out = []
        with torch.inference_mode(), inference_autocast(device):
            for start in range(0, len(examples), batch_size):
                index = np.arange(start, min(start + batch_size, len(examples)))
                batch = self._batch(examples, tabular, index, device)
                if represent:
                    out.append(self.model.features(batch)[0].float().cpu().numpy())
                else:
                    out.append(torch.sigmoid(model(batch).float()).cpu().numpy())
        if not out:
            return np.zeros((0,) if not represent else (0, 0))
        return np.concatenate(out)

    def predict_frames(self, frame, tabular=None, sequences=None) -> np.ndarray:
        return self._predict_examples(self._examples(frame, sequences), tabular)

    def represent_frames(self, frame, tabular=None, sequences=None) -> np.ndarray:
        """Penultimate head activations — the DL embedding for integration."""
        return self._predict_examples(self._examples(frame, sequences), tabular,
                                      represent=True)

    def save(self, path: Path | str) -> Path:
        torch = self._torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": "vpdl-dl-plm-finetune-v1", "backbone": self.backbone.spec.name,
                    "backbone_version": self.backbone.version, "peft": self.peft_summary,
                    "context": self.context, "adaptation": self.adaptation,
                    "trainable": trainable_state_dict(self.model)}, path)
        return path
