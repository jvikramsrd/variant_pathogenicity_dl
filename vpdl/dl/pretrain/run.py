"""Pretraining arms P0-P4 and the loop that runs one.

    P0  the original pretrained model — no training; the reference
    P1  continued MLM on the MMR corpus
    P2  MLM + mutation objectives (position + substitution)
    P3  MLM + WT/VT contrastive
    P4  combined: P2 + P3

Every arm writes ``backbone_delta.pt`` (only what training changed) that
:func:`vpdl.dl.plm.finetune.load_adapted_backbone` applies, then the SAME
downstream steps run for every arm: embeddings -> feature store -> probe /
fusion / fine-tune cells through :func:`vpdl.experiment.run_cell`, scored on
the same held-out ClinVar variants. A pretraining arm that loses to P0 there
is a documented regression, not a deleted result.

Early stopping watches the held-out corpus MLM loss (label-free), never a
clinical metric.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vpdl.dl.plm.peft import FinetuneStrategy, apply_strategy, trainable_state_dict
from vpdl.dl.pretrain.corpus import read_fasta
from vpdl.dl.pretrain.objectives import ObjectiveWeights, build_pretrain_model, mlm_mask
from vpdl.models.base import derive_seed

logger = logging.getLogger(__name__)

__all__ = ["PRETRAIN_ARMS", "PretrainConfig", "pretrain", "BACKBONE_DELTA_FORMAT"]

BACKBONE_DELTA_FORMAT = "vpdl-dl-backbone-delta-v1"

PRETRAIN_ARMS: dict[str, dict[str, float]] = {
    "P0": {},
    "P1": {"mlm": 1.0},
    "P2": {"mlm": 1.0, "mutation_position": 0.5, "substitution": 0.5},
    "P3": {"mlm": 1.0, "contrastive": 0.5},
    "P4": {"mlm": 1.0, "mutation_position": 0.5, "substitution": 0.5, "contrastive": 0.5},
}


@dataclass
class PretrainConfig:
    arm: str = "P1"
    backbone: str = "esm2_650m"
    corpus_dir: str = "data/dl/corpus/strict-MLH1"
    objectives: dict[str, float] = field(default_factory=dict)   # overrides the arm
    strategy: dict[str, Any] = field(default_factory=lambda: {
        "kind": "lora", "lora_rank": 16, "lora_alpha": 32.0,
        "lora_targets": ["query", "key", "value", "dense"]})
    crop: int = 512
    epochs: int = 20
    patience: int = 3
    batch_size: int = 16
    grad_accum: int = 2
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    precision: str = "auto"
    compile: bool = False
    gradient_checkpointing: bool = True
    seed: int = 0

    def weights(self) -> ObjectiveWeights:
        if self.arm not in PRETRAIN_ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; known {sorted(PRETRAIN_ARMS)}")
        return ObjectiveWeights.from_dict(PRETRAIN_ARMS[self.arm] | dict(self.objectives))


def _crops(sequences: list[str], length: int, rng: np.random.Generator) -> list[str]:
    out = []
    for sequence in sequences:
        if len(sequence) <= length:
            out.append(sequence)
        else:
            start = int(rng.integers(0, len(sequence) - length + 1))
            out.append(sequence[start:start + length])
    return out


def pretrain(config: PretrainConfig, out_dir: Path | str, backbone=None,
             resume: bool = False) -> dict[str, Any]:
    """Run one arm; returns the summary written to ``pretrain_summary.json``."""
    import torch

    from vpdl.dl.plm.backbones import load_backbone
    from vpdl.dl.trainer import TrainConfig, Trainer, seed_everything

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = config.weights()
    corpus = Path(config.corpus_dir)
    manifest = json.loads((corpus / "corpus_manifest.json").read_text())
    summary: dict[str, Any] = {"arm": config.arm, "config": asdict(config),
                               "objectives": asdict(weights), "corpus": manifest}
    if not weights.active():
        summary["note"] = "P0: the original model; nothing trained, no delta written."
        (out_dir / "pretrain_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return summary

    seed = derive_seed(config.seed, f"pretrain:{config.arm}:{manifest.get('holdout')}")
    seed_everything(seed)
    backbone = backbone or load_backbone(config.backbone,
                                         gradient_checkpointing=config.gradient_checkpointing)
    strategy = FinetuneStrategy.from_dict(config.strategy)
    if strategy.kind == "adapters":
        raise ValueError("pretraining deltas must be mergeable: use lora, last_n or partial")
    peft = apply_strategy(backbone.model, strategy)
    model = build_pretrain_model(backbone, weights)

    train = list(read_fasta(corpus / "train.fasta").values())
    val = list(read_fasta(corpus / "val.fasta").values())
    if not train or not val:
        raise ValueError(f"{corpus}: empty train or val split")
    val_rng = np.random.default_rng(derive_seed(seed, "val"))
    val_crops = _crops(val, config.crop, val_rng)
    val_generator = torch.Generator().manual_seed(derive_seed(seed, "val-mask"))
    val_ids, val_att = backbone.alphabet.encode(val_crops)
    val_inputs, val_labels = mlm_mask(val_ids, val_att, backbone.alphabet, weights.mask_rate,
                                      val_generator)          # fixed masks: comparable epochs

    trainer = Trainer(model, TrainConfig(
        epochs=config.epochs, patience=config.patience, lr=config.lr,
        weight_decay=config.weight_decay, batch_size=config.batch_size,
        grad_accum=config.grad_accum, warmup_frac=config.warmup_frac, schedule="cosine",
        precision=config.precision, compile=config.compile,
        checkpoint_dir=str(out_dir / "checkpoints"), resume=resume,
        checkpoint_trainable_only=True,
        extra={"arm": config.arm, "objectives": asdict(weights), "peft": peft["strategy"],
               "corpus_sha256": manifest.get("sha256")}),
        seed=seed, run_name=f"pretrain-{config.arm}")
    device = trainer.device
    state: dict[str, Any] = {}

    import hashlib

    def batch_fn(index):
        # Crops, substitutions and masks are drawn from generators keyed on the
        # batch's exact contents and order (which the epoch-keyed permutation
        # fixes), so a resumed run replays identical batches.
        key = hashlib.sha256(np.asarray(index, dtype=np.int64).tobytes()).hexdigest()[:16]
        rng = np.random.default_rng(derive_seed(seed, f"batch:{key}"))
        generator = torch.Generator().manual_seed(derive_seed(seed, f"mask:{key}"))
        return _crops([train[i] for i in index], config.crop, rng), rng, generator

    def loss_fn(module, batch):
        crops, rng, generator = batch
        parts = module.losses(crops, rng, generator)
        state["last"] = {k: float(v.detach()) for k, v in parts.items()}
        return module.weighted(parts)

    def score_fn(module):
        module.eval()
        losses = []
        with torch.inference_mode():
            for start in range(0, len(val_crops), config.batch_size):
                sl = slice(start, start + config.batch_size)
                logits = module.lm(input_ids=val_inputs[sl].to(device),
                                   attention_mask=val_att[sl].to(device)).logits
                losses.append(torch.nn.functional.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    val_labels[sl].to(device).reshape(-1), ignore_index=-100).item())
        return -float(np.mean(losses))

    result = trainer.fit(len(train), batch_fn, loss_fn, score_fn)
    delta = {"format": BACKBONE_DELTA_FORMAT, "arm": config.arm, "backbone": config.backbone,
             "backbone_version": backbone.version, "strategy": strategy.as_dict(),
             "state": trainable_state_dict(backbone.model), "objectives": asdict(weights),
             "corpus_manifest": manifest, "best_val_mlm_loss": -result.best_score}
    torch.save(delta, out_dir / "backbone_delta.pt")
    summary |= {"peft": peft, "fit": asdict(result), "best_val_mlm_loss": -result.best_score,
                "val_mlm_perplexity": float(np.exp(-result.best_score)),
                "delta": str(out_dir / "backbone_delta.pt")}
    (out_dir / "pretrain_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
