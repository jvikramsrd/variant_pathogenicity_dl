"""Parameter-efficient tuning for SLM backbones — the DL branch's strategies, reused.

BERT-family encoders have the layout ``vpdl.dl.plm.peft.apply_strategy``
already handles (``encoder.layer[i].attention.self.query/value``), so they
go straight to it: same strategies (frozen, lora, adapters, last_n, partial,
full), same refusal to fully fine-tune a large model on few labels, same
parameter summary. Decoders (Llama layout: ``layers[i].self_attn.q_proj``)
get the same strategies here, implemented the same way.
"""

from __future__ import annotations

import math
from typing import Any

from vpdl.dl.plm.peft import FinetuneStrategy, apply_strategy, count_parameters

__all__ = ["apply_peft", "FinetuneStrategy", "count_parameters"]

DECODER_LORA_TARGETS = ("q_proj", "v_proj")


def _decoder_lora_class():
    import torch
    from torch import nn

    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
            super().__init__()
            self.base = base
            self.lora_a = nn.Linear(base.in_features, rank, bias=False)
            self.lora_b = nn.Linear(rank, base.out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b.weight)
            self.scale = alpha / rank
            self.dropout = nn.Dropout(dropout)
            for p in self.base.parameters():
                p.requires_grad = False

        @property
        def weight(self):
            return self.base.weight

        def forward(self, x):
            return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scale

        def merge(self) -> None:
            with torch.no_grad():
                self.base.weight += (self.lora_b.weight @ self.lora_a.weight) * self.scale
                self.lora_b.weight.zero_()

    return LoRALinear


def apply_peft(model, strategy: FinetuneStrategy | dict | str) -> dict[str, Any]:
    strategy = FinetuneStrategy.from_dict(strategy)
    if hasattr(getattr(model, "encoder", None), "layer"):
        return apply_strategy(model, strategy)
    if not hasattr(model, "layers"):
        raise ValueError(f"{type(model).__name__}: neither an encoder (encoder.layer) nor a "
                         "decoder (layers) layout; cannot apply a tuning strategy")
    from torch import nn
    total = sum(p.numel() for p in model.parameters())
    if strategy.kind == "full":
        if total / 1e6 > strategy.full_max_params_m and not strategy.allow_full:
            raise ValueError(f"full fine-tuning of {total / 1e6:.0f}M parameters refused; use lora or "
                             "last_n, or allow_full=True with evidence it is needed")
        for p in model.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = False
    if strategy.kind == "lora":
        targets = tuple(t for t in strategy.lora_targets if t not in ("query", "value")) or DECODER_LORA_TARGETS
        lora = _decoder_lora_class()
        chosen = [(name, module) for name, module in model.named_modules()
                  if isinstance(module, nn.Linear) and name.split(".")[-1] in targets]
        if not chosen:
            raise ValueError(f"no linear layers named {targets}")
        for name, module in chosen:
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, lora(module, strategy.lora_rank, strategy.lora_alpha,
                                        strategy.lora_dropout))
    elif strategy.kind in ("last_n", "partial"):
        layers = model.layers
        for layer in layers[-max(1, min(strategy.last_n, len(layers))):]:
            for p in layer.parameters():
                p.requires_grad = True
        if getattr(model, "norm", None) is not None:
            for p in model.norm.parameters():
                p.requires_grad = True
    elif strategy.kind == "adapters":
        raise ValueError("adapters are implemented for encoders only; use lora or last_n for decoders")
    trainable, total = count_parameters(model)
    return {"strategy": strategy.as_dict(), "trainable_params": trainable, "total_params": total,
            "trainable_fraction": round(trainable / max(1, total), 6)}
