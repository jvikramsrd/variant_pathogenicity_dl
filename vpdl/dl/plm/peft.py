"""Parameter-efficient fine-tuning for ESM backbones.

Strategies, most conservative first — start at the top and move down only when
a validation result says the data supports it:

    frozen    no backbone parameter trains (a probe; embeddings can be cached)
    lora      low-rank updates on chosen linear layers (default: query, value)
    adapters  bottleneck adapters after attention and FFN outputs (Houlsby),
              zero-initialised so the model starts exactly as pretrained
    last_n    the top N transformer layers (and final layer norm) train
    partial   last_n + every LayerNorm in the backbone
    full      everything — refused above `full_max_params_m` unless
              `allow_full=True`, because ~700 clinical labels cannot constrain
              650M parameters and the failure is silent (it just overfits)

Implemented natively (~100 lines) rather than through ``peft``: no version
coupling with transformers on aarch64, and the wrapped modules are plain
``nn.Linear`` replacements the tests can inspect.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["STRATEGIES", "FinetuneStrategy", "apply_strategy", "trainable_state_dict",
           "merge_lora", "encoder_layers", "count_parameters"]

STRATEGIES = ("frozen", "lora", "adapters", "last_n", "partial", "full")


@dataclass
class FinetuneStrategy:
    kind: str = "frozen"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = ("query", "value")
    adapter_dim: int = 64
    last_n: int = 2
    allow_full: bool = False
    full_max_params_m: int = 150
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: "FinetuneStrategy | dict | str | None") -> "FinetuneStrategy":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if isinstance(value, str):
            return cls(kind=value)
        value = dict(value)
        if "lora_targets" in value:
            value["lora_targets"] = tuple(value["lora_targets"])
        return cls(**value)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _nn():
    import torch
    from torch import nn
    return torch, nn


def _lora_linear_class():
    torch, nn = _nn()

    class LoRALinear(nn.Module):
        """``base(x) + B(A(dropout(x))) * alpha / r`` with the base frozen, B zero-init."""

        def __init__(self, base: "nn.Linear", rank: int, alpha: float, dropout: float) -> None:
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad = False
            self.rank, self.scale = rank, alpha / rank
            self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
            self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            self.dropout = nn.Dropout(dropout)
            self.merged = False

        @property
        def weight(self):
            return self.base.weight

        @property
        def bias(self):
            return self.base.bias

        def forward(self, x):
            out = self.base(x)
            if self.merged:
                return out
            update = self.dropout(x) @ self.lora_a.t() @ self.lora_b.t()
            return out + update.to(out.dtype) * self.scale

        def merge(self) -> None:
            if not self.merged:
                with torch.no_grad():
                    self.base.weight += (self.lora_b @ self.lora_a).to(self.base.weight.dtype) \
                        * self.scale
                self.merged = True

    return LoRALinear


def _adapter_linear_class():
    torch, nn = _nn()

    class AdapterLinear(nn.Module):
        """``y = base(x); y + up(gelu(down(y)))`` with `up` zero-init (identity at start)."""

        def __init__(self, base: "nn.Linear", bottleneck: int) -> None:
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad = False
            self.down = nn.Linear(base.out_features, bottleneck)
            self.up = nn.Linear(bottleneck, base.out_features)
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

        @property
        def weight(self):
            return self.base.weight

        def forward(self, x):
            y = self.base(x)
            return y + self.up(torch.nn.functional.gelu(self.down(y)))

    return AdapterLinear


def encoder_layers(model):
    encoder = getattr(model, "esm", model)
    return encoder.encoder.layer


def count_parameters(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def _replace(module, name: str, new) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = module.get_submodule(parent_name) if parent_name else module
    setattr(parent, child, new)


def apply_strategy(model, strategy: FinetuneStrategy | dict | str) -> dict[str, Any]:
    """Freeze/wrap `model` (an EsmForMaskedLM or EsmModel) in place.

    Returns a summary with trainable/total counts, recorded in every run.
    """
    torch, nn = _nn()
    strategy = FinetuneStrategy.from_dict(strategy)
    if strategy.kind not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy.kind!r}; known {STRATEGIES}")
    encoder = getattr(model, "esm", model)
    total = sum(p.numel() for p in model.parameters())

    if strategy.kind == "full":
        if total / 1e6 > strategy.full_max_params_m and not strategy.allow_full:
            raise ValueError(
                f"full fine-tuning of a {total / 1e6:.0f}M-parameter backbone refused: the "
                f"labelled set is ~10^3 variants. Use lora/adapters/last_n, or pass "
                f"allow_full=True with the validation evidence that justifies it.")
        for p in model.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = False

    layers = encoder.encoder.layer
    if strategy.kind == "lora":
        lora = _lora_linear_class()
        targets = [(name, m) for name, m in encoder.named_modules()
                   if isinstance(m, nn.Linear) and ".layer." in f".{name}"
                   and name.split(".")[-1] in strategy.lora_targets]
        if not targets:
            raise ValueError(f"no linear layers named {strategy.lora_targets} in the encoder")
        for name, module in targets:
            _replace(encoder, name, lora(module, strategy.lora_rank, strategy.lora_alpha,
                                         strategy.lora_dropout))
    elif strategy.kind == "adapters":
        adapter = _adapter_linear_class()
        for index, layer in enumerate(layers):
            for path in ("attention.output.dense", "output.dense"):
                _replace(layer, path, adapter(layer.get_submodule(path), strategy.adapter_dim))
    elif strategy.kind in ("last_n", "partial"):
        n = max(1, min(strategy.last_n, len(layers)))
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
        final_norm = getattr(encoder.encoder, "emb_layer_norm_after", None)
        if final_norm is not None:
            for p in final_norm.parameters():
                p.requires_grad = True
        if strategy.kind == "partial":
            for module in encoder.modules():
                if isinstance(module, nn.LayerNorm):
                    for p in module.parameters():
                        p.requires_grad = True

    trainable, total = count_parameters(model)
    summary = {"strategy": strategy.as_dict(), "trainable_params": trainable,
               "total_params": total, "trainable_fraction": round(trainable / max(1, total), 6)}
    logger.info("fine-tune strategy %s: %d / %d parameters trainable (%.4f%%)",
                strategy.kind, trainable, total, 100 * trainable / max(1, total))
    return summary


def trainable_state_dict(model) -> dict[str, Any]:
    """Only what training changed — a LoRA checkpoint is megabytes, not gigabytes."""
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in names}


def merge_lora(model) -> int:
    """Fold every LoRA update into its base weight; returns how many were merged."""
    merged = 0
    for module in model.modules():
        if hasattr(module, "merge") and hasattr(module, "lora_a"):
            module.merge()
            merged += 1
    return merged
