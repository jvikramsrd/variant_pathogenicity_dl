"""DL fusion: modality encoders -> concatenation (or gate) -> small MLP -> score.

    sequence / ESM   \\
    structure         |-- Linear+GELU projection each -> concat -> MLP -> representation -> score
    genomic           |
    population       /

Generalises v1's two-branch ``ConcatFusionHead`` / ``GateWaveFusionHead``
(src/fusion.py) to any number of named modalities. The design constraint that
matters for later integration: **a modality is just a name and a set of input
columns.** The future reasoning/LLM embedding arrives as one more entry in
``modalities`` — no model code changes, and the DL encoders it sits beside can
stay frozen (see vpdl.dl.interface).

Also here, because small data makes them necessary rather than optional:

* **availability masks** — a variant with no structure gets a zero structure
  embedding and a mask bit, not a median-imputed fake structure;
* **modality dropout** — whole modalities dropped during training so the model
  cannot lean on one and collapse when it is missing (at least one is kept);
* **MC-dropout uncertainty** — :meth:`FusionClassifier.predict_with_uncertainty`
  returns the mean and standard deviation over stochastic passes. Seed
  ensembles (three seeds is the protocol floor) give a second, complementary
  estimate in vpdl.dl.runner.

``gated`` (softmax modality weights + GLU + residual, MVmamba's GateWave idea)
is implemented, but plain ``concat`` is the default and gating must earn its
place in the ablation: its published gain was ~0.001 AUC.
"""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import numpy as np

from vpdl.models.base import derive_seed

logger = logging.getLogger(__name__)

__all__ = ["FUSION_MODES", "FusionClassifier", "build_fusion_net"]

FUSION_MODES = ("concat", "gated")


def build_fusion_net(dims: Mapping[str, int], shared_dim: int = 64, hidden: int = 128,
                     dropout: float = 0.2, mode: str = "concat",
                     modality_dropout: float = 0.1):
    import torch
    from torch import nn

    if mode not in FUSION_MODES:
        raise ValueError(f"unknown fusion mode {mode!r}; known {FUSION_MODES}")
    names = list(dims)

    class FusionNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.names = names
            self.encoders = nn.ModuleDict({
                name: nn.Sequential(nn.Linear(dim, shared_dim), nn.GELU(), nn.Dropout(dropout))
                for name, dim in dims.items()})
            width = shared_dim * len(names)
            if mode == "concat":
                self.mixer = nn.Sequential(nn.Linear(width, hidden), nn.GELU(),
                                           nn.Dropout(dropout))
            else:
                self.gate = nn.Linear(width, len(names))
                self.value = nn.Linear(shared_dim, hidden)
                self.glu_gate = nn.Linear(shared_dim, hidden)
                self.skip = nn.Linear(shared_dim, hidden)
            self.norm = nn.LayerNorm(hidden)
            self.head = nn.Linear(hidden, 1)

        def represent(self, parts: Mapping[str, "torch.Tensor"], mask=None):
            batch = next(iter(parts.values())).shape[0]
            device = next(iter(parts.values())).device
            if mask is None:
                mask = torch.ones((batch, len(names)), device=device)
            mask = mask.float()
            if self.training and modality_dropout > 0 and len(names) > 1:
                drop = (torch.rand_like(mask) < modality_dropout).float()
                kept = mask * (1 - drop)
                # Never drop every available modality of a row.
                empty = kept.sum(1, keepdim=True) == 0
                mask = torch.where(empty, mask, kept)
            z = [self.encoders[name](parts[name].float()) * mask[:, i:i + 1]
                 for i, name in enumerate(names)]
            if mode == "concat":
                h = self.mixer(torch.cat(z, dim=-1))
            else:
                weights = torch.softmax(self.gate(torch.cat(z, dim=-1))
                                        .masked_fill(mask == 0, -1e4), dim=-1)
                fused = sum(weights[:, i:i + 1] * z[i] for i in range(len(names)))
                h = self.value(fused) * torch.sigmoid(self.glu_gate(fused)) + self.skip(fused)
            return self.norm(h)

        def forward(self, parts, mask=None):
            return self.head(self.represent(parts, mask)).squeeze(-1)

    return FusionNet()


class FusionClassifier:
    """Registry-facing fusion model (``vpdl train --model fusion``).

    `modalities` maps a modality name to column indices of X; `masks` maps a
    modality to the index of its 0/1 availability column (optional). Without
    `modalities` every column is one modality — a residual-free MLP baseline.
    """

    def __init__(self, n_features: int, seed: int = 42, split_index: int | str = 0,
                 modalities: Mapping[str, Sequence[int]] | None = None,
                 masks: Mapping[str, int] | None = None, shared_dim: int = 64,
                 hidden: int = 128, dropout: float = 0.2, mode: str = "concat",
                 modality_dropout: float = 0.1, lr: float = 1e-3, weight_decay: float = 1e-4,
                 epochs: int = 100, patience: int = 10, batch_size: int = 64,
                 mc_samples: int = 30, precision: str = "auto") -> None:
        self.seed = derive_seed(seed, split_index)
        from vpdl.dl.trainer import seed_everything
        seed_everything(self.seed)                      # before construction (L7)

        import torch
        self._torch = torch
        self.modalities = {name: list(map(int, columns)) for name, columns in
                           (modalities or {"all": list(range(n_features))}).items()}
        self.masks = dict(masks or {})
        used = sorted({c for columns in self.modalities.values() for c in columns})
        if used and max(used) >= n_features:
            raise ValueError(f"modality columns exceed the {n_features} features given")
        empty = [name for name, columns in self.modalities.items() if not columns]
        if empty:
            raise ValueError(f"modalities with no columns: {empty}")
        self.mode = mode
        self.mc_samples = int(mc_samples)
        self.hyper = dict(lr=lr, weight_decay=weight_decay, epochs=epochs, patience=patience,
                          batch_size=batch_size, precision=precision)
        self.net = build_fusion_net({n: len(c) for n, c in self.modalities.items()},
                                    shared_dim, hidden, dropout, mode, modality_dropout)
        first = next(p for p in self.net.parameters() if p.dim() >= 2)
        self._initial = first.detach().cpu().numpy().copy()

    def initial_weights(self) -> np.ndarray:
        return self._initial

    def _split(self, X, device):
        torch = self._torch
        X = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
        parts = {name: X[:, columns] for name, columns in self.modalities.items()}
        mask = None
        if self.masks:
            mask = torch.ones((X.shape[0], len(self.modalities)), device=device)
            for i, name in enumerate(self.modalities):
                if name in self.masks:
                    mask[:, i] = (X[:, self.masks[name]] > 0.5).float()
        return parts, mask

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        torch = self._torch
        from vpdl.dl.trainer import TrainConfig, Trainer
        from vpdl.evaluate import roc_auc

        h = self.hyper
        trainer = Trainer(self.net, TrainConfig(
            epochs=h["epochs"], patience=h["patience"], lr=h["lr"],
            weight_decay=h["weight_decay"], batch_size=h["batch_size"],
            precision=h["precision"], schedule="constant"),
            seed=self.seed, run_name=f"fusion-{self.mode}")
        device = trainer.device
        X_all = np.asarray(X, dtype=np.float32)
        target = torch.tensor(np.asarray(y, dtype=np.float32), device=device)
        positive = float(target.sum())
        weight = torch.tensor([(len(target) - positive) / positive if positive else 1.0],
                              device=device)

        def batch_fn(index):
            parts, mask = self._split(X_all[index], device)
            return parts, mask, target[torch.as_tensor(index, device=device)]

        def loss_fn(model, batch):
            logits = model(batch[0], batch[1])
            return torch.nn.functional.binary_cross_entropy_with_logits(
                logits.float(), batch[2], pos_weight=weight)

        score_fn = None
        if X_val is not None and y_val is not None and len(y_val):
            def score_fn(model):
                return roc_auc(np.asarray(y_val), self._predict(X_val, model))
        self.fit_result = trainer.fit(len(X_all), batch_fn, loss_fn, score_fn)
        return self

    def _predict(self, X, model=None, stochastic: bool = False) -> np.ndarray:
        torch = self._torch
        model = model or self.net
        device = next(self.net.parameters()).device
        model.train(stochastic)
        with torch.inference_mode():
            parts, mask = self._split(X, device)
            return torch.sigmoid(model(parts, mask).float()).cpu().numpy()

    def predict_proba(self, X) -> np.ndarray:
        return self._predict(X)

    def predict_with_uncertainty(self, X, samples: int | None = None
                                 ) -> tuple[np.ndarray, np.ndarray]:
        """MC dropout: mean and std of the probability over stochastic passes.

        Modality dropout is part of the stochasticity on purpose: disagreement
        between modality subsets is exactly the uncertainty worth reporting.
        """
        torch = self._torch
        generator_state = torch.get_rng_state()
        torch.manual_seed(derive_seed(self.seed, "mc-dropout"))
        draws = np.stack([self._predict(X, stochastic=True)
                          for _ in range(samples or self.mc_samples)])
        torch.set_rng_state(generator_state)
        self.net.eval()
        return draws.mean(axis=0), draws.std(axis=0)

    def represent(self, X) -> np.ndarray:
        """Fused penultimate representation — the DL embedding for integration."""
        torch = self._torch
        device = next(self.net.parameters()).device
        self.net.eval()
        with torch.inference_mode():
            parts, mask = self._split(X, device)
            return self.net.represent(parts, mask).float().cpu().numpy()
