"""Residual MLP head over the feature matrix.

Carried forward conceptually from v1 because every existing number on this
project used it; keeping the arm means the new results can be read against the
old ones rather than floating free.

Satisfies regression landmine L7 (every split is seeded before construction).
"""

from __future__ import annotations

import logging
import random
from typing import Sequence

import numpy as np

from vpdl.models.base import derive_seed

logger = logging.getLogger(__name__)

__all__ = ["MLPClassifier"]


def _seed_everything(seed: int) -> None:
    """Seed every RNG that can touch initialisation, BEFORE any module is built."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class MLPClassifier:
    """Two-hidden-layer residual MLP with a sigmoid head.

    The constructor seeds and *then* builds. That ordering is the entire point
    of this class's contract: v1 built first and seeded later, so the first
    leave-one-gene-out split of every process initialised from whatever entropy
    the process happened to start with. It went unnoticed for months because
    only one gene was affected and the numbers stayed plausible.
    """

    def __init__(
        self,
        n_features: int,
        seed: int = 42,
        split_index: int = 0,
        hidden: Sequence[int] = (256, 128),
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        patience: int = 10,
        batch_size: int = 64,
    ) -> None:
        self.seed = derive_seed(seed, split_index)
        _seed_everything(self.seed)          # <-- before construction, always

        import torch
        from torch import nn

        self._torch = torch
        self.n_features = n_features
        self.epochs, self.patience, self.batch_size = epochs, patience, batch_size
        self.lr, self.weight_decay = lr, weight_decay

        layers: list = []
        width = n_features
        for size in hidden:
            layers += [nn.Linear(width, size), nn.LayerNorm(size), nn.ReLU(),
                       nn.Dropout(dropout)]
            width = size
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

        first_linear = next(m for m in self.net if isinstance(m, nn.Linear))
        self._initial = first_linear.weight.detach().cpu().numpy().copy()

    def initial_weights(self) -> np.ndarray:
        """First-layer weights as constructed, for the determinism regression test."""
        return self._initial

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        torch = self._torch
        from torch import nn

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(device)

        X_t = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
        y_t = torch.tensor(np.asarray(y, dtype=np.float32), device=device).view(-1, 1)

        positive = float(y_t.sum().item())
        negative = float(len(y_t) - positive)
        pos_weight = torch.tensor(
            [negative / positive if positive else 1.0], device=device
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimiser = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        has_validation = X_val is not None and y_val is not None and len(y_val) > 0
        if has_validation:
            Xv = torch.tensor(np.asarray(X_val, dtype=np.float32), device=device)
            yv = np.asarray(y_val)

        best_score, best_state, stalled = -np.inf, None, 0
        generator = torch.Generator(device="cpu").manual_seed(self.seed)

        for epoch in range(self.epochs):
            self.net.train()
            order = torch.randperm(len(X_t), generator=generator).to(device)
            for start in range(0, len(order), self.batch_size):
                batch = order[start: start + self.batch_size]
                optimiser.zero_grad()
                loss = criterion(self.net(X_t[batch]), y_t[batch])
                loss.backward()
                optimiser.step()

            if not has_validation:
                continue

            from vpdl.evaluate import roc_auc
            self.net.eval()
            with torch.no_grad():
                scores = torch.sigmoid(self.net(Xv)).cpu().numpy().ravel()
            score = roc_auc(yv, scores)
            if np.isfinite(score) and score > best_score:
                best_score, stalled = score, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.net.state_dict().items()}
            else:
                stalled += 1
                if stalled >= self.patience:
                    logger.info("Early stop at epoch %d (best val AUC %.4f)",
                                epoch, best_score)
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict_proba(self, X) -> np.ndarray:
        torch = self._torch
        device = next(self.net.parameters()).device
        self.net.eval()
        with torch.no_grad():
            X_t = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
            return torch.sigmoid(self.net(X_t)).cpu().numpy().ravel()
