"""BiLSTM over local sequence context — included as a baseline to beat.

Read this before reporting its numbers.

This arm exists to answer "why not a recurrent model?" with a measurement
instead of an assertion. It is **not** expected to win, and the reasons are
structural rather than a matter of tuning:

* ESM-2 is already the sequence model in this pipeline — a 650M-parameter
  transformer pretrained on UniRef50. A BiLSTM trained from scratch on a
  four-gene panel (3,912 residues, 683 clinically labelled variants) sees
  roughly six orders of magnitude less sequence than the representation it is
  being compared against.
* The field moved from recurrent protein models (UniRep, early TAPE) to
  transformers on this exact class of task, and did so on evidence.
* The rest of the feature space is per-variant tabular data with no sequential
  structure for a recurrence to exploit.

So it is wired to the one place recurrence is even meaningful — the windowed
residue neighbourhood around the mutation — and reported honestly. A surprising
win here would be interesting and would need checking for leakage before it was
believed.

**Measured 2026-09-21: it tied the MLP** (ClinVar-only mean ROC-AUC 0.962 vs
0.963) and beat an untuned GBM. It sees the same six tabular features as the
MLP plus the residue windows, so the tie means the windows added nothing
measurable. No leakage path exists here: under leave-one-gene-out the held-out
gene's windows come from a protein the model never saw, and windows carry no
label information.
"""

from __future__ import annotations

import logging
import random
from typing import Sequence

import numpy as np

from vpdl.models.base import derive_seed

logger = logging.getLogger(__name__)

__all__ = ["BiLSTMClassifier", "AA_VOCAB", "encode_window"]

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY-"
_INDEX = {residue: index for index, residue in enumerate(AA_VOCAB)}


def encode_window(window: str) -> np.ndarray:
    """Residue window to integer indices; unknown residues map to the pad symbol."""
    return np.array([_INDEX.get(residue, _INDEX["-"]) for residue in window],
                    dtype=np.int64)


class BiLSTMClassifier:
    """Bidirectional LSTM over (wild-type, variant) residue windows.

    Consumes the SAME windows as the rest of the pipeline via
    :func:`vpdl.features.local_windows`, which guarantees the variant window is
    cut from the mutated sequence. If that guarantee breaks, this model's two
    inputs become identical and the contrast term goes to zero silently — the
    exact failure v1 shipped for MSH6 (regression landmine L12).
    """

    def __init__(
        self,
        n_tabular_features: int = 0,
        seed: int = 42,
        split_index: int = 0,
        embedding_dim: int = 32,
        hidden: int = 64,
        layers: int = 1,
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 100,
        patience: int = 10,
        batch_size: int = 32,
    ) -> None:
        self.seed = derive_seed(seed, split_index)
        random.seed(self.seed)
        np.random.seed(self.seed % (2**32))

        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self._torch = torch
        self.epochs, self.patience, self.batch_size = epochs, patience, batch_size
        self.lr = lr
        self.n_tabular_features = n_tabular_features

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Embedding(len(AA_VOCAB), embedding_dim,
                                          padding_idx=_INDEX["-"])
                self.lstm = nn.LSTM(
                    embedding_dim, hidden, num_layers=layers,
                    batch_first=True, bidirectional=True,
                    dropout=dropout if layers > 1 else 0.0,
                )
                # [h_wt, h_vt, h_vt - h_wt] over a bidirectional encoder.
                width = hidden * 2 * 3 + n_tabular_features
                self.head = nn.Sequential(
                    nn.LayerNorm(width), nn.Linear(width, 64), nn.ReLU(),
                    nn.Dropout(dropout), nn.Linear(64, 1),
                )

            def encode(self, window):
                output, _ = self.lstm(self.embed(window))
                return output.mean(dim=1)

            def forward(self, wt, vt, tabular=None):
                h_wt, h_vt = self.encode(wt), self.encode(vt)
                parts = [h_wt, h_vt, h_vt - h_wt]
                if tabular is not None and tabular.shape[1] > 0:
                    parts.append(tabular)
                return self.head(torch.cat(parts, dim=1))

        self.net = _Net()
        self._initial = self.net.embed.weight.detach().cpu().numpy().copy()

    def initial_weights(self) -> np.ndarray:
        return self._initial

    def _tensors(self, windows, tabular, device):
        torch = self._torch
        wt = torch.tensor(np.stack([encode_window(w) for w, _ in windows]),
                          device=device)
        vt = torch.tensor(np.stack([encode_window(v) for _, v in windows]),
                          device=device)
        tab = (torch.tensor(np.asarray(tabular, dtype=np.float32), device=device)
               if tabular is not None and self.n_tabular_features
               else None)
        return wt, vt, tab

    def fit(self, windows: Sequence[tuple[str, str]], y, tabular=None,
            windows_val=None, y_val=None, tabular_val=None):
        torch = self._torch
        from torch import nn
        from vpdl.evaluate import roc_auc

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(device)

        wt, vt, tab = self._tensors(windows, tabular, device)
        y_t = torch.tensor(np.asarray(y, dtype=np.float32), device=device).view(-1, 1)

        positive = float(y_t.sum().item())
        negative = float(len(y_t) - positive)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([negative / positive if positive else 1.0],
                                    device=device)
        )
        optimiser = torch.optim.AdamW(self.net.parameters(), lr=self.lr)

        has_validation = windows_val is not None and y_val is not None
        if has_validation:
            wt_v, vt_v, tab_v = self._tensors(windows_val, tabular_val, device)

        best_score, best_state, stalled = -np.inf, None, 0
        generator = torch.Generator(device="cpu").manual_seed(self.seed)

        for epoch in range(self.epochs):
            self.net.train()
            order = torch.randperm(len(y_t), generator=generator).to(device)
            for start in range(0, len(order), self.batch_size):
                batch = order[start: start + self.batch_size]
                optimiser.zero_grad()
                logits = self.net(wt[batch], vt[batch],
                                  tab[batch] if tab is not None else None)
                criterion(logits, y_t[batch]).backward()
                optimiser.step()

            if not has_validation:
                continue

            self.net.eval()
            with torch.no_grad():
                scores = torch.sigmoid(
                    self.net(wt_v, vt_v, tab_v)
                ).cpu().numpy().ravel()
            score = roc_auc(np.asarray(y_val), scores)
            if np.isfinite(score) and score > best_score:
                best_score, stalled = score, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.net.state_dict().items()}
            else:
                stalled += 1
                if stalled >= self.patience:
                    logger.info("BiLSTM early stop at epoch %d (val AUC %.4f)",
                                epoch, best_score)
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict_proba(self, windows, tabular=None) -> np.ndarray:
        torch = self._torch
        device = next(self.net.parameters()).device
        wt, vt, tab = self._tensors(windows, tabular, device)
        self.net.eval()
        with torch.no_grad():
            return torch.sigmoid(self.net(wt, vt, tab)).cpu().numpy().ravel()
