"""Sequence-window baselines: amino-acid MLP, CNN, BiLSTM+attention, Transformer.

Same inputs as the existing ``bilstm`` arm (:mod:`vpdl.models.bilstm`, which is
left exactly as it is — it has published numbers and is the regression
baseline): a wild-type and a variant residue window from
:func:`vpdl.features.local_windows`, plus the tabular feature matrix. Only the
window encoder differs, so a difference between these arms is a difference of
encoder, not of data.

These are baselines, framed as such. The measured context (RUNLOG 2026-09-21):
the BiLSTM tied the tabular MLP, i.e. the windows added nothing measurable over
six tabular features. Each encoder here tests whether a different inductive
bias finds signal the recurrence did not. None is expected to beat a pretrained
protein language model, which sees ~10^6 x more sequence.

Siamese contrast: the head sees ``[h_wt, h_vt, h_vt - h_wt, |h_vt - h_wt|]``.
If the variant window were ever cut from the wild-type sequence, the last two
blocks would be identically zero (landmine L12); the windows come from the one
function that guarantees they are not.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from vpdl.models.base import derive_seed
from vpdl.models.bilstm import AA_VOCAB, encode_window

logger = logging.getLogger(__name__)

__all__ = ["WINDOW_ENCODERS", "WindowEncoderClassifier"]

WINDOW_ENCODERS = ("aa_mlp", "cnn", "bilstm_attn", "transformer")
_PAD = AA_VOCAB.index("-")


def _build_encoder(kind: str, window: int, embedding_dim: int, hidden: int,
                   dropout: float):
    import torch
    from torch import nn

    class AAMLP(nn.Module):
        """Position-aware: concatenated per-position embeddings, then an MLP."""

        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(len(AA_VOCAB), embedding_dim, padding_idx=_PAD)
            self.net = nn.Sequential(nn.Linear(window * embedding_dim, hidden), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, hidden))

        def forward(self, tokens):
            return self.net(self.embed(tokens).flatten(1))

    class CNN(nn.Module):
        """Two residual conv blocks (k=3), then mean + max pooling."""

        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(len(AA_VOCAB), embedding_dim, padding_idx=_PAD)
            self.inp = nn.Conv1d(embedding_dim, hidden, 3, padding=1)
            self.blocks = nn.ModuleList([nn.Conv1d(hidden, hidden, 3, padding=1)
                                         for _ in range(2)])
            # GroupNorm(1, .) rather than BatchNorm: a last batch of one would
            # make BatchNorm raise in train mode, and its running statistics
            # would couple predictions to batch composition.
            self.norms = nn.ModuleList([nn.GroupNorm(1, hidden) for _ in range(2)])
            self.drop = nn.Dropout(dropout)
            self.out = nn.Linear(2 * hidden, hidden)

        def forward(self, tokens):
            x = self.inp(self.embed(tokens).transpose(1, 2))
            for conv, norm in zip(self.blocks, self.norms):
                x = x + self.drop(torch.nn.functional.gelu(norm(conv(x))))
            return self.out(torch.cat([x.mean(-1), x.amax(-1)], dim=-1))

    class BiLSTMAttention(nn.Module):
        """BiLSTM with additive attention pooling (pad positions masked)."""

        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(len(AA_VOCAB), embedding_dim, padding_idx=_PAD)
            self.lstm = nn.LSTM(embedding_dim, hidden // 2, batch_first=True,
                                bidirectional=True)
            self.score = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(),
                                       nn.Linear(hidden, 1))

        def forward(self, tokens):
            states, _ = self.lstm(self.embed(tokens))
            logits = self.score(states).squeeze(-1)
            logits = logits.masked_fill(tokens == _PAD, -1e4)
            weights = torch.softmax(logits, dim=-1).unsqueeze(-1)
            return (states * weights).sum(dim=1)

    class Transformer(nn.Module):
        """Small pre-norm encoder; centre token + masked mean pooling."""

        def __init__(self) -> None:
            super().__init__()
            heads = 4 if hidden % 4 == 0 else 1
            self.embed = nn.Embedding(len(AA_VOCAB), hidden, padding_idx=_PAD)
            self.position = nn.Parameter(torch.zeros(1, window, hidden))
            nn.init.normal_(self.position, std=0.02)
            layer = nn.TransformerEncoderLayer(hidden, heads, 2 * hidden, dropout,
                                               batch_first=True, norm_first=True,
                                               activation="gelu")
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(hidden)
            self.out = nn.Linear(2 * hidden, hidden)

        def forward(self, tokens):
            pad = tokens == _PAD
            x = self.encoder(self.embed(tokens) + self.position, src_key_padding_mask=pad)
            x = self.norm(x)
            keep = (~pad).unsqueeze(-1).float()
            mean = (x * keep).sum(1) / keep.sum(1).clamp_min(1.0)
            return self.out(torch.cat([x[:, window // 2], mean], dim=-1))

    encoders = {"aa_mlp": AAMLP, "cnn": CNN, "bilstm_attn": BiLSTMAttention,
                "transformer": Transformer}
    if kind not in encoders:
        raise ValueError(f"unknown window encoder {kind!r}; known: {WINDOW_ENCODERS}")
    return encoders[kind]()


class WindowEncoderClassifier:
    """Siamese window encoder + tabular features -> pathogenicity probability.

    Seeds before construction (landmine L7) and trains with
    :class:`vpdl.dl.trainer.Trainer`: early stopping on inner-validation
    ROC-AUC, class-weighted BCE (as the existing neural arms do), bf16 on CUDA.
    """

    def __init__(self, encoder: str = "cnn", n_tabular_features: int = 0, seed: int = 42,
                 split_index: int | str = 0, window: int = 15, embedding_dim: int = 32,
                 hidden: int = 64, dropout: float = 0.3, lr: float = 1e-3,
                 weight_decay: float = 1e-4, epochs: int = 100, patience: int = 10,
                 batch_size: int = 32, precision: str = "auto") -> None:
        self.seed = derive_seed(seed, split_index)
        from vpdl.dl.trainer import seed_everything
        seed_everything(self.seed)                        # <-- before construction

        import torch
        from torch import nn

        self._torch = torch
        self.encoder_kind = encoder
        self.window = window
        self.n_tabular_features = n_tabular_features
        self.hyper = dict(lr=lr, weight_decay=weight_decay, epochs=epochs,
                          patience=patience, batch_size=batch_size, precision=precision)
        encoder_module = _build_encoder(encoder, window, embedding_dim, hidden, dropout)

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = encoder_module
                width = hidden * 4 + n_tabular_features
                self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64),
                                          nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

            def forward(self, wt, vt, tabular=None):
                h_wt, h_vt = self.encoder(wt), self.encoder(vt)
                diff = h_vt - h_wt
                parts = [h_wt, h_vt, diff, diff.abs()]
                if tabular is not None and tabular.shape[1]:
                    parts.append(tabular)
                return self.head(torch.cat(parts, dim=1)).squeeze(-1)

        self.net = _Net()
        first = next(p for p in self.net.parameters() if p.dim() >= 2)
        self._initial = first.detach().cpu().numpy().copy()

    def initial_weights(self) -> np.ndarray:
        return self._initial

    def _tokens(self, windows: Sequence[tuple[str, str]]):
        torch = self._torch
        for wt, vt in windows[:1]:
            if len(wt) != self.window or len(vt) != self.window:
                raise ValueError(f"windows must be {self.window} residues; got {len(wt)}")
        wt = torch.tensor(np.stack([encode_window(w) for w, _ in windows]))
        vt = torch.tensor(np.stack([encode_window(v) for _, v in windows]))
        return wt, vt

    def _tabular(self, tabular, n: int):
        torch = self._torch
        if tabular is None or not self.n_tabular_features:
            return torch.zeros((n, 0))
        return torch.tensor(np.asarray(tabular, dtype=np.float32))

    def fit(self, windows, y, tabular=None, windows_val=None, y_val=None, tabular_val=None):
        torch = self._torch
        from vpdl.dl.trainer import TrainConfig, Trainer
        from vpdl.evaluate import roc_auc

        wt, vt = self._tokens(windows)
        tab = self._tabular(tabular, len(y))
        target = torch.tensor(np.asarray(y, dtype=np.float32))
        positive = float(target.sum())
        pos_weight = (len(target) - positive) / positive if positive else 1.0

        trainer = Trainer(self.net, TrainConfig(
            epochs=self.hyper["epochs"], patience=self.hyper["patience"],
            lr=self.hyper["lr"], weight_decay=self.hyper["weight_decay"],
            batch_size=self.hyper["batch_size"], precision=self.hyper["precision"],
            schedule="constant"), seed=self.seed, run_name=f"seqwin-{self.encoder_kind}")
        device = trainer.device
        weight = torch.tensor([pos_weight], device=device)

        def batch_fn(index):
            index = torch.as_tensor(index, dtype=torch.long)
            return (wt[index].to(device), vt[index].to(device), tab[index].to(device),
                    target[index].to(device))

        def loss_fn(model, batch):
            logits = model(batch[0], batch[1], batch[2])
            return torch.nn.functional.binary_cross_entropy_with_logits(
                logits.float(), batch[3], pos_weight=weight)

        score_fn = None
        if windows_val is not None and y_val is not None and len(y_val):
            def score_fn(model):
                return roc_auc(np.asarray(y_val), self._predict(windows_val, tabular_val, model))
        trainer.fit(len(target), batch_fn, loss_fn, score_fn)
        return self

    def _predict(self, windows, tabular, model=None) -> np.ndarray:
        torch = self._torch
        model = model or self.net
        device = next(self.net.parameters()).device
        wt, vt = self._tokens(windows)
        tab = self._tabular(tabular, len(windows))
        model.eval()
        out = []
        with torch.inference_mode():
            for start in range(0, len(wt), 1024):
                sl = slice(start, start + 1024)
                logits = model(wt[sl].to(device), vt[sl].to(device), tab[sl].to(device))
                out.append(torch.sigmoid(logits.float()).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def predict_proba(self, windows, tabular=None) -> np.ndarray:
        return self._predict(windows, tabular)
