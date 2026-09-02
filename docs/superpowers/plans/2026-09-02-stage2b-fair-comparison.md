# Stage-2b Fair Comparison + Grid Driver — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the Stage-2b ESM-2 fine-tune the same prior features the frozen probe already sees, fix four training defects that plausibly cause its epoch-1 early stopping, and ship a resumable grid driver so a 45-hour weekend GPU run returns per-variant predictions instead of four rows of metrics.

**Architecture:** `ESMFineTuneClassifier` splits into `encode()` (backbone → ESM feature block + PLLR) and `classify()` (features → logit). That split makes three things fall out cheaply: an optional prior-feature branch reusing the existing `src/fusion.py` heads, PLLR as a zero-initialised residual base so the untrained model *is* the 0.834-AUC zero-shot predictor, and an in-memory precompute path for `n_unfrozen_layers=0` where the backbone output is constant across epochs. A new `src/finetune_grid.py` defines cells and deterministic slugs; `scripts/run_stage2b_grid.py` runs them sequentially and resumably.

**Tech Stack:** Python 3.14, PyTorch 2.13, transformers 5.15, pandas, scikit-learn, pytest.

**Spec:** `docs/superpowers/specs/2026-09-02-accuracy-ablation-paper-design.md` (sub-projects 1 and 2 only; sub-projects 3–5 become separate plans)

## Global Constraints

- **Never break existing results.** 90 tests currently pass (`python -m pytest tests/ -q`). Every task ends with the full suite green, not just the new test.
- **Checkpoint compatibility.** `ConcatFusionHead`'s norm attribute stays named `bn` even when it holds a `LayerNorm`, so existing Stage-2 checkpoints keep loading. Legacy `use_pllr: bool` in a checkpoint config maps to `pllr_mode="concat"` (the old behaviour), not `"residual"`.
- **Backwards-compatible CLI.** `--use_pllr` / `--no-use_pllr` keep working exactly as today. The new `--pllr_mode {residual,concat}` selects *how* PLLR enters; `--no-use_pllr` still means off.
- **Gene-constant priors are dropped under leave-one-gene-out.** Always call `prior_columns_of(df, drop_gene_constant=True)` for `--eval lopo`. `RUNLOG.md` 2026-08-28 records MLH1 collapsing to ROC-AUC 0.500 in every seed when they are kept.
- **AF quarantine (spec §3.1).** AF-derived labels and AF-derived features must never both be active. Enforced in code with a test, not by convention.
- **Prior preprocessing fitted on the fine-tune partition only**, applied unchanged to inner-val and holdout, persisted into the checkpoint.
- **Tests must be network-free.** `facebook/esm2_t6_8M_UR50D` is in the local HF cache; tests that need a backbone skip cleanly if it is not resolvable offline.
- **Test file convention:** plain pytest functions, `sys.path.insert(0, PROJECT_ROOT)` at the top, matching `tests/test_mmr_modules.py`.
- Run tests with `.venv/bin/python -m pytest`.

---

### Task 1: LayerNorm option for `ConcatFusionHead`

The GPU config that fits a 650M full fine-tune on a 15 GiB card is `--batch_size 1 --grad_accum 8`. `nn.BatchNorm1d` raises `ValueError: Expected more than 1 value per channel when training` at batch size 1, so the fusion head cannot be used in the fine-tune path as written.

**Files:**
- Modify: `src/fusion.py:58-83` (`ConcatFusionHead`), `src/fusion.py:162-170` (`build_fusion_head`)
- Test: `tests/test_mmr_modules.py` (append near `test_branch_and_concat_shapes`, line 317)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `ConcatFusionHead(dims, shared_dim=128, dropout=0.2, norm="batch"|"layer")` and `build_fusion_head(name, dims, shared_dim=128, dropout=0.2, norm="batch")`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mmr_modules.py`:

```python
def test_concat_fusion_layer_norm_works_at_batch_size_one():
    """Micro-batch 1 is the only config that fits a 650M full fine-tune on a
    15 GiB card, and BatchNorm1d cannot compute batch statistics there."""
    batch_head = ConcatFusionHead((8, 4), shared_dim=6, norm="batch")
    batch_head.train()
    try:
        batch_head(torch.randn(1, 8), torch.randn(1, 4))
        raise AssertionError("BatchNorm1d should reject batch size 1 in train mode")
    except ValueError:
        pass

    layer_head = ConcatFusionHead((8, 4), shared_dim=6, norm="layer")
    layer_head.train()
    out = layer_head(torch.randn(1, 8), torch.randn(1, 4))
    assert out.shape == (1,)
    assert torch.isfinite(out).all()


def test_build_fusion_head_passes_norm_through():
    head = build_fusion_head("concat", (8, 4), shared_dim=6, norm="layer")
    assert isinstance(head.bn, torch.nn.LayerNorm)


def test_concat_fusion_rejects_unknown_norm():
    try:
        ConcatFusionHead((8, 4), shared_dim=6, norm="group")
        raise AssertionError("expected ValueError for unknown norm")
    except ValueError:
        pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_mmr_modules.py -k "fusion_layer_norm or norm_through or unknown_norm" -v`
Expected: FAIL — `ConcatFusionHead.__init__() got an unexpected keyword argument 'norm'`

- [ ] **Step 3: Write minimal implementation**

In `src/fusion.py`, replace `ConcatFusionHead.__init__` (currently lines 65-74):

```python
    def __init__(self, dims: tuple[int, ...], shared_dim: int = 128,
                 dropout: float = 0.2, norm: str = "batch") -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [BranchProjection(d, shared_dim) for d in dims])
        width = shared_dim * len(dims)
        # The attribute stays named `bn` whichever norm it holds: renaming it
        # would change every existing Stage-2 checkpoint's state-dict keys.
        # "layer" exists for the Stage-2b fine-tune, which runs at micro-batch
        # 1 on a 15 GiB card -- BatchNorm1d cannot compute batch statistics
        # over a single sample and raises in train mode.
        if norm == "batch":
            self.bn: nn.Module = nn.BatchNorm1d(width)
        elif norm == "layer":
            self.bn = nn.LayerNorm(width)
        else:
            raise ValueError(f"norm must be 'batch' or 'layer'; got {norm!r}")
        self.norm_kind = norm
        self.fc1 = nn.Linear(width, shared_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(shared_dim, 1)
```

And replace `build_fusion_head` (currently lines 162-170):

```python
def build_fusion_head(name: str, dims: tuple[int, ...], shared_dim: int = 128,
                      dropout: float = 0.2, norm: str = "batch") -> nn.Module:
    """Factory used by the transfer/fusion benchmark CLI.

    ``norm`` applies to the concat head only -- GateWave already normalises
    with :class:`~torch.nn.LayerNorm`, so it is safe at micro-batch 1 as-is.
    """
    name = name.lower().strip()
    if name == "concat":
        return ConcatFusionHead(dims, shared_dim, dropout, norm=norm)
    if name == "gatewave":
        return GateWaveFusionHead(dims, shared_dim, dropout)
    raise ValueError(f"Unknown fusion head '{name}' (expected concat|gatewave).")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_mmr_modules.py -q`
Expected: PASS, 93 tests

- [ ] **Step 5: Commit**

```bash
git add src/fusion.py tests/test_mmr_modules.py
git commit -m "feat(fusion): LayerNorm option for ConcatFusionHead

BatchNorm1d cannot compute batch statistics at micro-batch 1, which is the
only config that fits a 650M siamese full fine-tune on a 15 GiB card. The
norm attribute keeps its 'bn' name so existing Stage-2 checkpoints still load.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CP3LLMEoVNhJx8zUvHBCkm"
```

---

### Task 2: Split `ESMFineTuneClassifier` into `encode()` + `classify()`

A pure refactor with no behaviour change. It is the foundation for Tasks 3, 4 and 7: once the backbone pass is separable from the head, the prior branch, the PLLR residual, and the frozen-backbone precompute all become local changes.

**Files:**
- Modify: `src/esm_finetune.py:203-255` (`_encode`, `forward`)
- Create: `tests/test_esm_finetune.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `ESMFineTuneClassifier.encode(wt_ids, wt_mask, wt_pos, mut_ids=None, mut_mask=None, mut_pos=None, wt_tok_id=None, mut_tok_id=None) -> Tuple[Tensor, Tensor]` returning `(esm_block [B, E], pllr [B])`. `E` is `hidden_size*4` for siamese, `hidden_size` for `wt_site`. `pllr` is all-zeros when `pllr_mode == "off"`.
  - `ESMFineTuneClassifier.classify(esm_block, pllr, priors=None) -> Tensor` returning logits `[B]`.
  - `ESMFineTuneClassifier.esm_block_dim -> int` property.
  - `forward(...)` unchanged in signature and output.

- [ ] **Step 1: Write the failing test**

Create `tests/test_esm_finetune.py`:

```python
"""Offline unit tests for src/esm_finetune.py.

Model-backed tests use facebook/esm2_t6_8M_UR50D from the local HuggingFace
cache and skip cleanly when it is not resolvable offline. They are smoke
tests for correctness -- never results to compare against 650M runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.esm_finetune import ESMFineTuneClassifier, build_examples, make_collate_fn  # noqa: E402

TINY = "facebook/esm2_t6_8M_UR50D"
SEQ = "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPF"


def tiny_model_or_skip(**kwargs) -> ESMFineTuneClassifier:
    pytest.importorskip("transformers")
    try:
        return ESMFineTuneClassifier(model_name=TINY, **kwargs)
    except Exception as exc:  # network-free guard
        pytest.skip(f"ESM checkpoint {TINY} unavailable offline: {exc}")


def demo_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "gene": ["G", "G", "G", "G"],
        "position": [3, 10, 21, 30],
        "wt_aa": [SEQ[2], SEQ[9], SEQ[20], SEQ[29]],
        "mut_aa": ["A", "D", "W", "K"],
        "label": [1.0, 0.0, 1.0, 0.0],
    })


def demo_batch(model: ESMFineTuneClassifier) -> dict:
    ex = build_examples(demo_frame(), {"G": SEQ}, model.mode,
                        max_residues=64, tokenizer=model.tokenizer)
    assert len(ex) == 4, "fixture rows must survive wt validation"
    return make_collate_fn(model.tokenizer, model.mode)(ex)


def test_encode_returns_block_and_pllr_with_expected_shapes():
    model = tiny_model_or_skip(mode="siamese", n_unfrozen_layers=0)
    model.eval()
    batch = demo_batch(model)
    with torch.no_grad():
        block, pllr = model.encode(**{k: v for k, v in batch.items()
                                      if k not in ("labels", "weights")})
    assert block.shape == (4, model.esm_block_dim)
    assert model.esm_block_dim == model.hidden_size * 4
    assert pllr.shape == (4,)
    assert torch.isfinite(block).all() and torch.isfinite(pllr).all()


def test_forward_equals_classify_of_encode():
    """The refactor must not change what forward() computes."""
    model = tiny_model_or_skip(mode="siamese", n_unfrozen_layers=0)
    model.eval()
    batch = demo_batch(model)
    kwargs = {k: v for k, v in batch.items() if k not in ("labels", "weights")}
    with torch.no_grad():
        direct = model(**kwargs)
        block, pllr = model.encode(**kwargs)
        staged = model.classify(block, pllr)
    torch.testing.assert_close(direct, staged)


def test_encode_is_deterministic_in_eval_mode():
    """Task 7's precompute path depends on this."""
    model = tiny_model_or_skip(mode="siamese", n_unfrozen_layers=0)
    model.eval()
    batch = demo_batch(model)
    kwargs = {k: v for k, v in batch.items() if k not in ("labels", "weights")}
    with torch.no_grad():
        a_block, a_pllr = model.encode(**kwargs)
        b_block, b_pllr = model.encode(**kwargs)
    torch.testing.assert_close(a_block, b_block)
    torch.testing.assert_close(a_pllr, b_pllr)


def test_wt_site_mode_block_dim_is_hidden_size():
    model = tiny_model_or_skip(mode="wt_site", n_unfrozen_layers=0)
    assert model.esm_block_dim == model.hidden_size
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_esm_finetune.py -v`
Expected: FAIL — `AttributeError: 'ESMFineTuneClassifier' object has no attribute 'encode'`

- [ ] **Step 3: Write minimal implementation**

In `src/esm_finetune.py`, add an `esm_block_dim` property just after `head_params()` (line 189):

```python
    @property
    def esm_block_dim(self) -> int:
        """Width of the ESM feature block that :meth:`encode` returns."""
        return self.hidden_size * (4 if self.mode == "siamese" else 1)
```

Then replace `forward` (lines 206-255) with `encode`, `classify` and a thin `forward`:

```python
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
        return self.out(self.head(feat)).squeeze(-1)

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
```

This refactor references `self.pllr_mode`, which does not exist yet. Add the translation in `__init__`, replacing the `self.use_pllr = bool(use_pllr)` line (line 124):

```python
        self.pllr_mode = "concat" if use_pllr else "off"
        #: Retained for backwards compatibility with existing checkpoints and
        #: callers; Task 3 replaces the constructor argument itself.
        self.use_pllr = bool(use_pllr)
```

And in `__init__`, replace the `emb_dim` / `in_dim` lines (161, 169) so they use the new property:

```python
        emb_dim = self.esm_block_dim
```
```python
        in_dim = emb_dim + (1 if self.pllr_mode == "concat" else 0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_esm_finetune.py -v && .venv/bin/python -m pytest tests/ -q`
Expected: new file PASSes (4 tests), full suite PASSes (97 tests)

- [ ] **Step 5: Commit**

```bash
git add src/esm_finetune.py tests/test_esm_finetune.py
git commit -m "refactor(esm_finetune): split forward into encode() + classify()

Pure refactor, asserted output-identical. Separating the backbone pass from
the head is what makes the prior branch, the PLLR residual and the
frozen-backbone precompute local changes rather than surgery on forward().

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CP3LLMEoVNhJx8zUvHBCkm"
```

---

### Task 3: PLLR as a zero-initialised residual base

Today PLLR enters as 1 of 5,121 input dims with a random init weight, so an untrained model discards a signal that reaches ROC-AUC 0.834 on this panel unaided. As a residual base with a zero-initialised output layer, the untrained model *is* the zero-shot predictor and training learns a correction on top of it.

**Files:**
- Modify: `src/esm_finetune.py` (`__init__`, `config`, `classify`, `load_finetuned_model`)
- Test: `tests/test_esm_finetune.py`

**Interfaces:**
- Consumes: `encode`/`classify`/`esm_block_dim` from Task 2.
- Produces:
  - Module constant `PLLR_MODES = ("residual", "concat", "off")`.
  - `ESMFineTuneClassifier(..., pllr_mode: str = "residual")` replacing `use_pllr`. `use_pllr: Optional[bool] = None` stays accepted: `True` → `"concat"`, `False` → `"off"`, and passing both raises `ValueError`.
  - `self.pllr_gain`: `nn.Parameter` scalar, initialised `-1.0`, present only in `"residual"` mode.
  - `config()` emits `pllr_mode` and no longer emits `use_pllr`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_esm_finetune.py`:

```python
def test_residual_mode_untrained_model_is_the_zero_shot_predictor():
    """With out zero-initialised, logit == pllr_gain * pllr / pllr_scale, so
    the untrained model reproduces the zero-shot ESM ranking exactly."""
    model = tiny_model_or_skip(mode="siamese", n_unfrozen_layers=0,
                               pllr_mode="residual")
    model.eval()
    batch = demo_batch(model)
    kwargs = {k: v for k, v in batch.items() if k not in ("labels", "weights")}
    with torch.no_grad():
        block, pllr = model.encode(**kwargs)
        logits = model.classify(block, pllr)
        expected = model.pllr_gain * (pllr / model.pllr_scale)
    torch.testing.assert_close(logits, expected)


def test_residual_gain_is_negative_so_damaging_variants_score_high():
    """Raw PLLR is negative for damaging variants and pathogenic is the
    positive class, so the gain must start negative."""
    model = tiny_model_or_skip(mode="siamese", n_unfrozen_layers=0,
                               pllr_mode="residual")
    assert float(model.pllr_gain) == pytest.approx(-1.0)
    damaging = torch.tensor([-8.0])
    tolerated = torch.tensor([0.5])
    block = torch.zeros(1, model.esm_block_dim)
    with torch.no_grad():
        assert float(model.classify(block, damaging)) > float(
            model.classify(block, tolerated))


def test_residual_output_layer_trains_off_zero():
    """Zero-init must not freeze the head: out.weight has a non-zero gradient
    on the first backward even though it starts at zero."""
    model = tiny_model_or_skip(mode="siamese", n_unfrozen_layers=0,
                               pllr_mode="residual")
    model.train()
    block = torch.randn(4, model.esm_block_dim)
    pllr = torch.randn(4)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model.classify(block, pllr), torch.tensor([1.0, 0.0, 1.0, 0.0]))
    loss.backward()
    assert model.out.weight.grad is not None
    assert float(model.out.weight.grad.abs().sum()) > 0.0


def test_pllr_mode_off_ignores_the_term_entirely():
    model = tiny_model_or_skip(mode="siamese", n_unfrozen_layers=0,
                               pllr_mode="off")
    model.eval()
    block = torch.randn(2, model.esm_block_dim)
    with torch.no_grad():
        a = model.classify(block, torch.tensor([-9.0, -9.0]))
        b = model.classify(block, torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(a, b)


def test_legacy_use_pllr_true_maps_to_concat():
    model = tiny_model_or_skip(mode="siamese", n_unfrozen_layers=0, use_pllr=True)
    assert model.pllr_mode == "concat"
    assert model.config()["pllr_mode"] == "concat"


def test_legacy_use_pllr_false_maps_to_off():
    model = tiny_model_or_skip(mode="siamese", n_unfrozen_layers=0, use_pllr=False)
    assert model.pllr_mode == "off"


def test_passing_both_pllr_mode_and_use_pllr_raises():
    pytest.importorskip("transformers")
    with pytest.raises(ValueError, match="both"):
        ESMFineTuneClassifier(model_name=TINY, mode="siamese",
                              n_unfrozen_layers=0, pllr_mode="residual",
                              use_pllr=True)


def test_unknown_pllr_mode_raises():
    pytest.importorskip("transformers")
    with pytest.raises(ValueError, match="pllr_mode"):
        ESMFineTuneClassifier(model_name=TINY, mode="siamese",
                              n_unfrozen_layers=0, pllr_mode="sometimes")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_esm_finetune.py -k pllr -v`
Expected: FAIL — `__init__() got an unexpected keyword argument 'pllr_mode'`

- [ ] **Step 3: Write minimal implementation**

In `src/esm_finetune.py`, add beside `FINETUNE_MODES` (line 55):

```python
#: How the zero-shot term log P(mut|X) - log P(wt|X) reaches the classifier.
#: "residual" adds it as a learnable-gain skip connection past a
#: zero-initialised output layer, so an untrained model reproduces the
#: zero-shot ESM ranking (ROC-AUC ~0.834 on the MMR panel) and training learns
#: a correction on top of a working predictor. "concat" is the historical
#: behaviour: one more input dimension with a random init weight. "off" is the
#: ablation.
PLLR_MODES = ("residual", "concat", "off")
```

Replace the `use_pllr` constructor parameter (line 111) with:

```python
                 pllr_mode: str = "residual",
                 use_pllr: Optional[bool] = None) -> None:
```

Replace the `self.pllr_mode` / `self.use_pllr` assignment from Task 2 with:

```python
        if use_pllr is not None:
            # Legacy callers passed a bool. Honour it, but not alongside the
            # richer flag -- silently preferring one would make an ablation
            # cell run a configuration its name does not describe.
            if pllr_mode != "residual":
                raise ValueError(
                    "pass either pllr_mode or use_pllr, not both "
                    f"(got pllr_mode={pllr_mode!r}, use_pllr={use_pllr!r})")
            pllr_mode = "concat" if use_pllr else "off"
        if pllr_mode not in PLLR_MODES:
            raise ValueError(f"pllr_mode must be one of {PLLR_MODES}; got {pllr_mode!r}")
        self.pllr_mode = pllr_mode
        self.use_pllr = pllr_mode != "off"
```

After `self.out = nn.Linear(hidden_dim, 1)` (line 176), add:

```python
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
```

Replace `classify`'s body from Task 2 with:

```python
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
```

Update `config()` (lines 194-201) — replace the `"use_pllr": self.use_pllr` entry with `"pllr_mode": self.pllr_mode`.

Update `head_params()` (line 188) so the gain is optimised with the head:

```python
    def head_params(self) -> List[torch.nn.Parameter]:
        params = list(self.head.parameters()) + list(self.out.parameters())
        if self.pllr_mode == "residual":
            params.append(self.pllr_gain)
        return params
```

Update `load_finetuned_model` (lines 595-600) to migrate legacy configs:

```python
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
```

Add `"PLLR_MODES"` to `__all__` (line 608).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_esm_finetune.py -v && .venv/bin/python -m pytest tests/ -q`
Expected: PASS, 104 tests total

- [ ] **Step 5: Commit**

```bash
git add src/esm_finetune.py tests/test_esm_finetune.py
git commit -m "feat(esm_finetune): PLLR as a zero-initialised residual base

An untrained residual-mode model now reproduces the zero-shot ESM ranking
(ROC-AUC ~0.834 on the MMR panel) instead of discarding it behind a random
init weight, so training learns a correction on a working predictor rather
than rediscovering one from a few hundred labels. Legacy use_pllr bools and
checkpoints map to the historical 'concat' behaviour.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CP3LLMEoVNhJx8zUvHBCkm"
```

---

### Task 4: Prior-feature branch

The measured 6.5-point gap between Stage 2b (0.880) and the frozen priors probe (0.945) confounds freeze depth with feature set: the probe reads 27 prior columns, the fine-tune reads none. This task removes the confound.

**Files:**
- Modify: `src/esm_finetune.py` (`__init__`, `classify`, `config`, `FineTuneExample`, `build_examples`, `make_collate_fn`, `_forward`, `predict_proba`, `load_finetuned_model`)
- Test: `tests/test_esm_finetune.py`

**Interfaces:**
- Consumes: Task 1's `build_fusion_head(..., norm=)`; Tasks 2–3's `encode`/`classify`/`pllr_mode`.
- Produces:
  - `ESMFineTuneClassifier(..., n_prior_features: int = 0, fusion: str = "concat", shared_dim: int = 128)`. When `n_prior_features == 0` the architecture is exactly Task 3's.
  - `FineTuneExample.priors: Optional[np.ndarray] = None`.
  - `build_examples(..., prior_matrix: Optional[np.ndarray] = None)` — row-aligned to `df`, so a row dropped by wt-validation drops its prior vector with it.
  - Collate emits `"priors"` `[B, P]` float32 when priors are present.
  - `config()` gains `n_prior_features`, `fusion`, `shared_dim`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_esm_finetune.py`:

```python
def test_prior_branch_changes_the_logit():
    model = tiny_model_or_skip(mode="siamese", n_unfrozen_layers=0,
                               pllr_mode="off", n_prior_features=5)
    model.eval()
    block = torch.randn(3, model.esm_block_dim)
    pllr = torch.zeros(3)
    with torch.no_grad():
        a = model.classify(block, pllr, priors=torch.zeros(3, 5))
        b = model.classify(block, pllr, priors=torch.ones(3, 5))
    assert not torch.allclose(a, b), "prior vector must reach the head"


def test_prior_branch_requires_priors_when_configured():
    model = tiny_model_or_skip(mode="siamese", n_unfrozen_layers=0,
                               n_prior_features=5)
    with pytest.raises(ValueError, match="priors"):
        model.classify(torch.randn(2, model.esm_block_dim), torch.zeros(2))


def test_prior_branch_rejects_wrong_width():
    model = tiny_model_or_skip(mode="siamese", n_unfrozen_layers=0,
                               n_prior_features=5)
    with pytest.raises(ValueError, match="5"):
        model.classify(torch.randn(2, model.esm_block_dim), torch.zeros(2),
                       priors=torch.zeros(2, 4))


def test_prior_branch_works_at_micro_batch_one():
    """The only config that fits a 650M full fine-tune on a 15 GiB card."""
    model = tiny_model_or_skip(mode="siamese", n_unfrozen_layers=0,
                               n_prior_features=5, fusion="concat")
    model.train()
    out = model.classify(torch.randn(1, model.esm_block_dim), torch.zeros(1),
                         priors=torch.zeros(1, 5))
    assert out.shape == (1,) and torch.isfinite(out).all()


def test_gatewave_fusion_also_supported():
    model = tiny_model_or_skip(mode="siamese", n_unfrozen_layers=0,
                               n_prior_features=5, fusion="gatewave")
    model.train()
    out = model.classify(torch.randn(2, model.esm_block_dim), torch.zeros(2),
                         priors=torch.randn(2, 5))
    assert out.shape == (2,)


def test_residual_pllr_still_dominates_at_init_with_priors():
    """Zero-init of the fusion head's scalar projection must hold for the
    fusion path too, or the residual guarantee silently applies to only one
    architecture."""
    model = tiny_model_or_skip(mode="siamese", n_unfrozen_layers=0,
                               pllr_mode="residual", n_prior_features=5)
    model.eval()
    pllr = torch.tensor([-8.0, 1.0])
    with torch.no_grad():
        logits = model.classify(torch.randn(2, model.esm_block_dim), pllr,
                                priors=torch.randn(2, 5))
        expected = model.pllr_gain * (pllr / model.pllr_scale)
    torch.testing.assert_close(logits, expected)


def test_build_examples_carries_prior_rows_and_drops_them_with_their_row():
    model = tiny_model_or_skip(mode="siamese", n_unfrozen_layers=0)
    df = demo_frame()
    df.loc[1, "wt_aa"] = "X"          # forces a wt-validation drop
    priors = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
    ex = build_examples(df, {"G": SEQ}, "siamese", max_residues=64,
                        tokenizer=model.tokenizer, prior_matrix=priors)
    assert len(ex) == 3
    # Row 1 was dropped, so surviving examples keep rows 0, 2, 3.
    np.testing.assert_allclose(
        np.stack([e.priors for e in ex]), priors[[0, 2, 3]])


def test_collate_emits_prior_tensor():
    model = tiny_model_or_skip(mode="siamese", n_unfrozen_layers=0)
    priors = np.ones((4, 3), dtype=np.float32)
    ex = build_examples(demo_frame(), {"G": SEQ}, "siamese", max_residues=64,
                        tokenizer=model.tokenizer, prior_matrix=priors)
    batch = make_collate_fn(model.tokenizer, "siamese")(ex)
    assert batch["priors"].shape == (4, 3)
    assert batch["priors"].dtype == torch.float32


def test_build_examples_rejects_misaligned_prior_matrix():
    model = tiny_model_or_skip(mode="siamese", n_unfrozen_layers=0)
    with pytest.raises(ValueError, match="rows"):
        build_examples(demo_frame(), {"G": SEQ}, "siamese", max_residues=64,
                       tokenizer=model.tokenizer,
                       prior_matrix=np.zeros((2, 3), dtype=np.float32))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_esm_finetune.py -k prior -v`
Expected: FAIL — `__init__() got an unexpected keyword argument 'n_prior_features'`

- [ ] **Step 3: Write minimal implementation**

In `src/esm_finetune.py`, add to the imports (after line 51):

```python
from .fusion import build_fusion_head
```

Extend the constructor signature with `n_prior_features: int = 0, fusion: str = "concat", shared_dim: int = 128`, store them, and build the fusion head. Replace the head-construction block (lines 161-183, i.e. from `emb_dim = ...` through the `pllr_scale` buffer) with:

```python
        emb_dim = self.esm_block_dim
        # LayerNorm BEFORE the first Linear. Raw ESM hidden states have large,
        # position-dependent norms, and in siamese mode half the concatenated
        # vector (site_wt, site_mut) is near-duplicate while the informative
        # part (their difference at a single substituted residue) is small --
        # so an unnormalised input made the head spend its capacity on scale
        # rather than on the signal.
        self.feat_norm = nn.LayerNorm(emb_dim)
        in_dim = emb_dim + (1 if self.pllr_mode == "concat" else 0)
        self.n_prior_features = int(n_prior_features)
        self.fusion = fusion
        self.shared_dim = int(shared_dim)
        if self.n_prior_features > 0:
            # Reuse the Stage-2 fusion heads rather than a second
            # implementation, so Stage-2b architectures stay directly
            # comparable to Stage-2's concat_fusion / gatewave_fusion cells.
            # norm="layer": the fine-tune runs at micro-batch 1 on a 15 GiB
            # card, where BatchNorm1d cannot compute batch statistics.
            self.fusion_head: Optional[nn.Module] = build_fusion_head(
                fusion, (in_dim, self.n_prior_features),
                shared_dim=self.shared_dim, dropout=dropout, norm="layer")
            self.head = None
            self.out = None
        else:
            self.fusion_head = None
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
        #: the feature identical between training and single-variant inference,
        #: and keeps it comparable across genes and batch sizes.
        self.register_buffer("pllr_scale", torch.tensor(10.0))
        if self.pllr_mode == "residual":
            #: Learnable gain on the zero-shot term. Negative at init because
            #: raw PLLR is negative for damaging variants while pathogenic is
            #: the positive class.
            self.pllr_gain = nn.Parameter(torch.tensor(-1.0))
            _zero_init_scalar_output(self._scoring_module())
```

Add two helpers above the class (after `_autocast`, line 87):

```python
def _zero_init_scalar_output(module: nn.Module) -> None:
    """Zero the last ``Linear(..., 1)`` in *module*.

    Used by the PLLR residual mode so the untrained model's logit is exactly
    the scaled zero-shot term. Works for the plain head and for both fusion
    heads, which all end in a scalar projection.
    """
    last = None
    for m in module.modules():
        if isinstance(m, nn.Linear) and m.out_features == 1:
            last = m
    if last is None:
        raise ValueError("no scalar output Linear found to zero-initialise")
    nn.init.zeros_(last.weight)
    if last.bias is not None:
        nn.init.zeros_(last.bias)
```

And a small accessor method on the class, next to `head_params`:

```python
    def _scoring_module(self) -> nn.Module:
        """Whichever submodule turns features into a logit."""
        return self.fusion_head if self.fusion_head is not None else self.out
```

Update `head_params()`:

```python
    def head_params(self) -> List[torch.nn.Parameter]:
        if self.fusion_head is not None:
            params = list(self.fusion_head.parameters())
        else:
            params = list(self.head.parameters()) + list(self.out.parameters())
        if self.pllr_mode == "residual":
            params.append(self.pllr_gain)
        return params
```

Replace `classify`:

```python
    def classify(self, esm_block: torch.Tensor, pllr: torch.Tensor,
                 priors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``(esm_block, pllr[, priors])`` -> logits ``[B]``."""
        feat = self.feat_norm(esm_block)
        if self.pllr_mode == "concat":
            feat = torch.cat([feat, (pllr / self.pllr_scale).unsqueeze(-1)], dim=-1)
        if self.fusion_head is not None:
            if priors is None:
                raise ValueError(
                    f"this model was built with n_prior_features="
                    f"{self.n_prior_features}, so classify() requires priors.")
            if priors.shape[-1] != self.n_prior_features:
                raise ValueError(
                    f"expected {self.n_prior_features} prior features, got "
                    f"{priors.shape[-1]}")
            logit = self.fusion_head(feat, priors.to(feat.dtype))
        else:
            logit = self.out(self.head(feat)).squeeze(-1)
        if self.pllr_mode == "residual":
            logit = logit + self.pllr_gain * (pllr / self.pllr_scale)
        return logit
```

Extend `forward` with a `priors` parameter passed straight to `classify` (already present from Task 2).

Add to `config()`: `"n_prior_features": self.n_prior_features, "fusion": self.fusion, "shared_dim": self.shared_dim`. Mirror them in `load_finetuned_model`'s constructor call with `cfg.get("n_prior_features", 0)`, `cfg.get("fusion", "concat")`, `cfg.get("shared_dim", 128)`.

Extend `FineTuneExample` (line 262) with `priors: Optional[np.ndarray] = None`.

In `build_examples`, add the `prior_matrix: Optional[np.ndarray] = None` parameter, validate and index it. After `has_weight = "label_weight" in df.columns` insert:

```python
    if prior_matrix is not None and len(prior_matrix) != len(df):
        raise ValueError(
            f"prior_matrix has {len(prior_matrix)} rows but df has {len(df)}; "
            "it must be row-aligned to df so a wt-validation drop removes the "
            "variant and its prior vector together.")
```

Change the row loop to `for row_i, row in enumerate(df.itertuples(index=False)):` and add `priors=None if prior_matrix is None else np.asarray(prior_matrix[row_i], dtype=np.float32)` to the `FineTuneExample(...)` construction.

In `make_collate_fn`, before `return out`:

```python
        if batch[0].priors is not None:
            out["priors"] = torch.tensor(
                np.stack([b.priors for b in batch]), dtype=torch.float32)
```

In `_forward` (line 361), forward the priors:

```python
def _forward(model: ESMFineTuneClassifier, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    tok = {"wt_tok_id": batch.get("wt_tok_id"),
           "mut_tok_id": batch.get("mut_tok_id"),
           "priors": batch.get("priors")}
    if model.mode == "siamese":
        return model(batch["wt_ids"], batch["wt_mask"], batch["wt_pos"],
                     batch["mut_ids"], batch["mut_mask"], batch["mut_pos"], **tok)
    return model(batch["wt_ids"], batch["wt_mask"], batch["wt_pos"], **tok)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_esm_finetune.py -v && .venv/bin/python -m pytest tests/ -q`
Expected: PASS, 113 tests total

- [ ] **Step 5: Commit**

```bash
git add src/esm_finetune.py tests/test_esm_finetune.py
git commit -m "feat(esm_finetune): optional prior-feature branch via fusion heads

Stage 2b read zero of the 27 prior columns the frozen probe reads, so the
measured 6.5-point gap between them confounded freeze depth with feature set.
Reuses src/fusion.py's concat and GateWave heads so Stage-2b architectures
stay comparable to Stage-2's fusion cells.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CP3LLMEoVNhJx8zUvHBCkm"
```

---

### Task 5: Warmup + cosine LR schedule and `pos_weight`

`fit_esm_finetune` runs bare AdamW at a constant 1e-5 while `src/train.py:140` and `src/transfer.py:389` both use cosine annealing and a positive-class weight. A 650M backbone taking full-size steps from step 0 on ~300–500 examples is a plausible cause of the `best_epoch = 1–5` early stopping.

**Files:**
- Modify: `src/esm_finetune.py` (`fit_esm_finetune`, new module-level helpers)
- Test: `tests/test_esm_finetune.py`

**Interfaces:**
- Consumes: nothing from Tasks 1–4.
- Produces:
  - `make_lr_schedule(optimizer, total_steps: int, warmup_frac: float = 0.1) -> torch.optim.lr_scheduler.LambdaLR`
  - `positive_class_weight(labels: Sequence[float]) -> float` returning `n_neg / n_pos`, or `1.0` when either class is empty.
  - `fit_esm_finetune(..., warmup_frac: float = 0.1, use_pos_weight: bool = True)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_esm_finetune.py`:

```python
from src.esm_finetune import make_lr_schedule, positive_class_weight  # noqa: E402


def test_positive_class_weight_matches_n_neg_over_n_pos():
    assert positive_class_weight([1, 1, 0, 0, 0, 0]) == pytest.approx(2.0)
    assert positive_class_weight([1, 0]) == pytest.approx(1.0)


def test_positive_class_weight_is_one_for_a_degenerate_partition():
    """A single-class partition has no meaningful ratio; 1.0 keeps the loss
    finite instead of producing inf or a divide-by-zero."""
    assert positive_class_weight([1, 1, 1]) == pytest.approx(1.0)
    assert positive_class_weight([0, 0]) == pytest.approx(1.0)
    assert positive_class_weight([]) == pytest.approx(1.0)


def test_lr_schedule_warms_up_then_decays_to_zero():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=1.0)
    sched = make_lr_schedule(opt, total_steps=100, warmup_frac=0.1)
    seen = []
    for _ in range(100):
        seen.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert seen[0] < seen[5] < seen[9], "warmup must ramp"
    assert seen[9] == pytest.approx(1.0, abs=1e-6), "peaks at base LR"
    assert seen[-1] < 0.01, "cosine must decay to ~0"
    assert all(seen[i] >= seen[i + 1] - 1e-9 for i in range(10, 99)), \
        "must decay monotonically after warmup"


def test_lr_schedule_never_divides_by_zero_on_tiny_runs():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=1.0)
    sched = make_lr_schedule(opt, total_steps=1, warmup_frac=0.1)
    opt.step()
    sched.step()
    assert np.isfinite(opt.param_groups[0]["lr"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_esm_finetune.py -k "lr_schedule or positive_class" -v`
Expected: FAIL — `ImportError: cannot import name 'make_lr_schedule'`

- [ ] **Step 3: Write minimal implementation**

Add `import math` to the imports of `src/esm_finetune.py`, then add above `fit_esm_finetune` (line 399):

```python
def positive_class_weight(labels: Sequence[float]) -> float:
    """``n_neg / n_pos`` on the fitting partition, or 1.0 if a class is absent.

    Mirrors :mod:`src.train` and :mod:`src.transfer`, which both weight the
    positive class. Per-gene prevalence on the MMR panel runs 40-82% positive,
    so an unweighted objective optimises a different operating point in every
    leave-one-gene-out fold.
    """
    arr = np.asarray(list(labels), dtype=np.float64)
    if arr.size == 0:
        return 1.0
    n_pos = float((arr > 0.5).sum())
    n_neg = float(arr.size - n_pos)
    if n_pos == 0.0 or n_neg == 0.0:
        return 1.0
    return n_neg / n_pos


def make_lr_schedule(optimizer: torch.optim.Optimizer, total_steps: int,
                     warmup_frac: float = 0.1):
    """Linear warmup then cosine decay to zero, stepped per *optimizer* step.

    Per-epoch granularity is too coarse here: a leave-one-gene-out fold runs
    ten epochs over a few hundred examples, so the whole schedule is a few
    hundred optimizer steps. Warmup matters because the alternative -- a 650M
    backbone taking full-size steps from step 0 on ~300-500 labels -- damages
    the pretrained representation before the head has learned to read it.
    """
    total = max(1, int(total_steps))
    warmup = min(total, max(1, int(round(total * warmup_frac))))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

In `fit_esm_finetune`, add `warmup_frac: float = 0.1` and `use_pos_weight: bool = True` to the signature. After the optimizer is created (line 454) add:

```python
    steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum_steps))
    scheduler = make_lr_schedule(optimizer, epochs * steps_per_epoch, warmup_frac)
    pos_w = (positive_class_weight([e.label for e in train_examples])
             if use_pos_weight else 1.0)
    logger.info("LR schedule: %d warmup + cosine over %d optimizer steps | "
                "pos_weight=%.3f", max(1, int(round(epochs * steps_per_epoch
                                                    * warmup_frac))),
                epochs * steps_per_epoch, pos_w)
```

Note `train_loader` is currently constructed *after* the optimizer; move the `train_loader = DataLoader(...)` block (lines 456-458) above the optimizer so `len(train_loader)` is available.

Replace the loss computation (lines 475-477) with:

```python
            weights = batch["weights"]
            if pos_w != 1.0:
                weights = weights * torch.where(
                    batch["labels"] > 0.5,
                    torch.full_like(weights, pos_w),
                    torch.ones_like(weights))
            # BCE in fp32 regardless of autocast: the loss reduction is where
            # reduced precision actually costs accuracy, and it is free here.
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                logits.float(), batch["labels"], reduction="none")
                * weights).mean()
```

Add `scheduler.step()` immediately after `optimizer.zero_grad(set_to_none=True)` inside the accumulation-step branch (line 501).

Add `"make_lr_schedule", "positive_class_weight"` to `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_esm_finetune.py -v && .venv/bin/python -m pytest tests/ -q`
Expected: PASS, 117 tests total

- [ ] **Step 5: Commit**

```bash
git add src/esm_finetune.py tests/test_esm_finetune.py
git commit -m "feat(esm_finetune): warmup+cosine LR schedule and pos_weight

Brings the Stage-2b trainer in line with src/train.py and src/transfer.py,
which both anneal the LR and weight the positive class. Constant-LR AdamW on
a 650M backbone over ~300-500 labels is a plausible cause of the epoch-1
early stopping documented in PAPER_DRAFT.md 6.10.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CP3LLMEoVNhJx8zUvHBCkm"
```

---

### Task 6: Frozen-backbone precompute path

When `n_unfrozen_layers == 0` the backbone output is constant across epochs, so encoding it once turns a ~1 h ablation-floor cell into ~15 min. That is what makes three seeds affordable on every floor cell within the weekend budget (spec §5.1).

**Files:**
- Modify: `src/esm_finetune.py` (`fit_esm_finetune`, `predict_proba`)
- Test: `tests/test_esm_finetune.py`

**Interfaces:**
- Consumes: `encode`/`classify` from Task 2, the schedule from Task 5.
- Produces: `encode_all(model, examples, device, batch_size=16, amp=True) -> Tuple[Tensor, Tensor, Optional[Tensor]]` returning `(blocks [N, E], pllrs [N], priors [N, P] or None)` on CPU. `fit_esm_finetune` uses it automatically when `not model.trainable_backbone_params()`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_esm_finetune.py`:

```python
from src.esm_finetune import encode_all, fit_esm_finetune  # noqa: E402


def demo_examples(model, n_prior_features=0):
    priors = (None if n_prior_features == 0
              else np.zeros((4, n_prior_features), dtype=np.float32))
    return build_examples(demo_frame(), {"G": SEQ}, model.mode,
                          max_residues=64, tokenizer=model.tokenizer,
                          prior_matrix=priors)


def test_encode_all_matches_a_direct_forward_pass():
    """The precompute path must be numerically identical to encoding per
    batch, or the ablation floor silently measures a different model."""
    model = tiny_model_or_skip(mode="siamese", n_unfrozen_layers=0)
    model.eval()
    ex = demo_examples(model)
    blocks, pllrs, priors = encode_all(model, ex, torch.device("cpu"),
                                       batch_size=2, amp=False)
    assert priors is None
    batch = make_collate_fn(model.tokenizer, model.mode)(ex)
    with torch.no_grad():
        ref_block, ref_pllr = model.encode(
            **{k: v for k, v in batch.items() if k not in ("labels", "weights")})
    torch.testing.assert_close(blocks, ref_block)
    torch.testing.assert_close(pllrs, ref_pllr)


def test_encode_all_carries_priors_through():
    model = tiny_model_or_skip(mode="siamese", n_unfrozen_layers=0,
                               n_prior_features=3)
    model.eval()
    blocks, pllrs, priors = encode_all(model, demo_examples(model, 3),
                                       torch.device("cpu"), batch_size=2, amp=False)
    assert priors is not None and priors.shape == (4, 3)


def test_frozen_fit_uses_precompute_and_still_trains_the_head():
    model = tiny_model_or_skip(mode="siamese", n_unfrozen_layers=0,
                               pllr_mode="residual")
    ex = demo_examples(model)
    before = model.out.weight.detach().clone()
    fitted, best_epoch, best_auc = fit_esm_finetune(
        model, ex, ex, torch.device("cpu"), epochs=2, patience=2,
        batch_size=2, amp=False)
    assert best_epoch >= 1
    assert not torch.allclose(before, fitted.out.weight.detach()), \
        "head must have trained"


def test_frozen_fit_leaves_the_backbone_untouched():
    model = tiny_model_or_skip(mode="siamese", n_unfrozen_layers=0)
    ex = demo_examples(model)
    before = {k: v.detach().clone() for k, v in model.backbone.named_parameters()}
    fit_esm_finetune(model, ex, ex, torch.device("cpu"), epochs=2, patience=2,
                     batch_size=2, amp=False)
    for k, v in model.backbone.named_parameters():
        torch.testing.assert_close(before[k], v.detach())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_esm_finetune.py -k "encode_all or frozen_fit" -v`
Expected: FAIL — `ImportError: cannot import name 'encode_all'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/esm_finetune.py`, above `fit_esm_finetune`:

```python
def encode_all(model: "ESMFineTuneClassifier", examples: Sequence[FineTuneExample],
               device: torch.device, batch_size: int = 16, amp: bool = True,
               ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Run the backbone once over *examples* -> ``(blocks, pllrs, priors)``.

    Only valid with a frozen backbone, where the encoder output does not
    change between epochs. Returns CPU tensors; the caller moves the
    mini-batches it needs.
    """
    model.eval()
    amp_dtype = amp_dtype_for(device) if amp else None
    loader = DataLoader(FineTuneDataset(examples), batch_size=batch_size,
                        shuffle=False,
                        collate_fn=make_collate_fn(model.tokenizer, model.mode))
    blocks, pllrs, priors = [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch = _to_device(batch, device)
            with _autocast(device, amp_dtype):
                block, pllr = model.encode(
                    batch["wt_ids"], batch["wt_mask"], batch["wt_pos"],
                    batch.get("mut_ids"), batch.get("mut_mask"),
                    batch.get("mut_pos"), batch.get("wt_tok_id"),
                    batch.get("mut_tok_id"))
            blocks.append(block.float().cpu())
            pllrs.append(pllr.float().cpu())
            if "priors" in batch:
                priors.append(batch["priors"].float().cpu())
    return (torch.cat(blocks), torch.cat(pllrs),
            torch.cat(priors) if priors else None)


class _PrecomputedDataset(Dataset):
    """Encoded features + targets, for head-only training."""

    def __init__(self, blocks, pllrs, priors, labels, weights) -> None:
        self.blocks, self.pllrs, self.priors = blocks, pllrs, priors
        self.labels, self.weights = labels, weights

    def __len__(self) -> int:
        return int(self.blocks.shape[0])

    def __getitem__(self, i: int):
        priors = None if self.priors is None else self.priors[i]
        return self.blocks[i], self.pllrs[i], priors, self.labels[i], self.weights[i]


def _precomputed_collate(batch):
    blocks = torch.stack([b[0] for b in batch])
    pllrs = torch.stack([b[1] for b in batch])
    priors = None if batch[0][2] is None else torch.stack([b[2] for b in batch])
    labels = torch.stack([b[3] for b in batch])
    weights = torch.stack([b[4] for b in batch])
    return blocks, pllrs, priors, labels, weights
```

In `fit_esm_finetune`, immediately after `model.to(device)`, add the branch:

```python
    frozen = not model.trainable_backbone_params()
    if frozen:
        # The encoder output is constant across epochs, so encode once instead
        # of ten times. This is what makes the ablation floor cheap enough to
        # carry three seeds within the weekend GPU budget.
        logger.info("Frozen backbone: precomputing features for %d train / "
                    "%d val examples.", len(train_examples), len(val_examples))
        return _fit_precomputed(
            model, train_examples, val_examples, device,
            head_lr=head_lr, epochs=epochs, patience=patience,
            batch_size=batch_size, weight_decay=weight_decay,
            warmup_frac=warmup_frac, use_pos_weight=use_pos_weight,
            max_grad_norm=max_grad_norm, amp=amp)
```

Then add `_fit_precomputed` after `fit_esm_finetune`:

```python
def _fit_precomputed(model, train_examples, val_examples, device, *, head_lr,
                     epochs, patience, batch_size, weight_decay, warmup_frac,
                     use_pos_weight, max_grad_norm, amp,
                     ) -> Tuple["ESMFineTuneClassifier", int, float]:
    """Head-only training on features encoded once. Frozen backbone only."""
    from sklearn.metrics import roc_auc_score

    tr_block, tr_pllr, tr_prior = encode_all(model, train_examples, device,
                                             batch_size=batch_size, amp=amp)
    va_block, va_pllr, va_prior = encode_all(model, val_examples, device,
                                             batch_size=batch_size, amp=amp)
    tr_y = torch.tensor([e.label for e in train_examples], dtype=torch.float32)
    tr_w = torch.tensor([e.weight for e in train_examples], dtype=torch.float32)
    val_labels = np.array([e.label for e in val_examples], dtype=np.float32)

    pos_w = positive_class_weight([e.label for e in train_examples]) if use_pos_weight else 1.0
    if pos_w != 1.0:
        tr_w = tr_w * torch.where(tr_y > 0.5, torch.full_like(tr_w, pos_w),
                                  torch.ones_like(tr_w))

    optimizer = torch.optim.AdamW(model.head_params(), lr=head_lr,
                                  weight_decay=weight_decay)
    loader = DataLoader(_PrecomputedDataset(tr_block, tr_pllr, tr_prior, tr_y, tr_w),
                        batch_size=max(2, batch_size), shuffle=True,
                        collate_fn=_precomputed_collate)
    scheduler = make_lr_schedule(optimizer, epochs * max(1, len(loader)), warmup_frac)

    best_auc, best_state, best_epoch, left = -np.inf, None, -1, patience
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for blocks, pllrs, priors, labels, weights in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model.classify(blocks.to(device), pllrs.to(device),
                                    None if priors is None else priors.to(device))
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                logits.float(), labels.to(device), reduction="none")
                * weights.to(device)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head_params(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            running += float(loss.item()) * len(labels)

        model.eval()
        with torch.inference_mode():
            val_logits = model.classify(
                va_block.to(device), va_pllr.to(device),
                None if va_prior is None else va_prior.to(device))
            val_probs = torch.sigmoid(val_logits.float()).cpu().numpy()
        try:
            val_auc = float(roc_auc_score(val_labels, val_probs)) \
                if len(np.unique(val_labels)) > 1 else -np.inf
        except ValueError:
            val_auc = -np.inf
        logger.info("epoch %d/%d [frozen]: train_loss=%.4f val_auc=%.4f",
                    epoch + 1, epochs, running / max(1, len(train_examples)), val_auc)

        if val_auc > best_auc:
            best_auc, best_epoch, left = val_auc, epoch + 1, patience
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            left -= 1
            if left <= 0:
                logger.info("Early stopping at epoch %d (best val_auc=%.4f @ epoch %d).",
                            epoch + 1, best_auc, best_epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, max(best_epoch, 0), float(best_auc)
```

Add `"encode_all"` to `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_esm_finetune.py -v && .venv/bin/python -m pytest tests/ -q`
Expected: PASS, 122 tests total

- [ ] **Step 5: Commit**

```bash
git add src/esm_finetune.py tests/test_esm_finetune.py
git commit -m "perf(esm_finetune): precompute features when the backbone is frozen

A frozen encoder produces the same output every epoch, so encode once and
train the head on cached tensors. Drops an ablation-floor cell from ~1h to
~15min, which is what makes 3 seeds per floor cell fit the weekend budget.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CP3LLMEoVNhJx8zUvHBCkm"
```

---

### Task 7: Cell identity, AF quarantine, and the grid spec

**Files:**
- Create: `src/finetune_grid.py`
- Modify: `src/transfer.py` (add `AF_DERIVED_PRIOR_COLS` and `assert_af_quarantine`)
- Create: `tests/test_finetune_grid.py`

**Interfaces:**
- Consumes: nothing from Tasks 1–6.
- Produces:
  - `src.transfer.AF_DERIVED_PRIOR_COLS: Tuple[str, ...]`
  - `src.transfer.assert_af_quarantine(prior_cols, af_labels_active: bool) -> None` — raises `ValueError` when both are active.
  - `src.finetune_grid.GridCell` dataclass with fields `branch` (`"esm"|"esm+priors"`), `n_unfrozen_layers`, `pllr_mode`, `seed`, `fusion`, `tier`, plus `slug()` and `to_dict()`.
  - `src.finetune_grid.TIERS: Dict[str, List[GridCell]]` keyed `"1".."6"`, and `cells_for(tiers: Sequence[str]) -> List[GridCell]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_finetune_grid.py`:

```python
"""Grid-cell identity, tier composition, and the AF label/feature quarantine."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.finetune_grid import TIERS, GridCell, cells_for  # noqa: E402
from src.transfer import AF_DERIVED_PRIOR_COLS, assert_af_quarantine  # noqa: E402


def test_af_quarantine_raises_when_labels_and_features_are_both_active():
    """Minting benign labels from allele frequency while feeding allele
    frequency as a feature makes acmg_bs1 == label by construction -- the
    same target leakage as Finding 2, in the paper that reports Finding 2."""
    with pytest.raises(ValueError, match="quarantine"):
        assert_af_quarantine(["am_pathogenicity", "acmg_bs1"], af_labels_active=True)


def test_af_quarantine_passes_when_af_columns_are_removed():
    assert_af_quarantine(["am_pathogenicity", "in_domain"], af_labels_active=True)


def test_af_quarantine_allows_af_features_without_af_labels():
    assert_af_quarantine(list(AF_DERIVED_PRIOR_COLS), af_labels_active=False)


def test_af_derived_cols_cover_every_frequency_column():
    assert set(AF_DERIVED_PRIOR_COLS) == {
        "gnomad_log10_af", "acmg_ba1", "acmg_bs1", "acmg_pm2"}


def test_cell_slug_is_deterministic_and_distinguishes_every_axis():
    base = dict(branch="esm+priors", n_unfrozen_layers=-1,
                pllr_mode="residual", seed=42, fusion="concat")
    a = GridCell(**base)
    assert a.slug() == GridCell(**base).slug(), "slug must be deterministic"
    for field, other in [("branch", "esm"), ("n_unfrozen_layers", 0),
                         ("pllr_mode", "off"), ("seed", 43),
                         ("fusion", "gatewave")]:
        assert GridCell(**{**base, field: other}).slug() != a.slug(), \
            f"slug must vary with {field}"


def test_cell_slug_is_filesystem_safe():
    slug = GridCell(branch="esm+priors", n_unfrozen_layers=-1,
                    pllr_mode="residual", seed=42).slug()
    assert all(c.isalnum() or c in "-_" for c in slug), slug


def test_tier_one_is_the_headline_fair_fight_with_three_seeds():
    cells = TIERS["1"]
    assert {c.seed for c in cells} == {42, 43, 44}
    assert {c.branch for c in cells} == {"esm+priors"}
    assert {c.n_unfrozen_layers for c in cells} == {-1, 0}
    assert len(cells) == 6


def test_cells_for_deduplicates_and_preserves_tier_order():
    cells = cells_for(["1", "2", "1"])
    assert len(cells) == len({c.slug() for c in cells})
    assert cells[0].tier == "1"


def test_every_tier_cell_has_a_unique_slug():
    everything = cells_for(sorted(TIERS))
    assert len(everything) == len({c.slug() for c in everything})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_finetune_grid.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.finetune_grid'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/transfer.py` after `GENE_CONSTANT_PRIOR_COLS` (line 82):

```python
#: gnomAD allele-frequency columns and the ACMG flags derived from them.
#: :func:`src.gnomad.add_frequency_flags` defines acmg_ba1 as ``AF > 0.05``,
#: acmg_bs1 as ``AF > bs1_af`` and acmg_pm2 as ``AF < pm2_af``, so any label
#: minted from allele frequency is reproduced exactly by these columns.
AF_DERIVED_PRIOR_COLS: Tuple[str, ...] = (
    "gnomad_log10_af", "acmg_ba1", "acmg_bs1", "acmg_pm2",
)


def assert_af_quarantine(prior_cols: Sequence[str],
                         af_labels_active: bool) -> None:
    """Refuse a configuration that both labels *and* features on frequency.

    Minting benign labels from gnomAD allele frequency while feeding allele
    frequency as a feature makes ``acmg_bs1 == label`` by construction on
    every minted row -- the same target leakage as ``dms_bin_median ==
    1 - label`` (docs/PAPER.md Finding 2). Enforced here rather than left to
    convention, because the failure is invisible in every metric.
    """
    if not af_labels_active:
        return
    offenders = sorted(set(prior_cols) & set(AF_DERIVED_PRIOR_COLS))
    if offenders:
        raise ValueError(
            "AF quarantine violated: allele-frequency-derived labels are "
            f"active while these AF-derived features are in the feature set: "
            f"{offenders}. Drop them, or disable the AF labels.")
```

Add both names to `src/transfer.py`'s `__all__` (line 696).

Create `src/finetune_grid.py`:

```python
"""Stage-2b ablation grid: cell identity, tiers, and output naming.

The grid answers a question docs/PAPER_DRAFT.md 6.12 asks but cannot
currently settle. That section varies freeze depth only, and compares the
result against a frozen *priors* probe -- but Stage 2b reads no prior
features at all, so the 6.5-point gap it measures confounds freeze depth
with feature set. Adding the ``branch`` axis makes freeze depth measurable
with the feature set held constant.

Cells are named by a deterministic slug so every artefact is
self-identifying. Before this existed, RUNLOG.md instructed the operator to
"move each esm_finetune_results_siamese_lopo.csv aside before the next run"
across a dozen cells -- an error-prone manual step in the middle of a
multi-day run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

BRANCHES = ("esm", "esm+priors")


@dataclass(frozen=True)
class GridCell:
    """One configuration of the Stage-2b ablation grid."""

    branch: str
    n_unfrozen_layers: int
    pllr_mode: str
    seed: int = 42
    fusion: str = "concat"
    tier: str = ""

    def __post_init__(self) -> None:
        if self.branch not in BRANCHES:
            raise ValueError(f"branch must be one of {BRANCHES}; got {self.branch!r}")

    def slug(self) -> str:
        """Deterministic, filesystem-safe identifier for this cell."""
        branch = "esmpri" if self.branch == "esm+priors" else "esm"
        nuf = {-1: "full", 0: "frozen"}.get(
            self.n_unfrozen_layers, f"last{self.n_unfrozen_layers}")
        parts = [branch, nuf, f"pllr-{self.pllr_mode}", f"seed{self.seed}"]
        if self.branch == "esm+priors":
            parts.insert(1, self.fusion)
        return "_".join(parts)

    def to_dict(self) -> Dict[str, object]:
        return {"branch": self.branch,
                "n_unfrozen_layers": self.n_unfrozen_layers,
                "pllr_mode": self.pllr_mode, "seed": self.seed,
                "fusion": self.fusion, "tier": self.tier, "slug": self.slug()}


def _tier(name: str, cells: Sequence[GridCell]) -> List[GridCell]:
    return [GridCell(**{**c.to_dict_kwargs(), "tier": name}) for c in cells] \
        if False else [
            GridCell(branch=c.branch, n_unfrozen_layers=c.n_unfrozen_layers,
                     pllr_mode=c.pllr_mode, seed=c.seed, fusion=c.fusion,
                     tier=name)
            for c in cells]


#: Tiers are ordered by scientific value so an interrupted run degrades
#: gracefully: tier 1 alone settles the paper's headline claim.
TIERS: Dict[str, List[GridCell]] = {
    # 1 -- the headline fair fight: does ESM add anything on top of priors?
    "1": _tier("1", [
        GridCell("esm+priors", nuf, "residual", seed)
        for nuf in (-1, 0) for seed in (42, 43, 44)
    ]),
    # 2 -- branch attribution: how much of any gain is priors vs backbone?
    "2": _tier("2", [
        GridCell("esm", -1, "residual", 42),
        GridCell("esm", 0, "residual", 42),
    ]),
    # 3 -- the PLLR axis, measured where it is clean: at the frozen floor the
    # backbone cannot relearn the term, so the on/off gap is attributable.
    "3": _tier("3", [
        GridCell("esm+priors", 0, "off", 42),
        GridCell("esm", 0, "off", 42),
        GridCell("esm+priors", 0, "concat", 42),
        GridCell("esm", 0, "concat", 42),
        GridCell("esm+priors", 0, "residual", 42, fusion="gatewave"),
    ]),
    # 4 -- freeze-depth middle ground.
    "4": _tier("4", [
        GridCell("esm+priors", 2, "residual", 42),
        GridCell("esm", 2, "residual", 42),
    ]),
    # 5 -- PLLR at full fine-tune (confounded with the backbone relearning it,
    # which is exactly why tier 3 exists as well).
    "5": _tier("5", [GridCell("esm+priors", -1, "off", 42)]),
}


def cells_for(tiers: Sequence[str]) -> List[GridCell]:
    """Cells for *tiers*, in tier order, de-duplicated by slug."""
    seen, out = set(), []
    for name in tiers:
        if name not in TIERS:
            raise ValueError(f"unknown tier {name!r}; expected {sorted(TIERS)}")
        for cell in TIERS[name]:
            if cell.slug() not in seen:
                seen.add(cell.slug())
                out.append(cell)
    return out


__all__ = ["BRANCHES", "GridCell", "TIERS", "cells_for"]
```

Simplify `_tier` to just the list comprehension (the `if False else` above is an artefact — write it as):

```python
def _tier(name: str, cells: Sequence[GridCell]) -> List[GridCell]:
    return [GridCell(branch=c.branch, n_unfrozen_layers=c.n_unfrozen_layers,
                     pllr_mode=c.pllr_mode, seed=c.seed, fusion=c.fusion,
                     tier=name)
            for c in cells]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_finetune_grid.py -v && .venv/bin/python -m pytest tests/ -q`
Expected: PASS, 131 tests total

- [ ] **Step 5: Commit**

```bash
git add src/finetune_grid.py src/transfer.py tests/test_finetune_grid.py
git commit -m "feat: Stage-2b grid cells, tiers, and the AF label/feature quarantine

Adds the branch axis PAPER_DRAFT 6.12 is missing, deterministic cell slugs so
artefacts are self-identifying, and a hard guard against minting benign labels
from allele frequency while feeding allele frequency as a feature.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CP3LLMEoVNhJx8zUvHBCkm"
```

---

### Task 8: Wire priors and predictions into `finetune_esm_mmr.py`

**Files:**
- Modify: `scripts/finetune_esm_mmr.py`
- Test: `tests/test_finetune_grid.py` (append; the helpers under test are pure)

**Interfaces:**
- Consumes: Tasks 3, 4, 7.
- Produces:
  - `scripts/finetune_esm_mmr.py::build_prior_inputs(ft_df, ho_df, tr_idx, va_idx, drop_gene_constant, af_labels_active) -> PriorInputs` with fields `train`, `val`, `holdout` (each `np.ndarray` or `None`), `columns: List[str]`, `impute_values: Dict[str, float]`, `mean: np.ndarray`, `scale: np.ndarray`.
  - New CLI flags: `--branch {esm,esm+priors}` (default `esm`), `--fusion {concat,gatewave}`, `--pllr_mode {residual,concat}`, `--cell_slug`, `--af_labels_active`.
  - Per-run outputs `predictions_<slug>.csv`, `results_<slug>.csv`, `summary_<slug>.json`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_finetune_grid.py`:

```python
import importlib.util

import numpy as np
import pandas as pd


def load_finetune_script():
    spec = importlib.util.spec_from_file_location(
        "finetune_esm_mmr", PROJECT_ROOT / "scripts" / "finetune_esm_mmr.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prior_frame(n=8):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "gene": ["MLH1"] * n,
        "am_pathogenicity": rng.random(n),
        "gnomad_pli": np.full(n, 0.99),      # gene-constant
        "gnomad_log10_af": rng.random(n),    # AF-derived
        "acmg_bs1": rng.integers(0, 2, n),   # AF-derived
        "label": rng.integers(0, 2, n),
    })


def test_prior_inputs_standardise_on_train_only():
    """Val and holdout must be transformed with the train partition's
    constants, never their own -- otherwise each split is centred differently
    and the head reads shifted inputs."""
    mod = load_finetune_script()
    ft, ho = prior_frame(8), prior_frame(4)
    out = mod.build_prior_inputs(ft, ho, [0, 1, 2, 3, 4, 5], [6, 7],
                                 drop_gene_constant=True,
                                 af_labels_active=False)
    assert out.train.shape[0] == 6 and out.val.shape[0] == 2
    assert out.holdout.shape[0] == 4
    # Train partition is standardised to ~zero mean by construction.
    np.testing.assert_allclose(out.train.mean(axis=0), 0.0, atol=1e-6)
    assert out.mean.shape[0] == out.train.shape[1]


def test_prior_inputs_drop_gene_constant_columns_under_lopo():
    mod = load_finetune_script()
    ft, ho = prior_frame(8), prior_frame(4)
    out = mod.build_prior_inputs(ft, ho, [0, 1, 2, 3], [4, 5],
                                 drop_gene_constant=True,
                                 af_labels_active=False)
    assert "gnomad_pli" not in out.columns


def test_prior_inputs_enforce_the_af_quarantine():
    mod = load_finetune_script()
    ft, ho = prior_frame(8), prior_frame(4)
    with pytest.raises(ValueError, match="quarantine"):
        mod.build_prior_inputs(ft, ho, [0, 1, 2, 3], [4, 5],
                               drop_gene_constant=True, af_labels_active=True)


def test_prior_inputs_are_finite_even_with_all_nan_columns():
    mod = load_finetune_script()
    ft, ho = prior_frame(8), prior_frame(4)
    ft["am_pathogenicity"] = np.nan
    ho["am_pathogenicity"] = np.nan
    out = mod.build_prior_inputs(ft, ho, [0, 1, 2, 3], [4, 5],
                                 drop_gene_constant=True,
                                 af_labels_active=False)
    assert np.isfinite(out.train).all() and np.isfinite(out.holdout).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_finetune_grid.py -k prior_inputs -v`
Expected: FAIL — `AttributeError: module 'finetune_esm_mmr' has no attribute 'build_prior_inputs'`

- [ ] **Step 3: Write minimal implementation**

In `scripts/finetune_esm_mmr.py`, extend the imports:

```python
from dataclasses import dataclass  # noqa: E402
from src.finetune_grid import GridCell  # noqa: E402
from src.transfer import (  # noqa: E402
    assert_af_quarantine,
    prior_columns_of,
    prior_impute_values,
    prior_matrix,
)
```

Add after `sample_weights_for` (line 158):

```python
@dataclass
class PriorInputs:
    """Standardised prior matrices for the three partitions of one split."""

    train: np.ndarray
    val: np.ndarray
    holdout: np.ndarray
    columns: list
    impute_values: dict
    mean: np.ndarray
    scale: np.ndarray


def build_prior_inputs(ft_df: pd.DataFrame, ho_df: pd.DataFrame,
                       tr_idx, va_idx, *, drop_gene_constant: bool,
                       af_labels_active: bool) -> PriorInputs:
    """Prior features for the fine-tune, inner-val and holdout partitions.

    Imputation medians and standardisation constants are fitted on the
    **fine-tune training rows only** and applied unchanged to inner-val and
    holdout: fitting them per partition would centre each split differently
    and hand the head shifted inputs (the same defect PAPER.md Finding 5
    records for the Stage-1/Stage-2 imputation constants).

    *drop_gene_constant* must be true for leave-one-gene-out. The five gnomAD
    gene-level columns hold one value per gene, so across genes they are a
    5-dimensional gene identifier rather than evidence -- RUNLOG.md 2026-08-28
    records MLH1 collapsing to ROC-AUC 0.500 in every seed with them kept.
    """
    columns = prior_columns_of(ft_df, drop_gene_constant=drop_gene_constant)
    assert_af_quarantine(columns, af_labels_active)
    if not columns:
        raise ValueError("no prior columns found in the fine-tune table")

    train_rows = ft_df.iloc[list(tr_idx)]
    impute = prior_impute_values(train_rows, columns)
    x_tr, cols = prior_matrix(train_rows, columns=columns, impute_values=impute)
    x_va, _ = prior_matrix(ft_df.iloc[list(va_idx)], columns=columns,
                           impute_values=impute)
    x_ho, _ = prior_matrix(ho_df, columns=columns, impute_values=impute)

    mean = x_tr.mean(axis=0)
    scale = x_tr.std(axis=0)
    scale[scale < 1e-8] = 1.0          # constant column -> leave it centred
    to_std = lambda x: ((x - mean) / scale).astype(np.float32)
    return PriorInputs(train=to_std(x_tr), val=to_std(x_va), holdout=to_std(x_ho),
                       columns=list(cols), impute_values=dict(impute),
                       mean=mean, scale=scale)
```

Add the CLI flags in `parse_args`:

```python
    p.add_argument("--branch", choices=("esm", "esm+priors"), default="esm",
                   help="'esm+priors' fuses the leakage-safe prior columns "
                        "into the fine-tune head. The frozen Stage-2 probe "
                        "reads 27 such columns; without this flag Stage 2b "
                        "reads none, so the two are not comparable.")
    p.add_argument("--fusion", choices=("concat", "gatewave"), default="concat")
    p.add_argument("--pllr_mode", choices=("residual", "concat"), default="residual",
                   help="How the zero-shot term enters the head. 'residual' "
                        "makes the untrained model the zero-shot predictor.")
    p.add_argument("--af_labels_active", action="store_true",
                   help="Declare that allele-frequency-derived labels are in "
                        "the training pool. Forces the AF-derived feature "
                        "columns out of the feature set.")
    p.add_argument("--cell_slug", type=str, default=None,
                   help="Tag for this cell's output files. Defaults to a slug "
                        "derived from the configuration.")
```

In `run_one_split`, after the `tr_local, va_local` split is computed, build the priors and pass them through:

```python
    priors = None
    if args.branch == "esm+priors":
        priors = build_prior_inputs(
            ft_df, ho_df, tr_local, va_local,
            drop_gene_constant=(args.eval == "lopo"),
            af_labels_active=args.af_labels_active)
        logger.info("Prior branch: %d columns (gene-constant dropped=%s)",
                    len(priors.columns), args.eval == "lopo")

    pllr_mode = args.pllr_mode if args.use_pllr else "off"
    model = ESMFineTuneClassifier(
        model_name=args.esm_model, mode=args.mode,
        n_unfrozen_layers=args.n_unfrozen_layers, hidden_dim=args.hidden_dim,
        dropout=args.dropout, gradient_checkpointing=args.gradient_checkpointing,
        pllr_mode=pllr_mode,
        n_prior_features=0 if priors is None else priors.train.shape[1],
        fusion=args.fusion)
```

and pass the matrices to `build_examples`:

```python
    train_ex = build_examples(ft_df.iloc[tr_local], sequence_by_gene, args.mode,
                              max_residues=args.max_residues, tokenizer=tok,
                              prior_matrix=None if priors is None else priors.train)
    val_ex = build_examples(ft_df.iloc[va_local], sequence_by_gene, args.mode,
                            max_residues=args.max_residues, tokenizer=tok,
                            prior_matrix=None if priors is None else priors.val)
    ho_ex = build_examples(ho_df, sequence_by_gene, args.mode,
                           max_residues=args.max_residues, tokenizer=tok,
                           prior_matrix=None if priors is None else priors.holdout)
```

Add `"branch"`, `"fusion"`, `"pllr_mode"` and `"cell_slug"` to the `row` dict, and return the per-variant predictions alongside it by adding to `row`:

```python
    row["_predictions"] = pd.DataFrame({
        "holdout_gene": holdout,
        "gene": ho_df["gene"].to_numpy()[:len(ho_probs)],
        "uniprot_id": ho_df["uniprot_id"].to_numpy()[:len(ho_probs)],
        "position": ho_df["position"].to_numpy()[:len(ho_probs)],
        "wt_aa": ho_df["wt_aa"].to_numpy()[:len(ho_probs)],
        "mut_aa": ho_df["mut_aa"].to_numpy()[:len(ho_probs)],
        "label": ho_labels,
        "prob": ho_probs,
        "threshold": float(thr),
        "seed": args.seed,
        "cell_slug": args.cell_slug,
    })
```

In `main`, derive the slug when not supplied and write the three tagged files:

```python
    if args.cell_slug is None:
        args.cell_slug = GridCell(
            branch=args.branch, n_unfrozen_layers=args.n_unfrozen_layers,
            pllr_mode=(args.pllr_mode if args.use_pllr else "off"),
            seed=args.seed, fusion=args.fusion).slug()
```

Replace the results-writing block so `predictions` are concatenated and saved, and every filename carries `args.cell_slug`:

```python
    predictions = pd.concat([r.pop("_predictions") for r in rows],
                            ignore_index=True) if rows else pd.DataFrame()
    results = pd.DataFrame(rows)
    tag = f"{args.mode}_{'lopo' if args.eval == 'lopo' else 'holdout_' + args.holdout_gene}_{args.cell_slug}"
    results_path = args.out_dir / f"esm_finetune_results_{tag}.csv"
    predictions_path = args.out_dir / f"esm_finetune_predictions_{tag}.csv"
    results.to_csv(results_path, index=False)
    predictions.to_csv(predictions_path, index=False)
```

and add `"cell": args.cell_slug`, `"branch": args.branch`, `"fusion": args.fusion`, `"pllr_mode": args.pllr_mode`, `"seed": args.seed`, `"predictions_csv": str(predictions_path)` to the `summary` dict.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_finetune_grid.py -v && .venv/bin/python -m pytest tests/ -q`
Expected: PASS, 135 tests total

- [ ] **Step 5: Commit**

```bash
git add scripts/finetune_esm_mmr.py tests/test_finetune_grid.py
git commit -m "feat(finetune_esm_mmr): prior branch, per-variant predictions, cell tags

A 45-hour GPU run previously returned four rows of metrics and nothing else,
so no post-hoc seed ensembling or calibration analysis was possible from the
returned artefacts. Every output file is now cell-tagged, removing the manual
'move each CSV aside before the next run' step RUNLOG.md currently mandates.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CP3LLMEoVNhJx8zUvHBCkm"
```

---

### Task 9: The resumable grid driver

**Files:**
- Create: `scripts/run_stage2b_grid.py`
- Test: `tests/test_finetune_grid.py` (append)

**Interfaces:**
- Consumes: `src.finetune_grid.cells_for`, `GridCell.slug()`; `scripts/finetune_esm_mmr.py`'s CLI.
- Produces: `scripts/run_stage2b_grid.py::cell_is_complete(out_dir, cell, mode, eval_mode) -> bool`, `::build_cell_argv(args, cell) -> List[str]`, `::aggregate(out_dir) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_finetune_grid.py`:

```python
import json


def load_grid_script():
    spec = importlib.util.spec_from_file_location(
        "run_stage2b_grid", PROJECT_ROOT / "scripts" / "run_stage2b_grid.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cell_is_incomplete_before_anything_runs(tmp_path):
    mod = load_grid_script()
    cell = GridCell("esm+priors", -1, "residual", 42)
    assert not mod.cell_is_complete(tmp_path, cell, "siamese", "lopo")


def test_cell_is_complete_only_with_results_and_predictions(tmp_path):
    """Resumability must not skip a cell whose run died between writing the
    summary and writing the predictions."""
    mod = load_grid_script()
    cell = GridCell("esm+priors", -1, "residual", 42)
    tag = f"siamese_lopo_{cell.slug()}"
    (tmp_path / f"esm_finetune_summary_{tag}.json").write_text(
        json.dumps({"cell": cell.slug()}))
    assert not mod.cell_is_complete(tmp_path, cell, "siamese", "lopo")
    (tmp_path / f"esm_finetune_results_{tag}.csv").write_text("holdout_gene\nMLH1\n")
    assert not mod.cell_is_complete(tmp_path, cell, "siamese", "lopo")
    (tmp_path / f"esm_finetune_predictions_{tag}.csv").write_text("label,prob\n1,0.9\n")
    assert mod.cell_is_complete(tmp_path, cell, "siamese", "lopo")


def test_build_cell_argv_carries_every_axis():
    mod = load_grid_script()
    args = mod.parse_args(["--tiers", "1", "--esm_model", "facebook/esm2_t6_8M_UR50D"])
    cell = GridCell("esm+priors", 0, "off", 43, fusion="gatewave")
    argv = mod.build_cell_argv(args, cell)
    assert "--branch" in argv and argv[argv.index("--branch") + 1] == "esm+priors"
    assert argv[argv.index("--n_unfrozen_layers") + 1] == "0"
    assert argv[argv.index("--seed") + 1] == "43"
    assert argv[argv.index("--fusion") + 1] == "gatewave"
    assert argv[argv.index("--cell_slug") + 1] == cell.slug()
    assert "--no-use_pllr" in argv, "pllr_mode 'off' must disable the term"


def test_build_cell_argv_uses_pllr_mode_when_the_term_is_on():
    mod = load_grid_script()
    args = mod.parse_args(["--tiers", "1"])
    argv = mod.build_cell_argv(args, GridCell("esm", -1, "residual", 42))
    assert "--no-use_pllr" not in argv
    assert argv[argv.index("--pllr_mode") + 1] == "residual"


def test_aggregate_merges_every_cell_result(tmp_path):
    mod = load_grid_script()
    for slug, auc in [("a", 0.9), ("b", 0.8)]:
        (tmp_path / f"esm_finetune_results_siamese_lopo_{slug}.csv").write_text(
            f"holdout_gene,roc_auc,cell_slug\nMLH1,{auc},{slug}\n")
    df = mod.aggregate(tmp_path)
    assert len(df) == 2
    assert set(df["cell_slug"]) == {"a", "b"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_finetune_grid.py -k "cell_is or cell_argv or aggregate" -v`
Expected: FAIL — `FileNotFoundError: scripts/run_stage2b_grid.py`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/run_stage2b_grid.py`:

```python
#!/usr/bin/env python3
"""Run the Stage-2b ablation grid, resumably, one cell at a time.

Answers the question docs/PAPER_DRAFT.md 6.12 poses but cannot currently
settle: that section varies freeze depth alone and compares the result to a
frozen *priors* probe, while Stage 2b reads no prior features -- so the gap it
measures confounds freeze depth with feature set. The ``branch`` axis here
holds the feature set constant.

Tiers are ordered by scientific value, so an interrupted weekend degrades
gracefully; tier 1 alone settles the headline. Completed cells are skipped on
restart, which matters because a multi-day run on a shared desktop will be
interrupted.

Example
-------
    python scripts/run_stage2b_grid.py --tiers 1 2 3 4 5 \
        --esm_model facebook/esm2_t33_650M_UR50D --mode siamese --eval lopo \
        --batch_size 1 --grad_accum 8 --gradient_checkpointing
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.finetune_grid import TIERS, GridCell, cells_for  # noqa: E402

logger = logging.getLogger("run_stage2b_grid")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--tiers", nargs="+", default=["1"], choices=sorted(TIERS))
    p.add_argument("--out_dir", type=Path, default=ROOT / "data/processed/stage2b_grid")
    p.add_argument("--mode", choices=("wt_site", "siamese"), default="siamese")
    p.add_argument("--eval", dest="eval_mode", choices=("lopo", "holdout"), default="lopo")
    p.add_argument("--esm_model", type=str, default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--n_bootstrap", type=int, default=10_000)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--mmr_csv", type=Path, default=None)
    p.add_argument("--panel_json", type=Path, default=None)
    p.add_argument("--af_labels_active", action="store_true")
    p.add_argument("--dry_run", action="store_true",
                   help="Print the plan and the per-cell command lines, run nothing.")
    p.add_argument("--force", action="store_true",
                   help="Re-run cells that already have complete artefacts.")
    return p.parse_args(argv)


def cell_tag(cell: GridCell, mode: str, eval_mode: str) -> str:
    return f"{mode}_{eval_mode}_{cell.slug()}"


def cell_is_complete(out_dir: Path, cell: GridCell, mode: str, eval_mode: str) -> bool:
    """True only when every artefact of this cell exists.

    All three are required: a run that died between writing the summary and
    writing the predictions must re-run, not be silently skipped.
    """
    tag = cell_tag(cell, mode, eval_mode)
    return all((Path(out_dir) / f"esm_finetune_{kind}_{tag}.{ext}").exists()
               for kind, ext in (("summary", "json"), ("results", "csv"),
                                 ("predictions", "csv")))


def build_cell_argv(args: argparse.Namespace, cell: GridCell) -> list:
    argv = [
        sys.executable, str(ROOT / "scripts" / "finetune_esm_mmr.py"),
        "--mode", args.mode,
        "--eval", args.eval_mode,
        "--esm_model", args.esm_model,
        "--branch", cell.branch,
        "--fusion", cell.fusion,
        "--n_unfrozen_layers", str(cell.n_unfrozen_layers),
        "--seed", str(cell.seed),
        "--cell_slug", cell.slug(),
        "--out_dir", str(args.out_dir),
        "--batch_size", str(args.batch_size),
        "--grad_accum", str(args.grad_accum),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--n_bootstrap", str(args.n_bootstrap),
        "--no-save_checkpoints",
    ]
    if cell.pllr_mode == "off":
        argv.append("--no-use_pllr")
    else:
        argv += ["--pllr_mode", cell.pllr_mode]
    if args.gradient_checkpointing:
        argv.append("--gradient_checkpointing")
    if args.af_labels_active:
        argv.append("--af_labels_active")
    if args.mmr_csv:
        argv += ["--mmr_csv", str(args.mmr_csv)]
    if args.panel_json:
        argv += ["--panel_json", str(args.panel_json)]
    return argv


def aggregate(out_dir: Path) -> pd.DataFrame:
    """Every cell's results CSV, concatenated."""
    frames = [pd.read_csv(p) for p in sorted(Path(out_dir).glob(
        "esm_finetune_results_*.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cells = cells_for(args.tiers)

    pending = [c for c in cells
               if args.force or not cell_is_complete(args.out_dir, c, args.mode,
                                                     args.eval_mode)]
    logger.info("Grid: %d cells in tiers %s | %d already complete | %d to run",
                len(cells), ",".join(args.tiers), len(cells) - len(pending),
                len(pending))
    for c in pending:
        logger.info("  [tier %s] %s", c.tier, c.slug())
    if args.dry_run:
        for c in pending:
            print(" ".join(build_cell_argv(args, c)))
        return 0

    t0 = time.time()
    completed, failed = [], []
    for i, cell in enumerate(pending, start=1):
        logger.info("=== cell %d/%d: %s (tier %s) ===", i, len(pending),
                    cell.slug(), cell.tier)
        cell_t0 = time.time()
        result = subprocess.run(build_cell_argv(args, cell), cwd=str(ROOT))
        elapsed = (time.time() - cell_t0) / 60
        if result.returncode == 0:
            completed.append(cell.slug())
            logger.info("cell done in %.1f min | %.1f h elapsed of grid",
                        elapsed, (time.time() - t0) / 3600)
        else:
            # Keep going: one cell's OOM should not cost the remaining tiers.
            failed.append(cell.slug())
            logger.error("cell FAILED (exit %d) after %.1f min -- continuing",
                         result.returncode, elapsed)

    combined = aggregate(args.out_dir)
    combined_path = args.out_dir / "stage2b_grid_results.csv"
    combined.to_csv(combined_path, index=False)
    (args.out_dir / "stage2b_grid_manifest.json").write_text(json.dumps({
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "tiers": args.tiers,
        "cells": [c.to_dict() for c in cells],
        "completed": completed, "failed": failed,
        "esm_model": args.esm_model, "mode": args.mode, "eval": args.eval_mode,
        "runtime_h": round((time.time() - t0) / 3600, 2),
    }, indent=2))

    print(f"\nGrid complete: {len(completed)} ok, {len(failed)} failed, "
          f"{(time.time() - t0) / 3600:.1f} h")
    if failed:
        print("FAILED cells (re-run the same command to retry them):")
        for slug in failed:
            print(f"  {slug}")
    print(f"Combined results -> {combined_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_finetune_grid.py -v && .venv/bin/python -m pytest tests/ -q`
Expected: PASS, 140 tests total

- [ ] **Step 5: Commit**

```bash
git add scripts/run_stage2b_grid.py tests/test_finetune_grid.py
git commit -m "feat: resumable Stage-2b grid driver

Runs cells sequentially, skips complete ones on restart, and keeps going past
a failed cell so one OOM does not cost the remaining tiers.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CP3LLMEoVNhJx8zUvHBCkm"
```

---

### Task 10: End-to-end CPU smoke run and the GPU playbook

The last gate before the code leaves for the GPU box: prove the whole path runs on the 8M checkpoint, then write the operator's document.

**Files:**
- Create: `docs/GPU_RUN_PLAYBOOK.md`
- Modify: `docs/RUNLOG.md`

**Interfaces:**
- Consumes: everything above.
- Produces: no code.

- [ ] **Step 1: Run the grid end-to-end on CPU with the tiny checkpoint**

```bash
.venv/bin/python scripts/run_stage2b_grid.py \
  --tiers 3 \
  --esm_model facebook/esm2_t6_8M_UR50D \
  --mode siamese --eval holdout \
  --mmr_csv data/mmr/processed/extended/extended_dataset.csv \
  --panel_json data/mmr/processed/extended/panel_sequences.json \
  --out_dir /tmp/claude-1000/-home-jvikramsrd-Projects-variant-pathogenicity-dl/6ef835e7-96c7-444e-951b-a5e387bd9fe3/scratchpad/grid_smoke \
  --epochs 2 --n_bootstrap 200
```

Expected: every tier-3 cell completes; the out_dir holds a `results`, `predictions` and `summary` file per cell plus `stage2b_grid_results.csv` and `stage2b_grid_manifest.json`.

This is an 8M-parameter smoke test for plumbing, **not** a result to compare against 650M runs.

- [ ] **Step 2: Verify resumability**

Re-run the exact same command.
Expected: log line `N already complete | 0 to run`, and no cell re-executes.

- [ ] **Step 3: Verify the predictions are usable for seed ensembling**

```bash
.venv/bin/python -c "
import pandas as pd, glob
fs = glob.glob('/tmp/claude-1000/-home-jvikramsrd-Projects-variant-pathogenicity-dl/6ef835e7-96c7-444e-951b-a5e387bd9fe3/scratchpad/grid_smoke/esm_finetune_predictions_*.csv')
df = pd.concat([pd.read_csv(f) for f in fs])
print(df.groupby('cell_slug').size())
assert {'label','prob','cell_slug','seed'} <= set(df.columns)
print('OK: predictions carry labels, probabilities and cell identity')
"
```

Expected: one row group per cell, assertion passes.

- [ ] **Step 4: Write `docs/GPU_RUN_PLAYBOOK.md`**

Contents, in this order: the one-time setup on the CUDA box (`requirements-cuda.txt`, `torch.cuda.is_available()` check, dataset build command with `--pms2_codon_range 382 862`); the per-tier commands with the `--batch_size 1 --grad_accum 8 --gradient_checkpointing` flags that make a 650M siamese fine-tune fit 15 GiB; the expected wall-clock per tier from spec §5.1 (tier 1 ≈ 15.75 h, 2 ≈ 5.25 h, 3 ≈ 1 h, 4 ≈ 4 h, 5 ≈ 5 h); the resume instruction (re-run the identical command); and an explicit "bring these files back" list — `data/processed/stage2b_grid/` in full, being the per-cell results, predictions and summaries plus the combined CSV and manifest.

- [ ] **Step 5: Append a `RUNLOG.md` entry and commit**

Add an entry at the top of `docs/RUNLOG.md` (newest first, per its existing format) recording the CPU smoke run: date, command, that it was an 8M plumbing check and not a result, and that the grid driver and playbook are ready for the CUDA box.

```bash
git add docs/GPU_RUN_PLAYBOOK.md docs/RUNLOG.md
git commit -m "docs: GPU run playbook and CPU smoke-run log entry

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CP3LLMEoVNhJx8zUvHBCkm"
```

---

## Self-Review

**Spec coverage (sub-projects 1 and 2):**

| Spec section | Task |
|---|---|
| §4.1 priors branch | 4 (+1 for the LayerNorm prerequisite) |
| §4.1 gene-constant drop, train-only impute/scaling, persisted | 8 |
| §4.2 PLLR residual | 3 |
| §4.3 warmup + cosine | 5 |
| §4.4 `pos_weight` | 5 |
| §4.5 frozen precompute | 6 |
| §4.6 predictions + cell tags | 8 |
| §4.7 CPU verification | 2, 3, 4, 5, 6 tests; 10 end-to-end |
| §3.1 AF quarantine | 7 (guard + test), 8 (enforcement) |
| §5 grid driver, resumability, manifest | 7 (cells/tiers), 9 (driver) |

Not covered here, by design: spec §3.2 (`prepare_split` train/eval label policies), §6 (CPU experiments), §7 (data expansion), §8 (paper/docs and `ingest_gpu_results.py`). Those are separate plans — §3.2 and §7 belong together, and §8 needs results in hand first.

**Type consistency check:** `pllr_mode` is the model kwarg and the CLI value throughout, with `use_pllr` retained only as the legacy bool. `n_prior_features` is the constructor arg; `prior_matrix` is the `build_examples` kwarg; `priors` is the batch key, the `classify` parameter and the `forward` parameter. `GridCell.slug()` is the single source of cell identity in Tasks 7, 8 and 9. `encode_all` returns the 3-tuple that `_fit_precomputed` destructures.

**Known deviation:** Task 4 sets `self.head = None` / `self.out = None` in the fusion path, so `_scoring_module()` exists to hide the branch from the residual zero-init. Task 3's `test_residual_output_layer_trains_off_zero` touches `model.out` directly and therefore only runs in the no-priors configuration — which is how it is written.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-02-stage2b-fair-comparison.md`.
