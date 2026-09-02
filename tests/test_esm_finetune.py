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
