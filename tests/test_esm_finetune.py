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
    """Build a tiny model, skipping only if the checkpoint cannot be resolved.

    The guard is deliberately narrow: HuggingFace raises OSError when a
    checkpoint is missing from the local cache and no network is available,
    while a TypeError or ValueError means *this repository's* code is wrong.
    Catching Exception here would turn every constructor-signature bug into a
    green skip.
    """
    pytest.importorskip("transformers")
    try:
        return ESMFineTuneClassifier(model_name=TINY, **kwargs)
    except OSError as exc:  # network-free guard
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


# --------------------------------------------------------------------------- #
# PLLR residual base
# --------------------------------------------------------------------------- #
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
    assert float(model.pllr_gain.detach()) == pytest.approx(-1.0)
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
