"""Synthetic-data tests for src.inference. No real checkpoint, dataset, or
network call is used anywhere in this file -- every model is a hand-built
`torch.nn.Module` stub and every input frame is a few in-memory rows.

Run with:  python -m pytest tests/test_inference.py -v
(not run in this environment -- see docs/INFERENCE.md)
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

import src.inference as inference  # noqa: E402
from src.gene_aliases import CANONICAL_GENE_IDS, UnknownGeneError  # noqa: E402
from src.inference import (  # noqa: E402
    InferenceInputError,
    LoadedInferenceModel,
    UnsupportedArchitectureError,
    align_genes,
    load_inference_model,
    predict,
    validate_input_frame,
)
from src.transfer import TransferHeadFormatError  # noqa: E402

_SYMBOL_A, _ACCESSION_A = next(iter(CANONICAL_GENE_IDS.items()))
_OTHER = [(s, a) for s, a in CANONICAL_GENE_IDS.items() if s != _SYMBOL_A]
_SYMBOL_B, _ACCESSION_B = _OTHER[0]

FEATURE_COLS = ("p0", "p1")


class _TinyLinear(torch.nn.Module):
    """Deterministic stand-in model: logit = x . w + b. No training, no
    randomness -- exists only so predict_logits has a real nn.Module to call.
    """

    def __init__(self, w=(1.0, -1.0), b=0.0):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(w, dtype=torch.float32), requires_grad=False)
        self.b = torch.nn.Parameter(torch.tensor(float(b)), requires_grad=False)

    def forward(self, x):
        return (x @ self.w + self.b)


def _fake_loaded(*, threshold=0.5, device=None) -> LoadedInferenceModel:
    device = device or torch.device("cpu")
    return LoadedInferenceModel(
        model=_TinyLinear().to(device),
        scale_views=lambda mats: [np.asarray(m, dtype=np.float32) for m in mats],
        feature_columns=FEATURE_COLS,
        config={"arch": "priors"},
        threshold=threshold,
        checkpoint_path="fake.pt",
        device=device,
    )


def _frame(genes=(_SYMBOL_A, _SYMBOL_A), positions=(1, 2), p0=(0.1, 0.2), p1=(0.0, 0.0)):
    return pd.DataFrame({
        "uniprot_id": list(genes),
        "position": list(positions),
        "wt_aa": ["A"] * len(genes),
        "mut_aa": ["G"] * len(genes),
        "p0": list(p0),
        "p1": list(p1),
    })


# --------------------------------------------------------------------------- #
# validate_input_frame
# --------------------------------------------------------------------------- #
def test_validate_input_frame_rejects_missing_identifying_columns():
    loaded = _fake_loaded()
    df = _frame().drop(columns=["wt_aa"])
    with pytest.raises(InferenceInputError, match="wt_aa"):
        validate_input_frame(df, loaded)


def test_validate_input_frame_rejects_missing_feature_columns_by_name():
    loaded = _fake_loaded()
    df = _frame().drop(columns=["p1"])
    with pytest.raises(InferenceInputError, match="p1"):
        validate_input_frame(df, loaded)


def test_validate_input_frame_rejects_empty_input():
    loaded = _fake_loaded()
    with pytest.raises(InferenceInputError, match="zero rows"):
        validate_input_frame(_frame().iloc[0:0], loaded)


def test_validate_input_frame_reorders_columns_regardless_of_input_order():
    """A caller whose CSV columns are in a different order than the
    checkpoint's schema must not be rejected, and must get the checkpoint's
    own column order back."""
    loaded = _fake_loaded()
    df = _frame()[["p1", "mut_aa", "p0", "position", "wt_aa", "uniprot_id"]]
    out = validate_input_frame(df, loaded)
    assert list(out.columns) == ["uniprot_id", "position", "wt_aa", "mut_aa", "p0", "p1"]
    assert out["p0"].tolist() == [0.1, 0.2]


# --------------------------------------------------------------------------- #
# align_genes
# --------------------------------------------------------------------------- #
def test_align_genes_resolves_symbol_and_accession_to_the_same_canonical_id():
    loaded = _fake_loaded()
    by_symbol = validate_input_frame(_frame(genes=(_SYMBOL_A, _SYMBOL_A)), loaded)
    by_accession = validate_input_frame(_frame(genes=(_ACCESSION_A, _ACCESSION_A)), loaded)
    assert align_genes(by_symbol)["uniprot_id"].tolist() == [_ACCESSION_A, _ACCESSION_A]
    assert align_genes(by_accession)["uniprot_id"].tolist() == [_ACCESSION_A, _ACCESSION_A]


def test_align_genes_rejects_an_unknown_gene():
    loaded = _fake_loaded()
    df = validate_input_frame(_frame(genes=("NOT_A_REAL_GENE", _SYMBOL_A)), loaded)
    with pytest.raises(UnknownGeneError):
        align_genes(df)


def test_align_genes_rejects_a_gene_outside_the_known_set():
    """An entirely unseen gene (resolvable, but not one the model was
    trained on) must be a named, explicit rejection, not a silent score."""
    loaded = _fake_loaded()
    df = validate_input_frame(_frame(genes=(_SYMBOL_B, _SYMBOL_B)), loaded)
    with pytest.raises(InferenceInputError, match=_SYMBOL_B):
        align_genes(df, known_genes=[_SYMBOL_A])


def test_align_genes_allows_a_gene_outside_known_set_when_none_given():
    loaded = _fake_loaded()
    df = validate_input_frame(_frame(genes=(_SYMBOL_B, _SYMBOL_B)), loaded)
    align_genes(df, known_genes=None)  # must not raise


# --------------------------------------------------------------------------- #
# load_inference_model
# --------------------------------------------------------------------------- #
def test_load_inference_model_rejects_an_unsupported_architecture(monkeypatch):
    def fake_load_transfer_head(path, device=None):
        return _TinyLinear(), (lambda mats: mats), {
            "config": {"arch": "concat"}, "feature_columns": list(FEATURE_COLS)}

    monkeypatch.setattr(inference, "load_transfer_head", fake_load_transfer_head)
    with pytest.raises(UnsupportedArchitectureError, match="concat"):
        load_inference_model(Path("fake.pt"))


def test_load_inference_model_propagates_a_format_error(monkeypatch):
    def fake_load_transfer_head(path, device=None):
        raise TransferHeadFormatError("fake.pt: bad format")

    monkeypatch.setattr(inference, "load_transfer_head", fake_load_transfer_head)
    with pytest.raises(TransferHeadFormatError):
        load_inference_model(Path("fake.pt"))


def test_load_inference_model_falls_back_to_cpu_when_no_cuda(monkeypatch):
    def fake_load_transfer_head(path, device=None):
        assert device == torch.device("cpu")
        return _TinyLinear(), (lambda mats: mats), {
            "config": {"arch": "priors"}, "feature_columns": list(FEATURE_COLS)}

    monkeypatch.setattr(inference, "load_transfer_head", fake_load_transfer_head)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    loaded = load_inference_model(Path("fake.pt"), device=None)
    assert loaded.device == torch.device("cpu")


# --------------------------------------------------------------------------- #
# predict
# --------------------------------------------------------------------------- #
def test_predict_rejects_nan_in_a_required_feature_column():
    loaded = _fake_loaded()
    df = _frame(p0=(0.1, float("nan")))
    with pytest.raises(InferenceInputError, match="NaN"):
        predict(loaded, df)


def test_predict_produces_the_documented_output_schema():
    loaded = _fake_loaded(threshold=0.5)
    result = predict(loaded, _frame())
    cols = set(result.predictions.columns)
    assert {"uniprot_id", "position", "wt_aa", "mut_aa",
            "probability", "predicted_label", "threshold_used", "checkpoint"} <= cols
    assert result.checkpoint_path == "fake.pt"
    assert result.threshold == 0.5


def test_predict_single_row_matches_the_same_row_scored_in_a_batch():
    """No separate single-item code path exists; a row's prediction must not
    depend on how many other rows accompany it."""
    loaded = _fake_loaded()
    batch = predict(loaded, _frame(genes=(_SYMBOL_A, _SYMBOL_A), p0=(0.1, 0.9),
                                   p1=(0.0, 0.0)))
    single = predict(loaded, _frame(genes=(_SYMBOL_A,), positions=(1,),
                                    p0=(0.1,), p1=(0.0,)))
    assert single.predictions["probability"].iloc[0] == pytest.approx(
        batch.predictions["probability"].iloc[0])


def test_predict_uses_an_explicit_threshold_override():
    loaded = _fake_loaded(threshold=0.5)
    result = predict(loaded, _frame(), threshold=0.9)
    assert result.threshold == 0.9
    assert (result.predictions["threshold_used"] == 0.9).all()


def test_predict_runs_on_a_cpu_only_device():
    """CPU-only fallback: a LoadedInferenceModel built with device=cpu must
    score without touching CUDA at all."""
    loaded = _fake_loaded(device=torch.device("cpu"))
    result = predict(loaded, _frame())
    assert len(result.predictions) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
