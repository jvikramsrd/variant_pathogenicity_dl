"""The paper's statistics (paper/analysis.py) against the project's reference implementations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "paper"))

import analysis as A  # noqa: E402
from vpdl.evaluate import _mcc, roc_auc  # noqa: E402


def _gene(seed: int, n: int = 120, seeds: int = 3, ties: bool = False) -> A.GeneData:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.6).astype(int)
    scores = y[:, None] * 0.8 + rng.normal(size=(n, seeds))
    if ties:
        scores = np.round(scores, 1)
    return A.GeneData(y, scores, np.full(seeds, 0.4))


@pytest.mark.parametrize("ties", [False, True])
def test_weighted_auc_equals_rank_auc_on_the_resample(ties):
    gene = _gene(1, ties=ties)
    rng = np.random.default_rng(7)
    w = A._weights(rng, len(gene.y), 5)
    fast = A.auc_given_weights(gene, w)
    for row, weights in enumerate(w):
        pick = np.repeat(np.arange(len(gene.y)), weights.astype(int))
        slow = np.mean([roc_auc(gene.y[pick], gene.scores[pick, k]) for k in range(gene.scores.shape[1])])
        assert fast[row] == pytest.approx(slow, abs=1e-12)


def test_weighted_mcc_equals_reference_mcc_on_the_resample():
    gene = _gene(2)
    w = A._weights(np.random.default_rng(3), len(gene.y), 4)
    fast = A.mcc_given_weights(gene, w)
    for row, weights in enumerate(w):
        pick = np.repeat(np.arange(len(gene.y)), weights.astype(int))
        slow = np.mean([_mcc(gene.y[pick], (gene.scores[pick, k] >= 0.4).astype(int))
                        for k in range(gene.scores.shape[1])])
        assert fast[row] == pytest.approx(slow, abs=1e-12)


def test_paired_difference_of_identical_arms_is_exactly_zero():
    cond = {"G1": _gene(4), "G2": _gene(5)}
    result = A.bootstrap([cond, cond], ["G1", "G2"], "auc", n_boot=200)
    assert result["point"] == result["low"] == result["high"] == 0.0


def test_compare_drops_genes_where_both_arms_predict_identically():
    shared = _gene(6)
    a = {"G1": _gene(7), "G2": shared}
    b = {"G1": _gene(8), "G2": shared}
    result = A.compare(a, b, ["G1", "G2"], "auc")
    assert result["genes"] == ["G1"] and result["identical_genes_dropped"] == ["G2"]


def test_point_estimate_is_the_mean_of_per_gene_seed_means():
    cond = {"G1": _gene(9), "G2": _gene(10)}
    expected = np.mean([np.mean([roc_auc(g.y, g.scores[:, k]) for k in range(3)]) for g in cond.values()])
    assert A.bootstrap([cond], ["G1", "G2"], "auc", n_boot=50)["point"] == pytest.approx(expected)


def test_exact_scores_recovers_float32_values_from_their_csv_spelling():
    original = np.float32([0.93325883, 0.1, 0.3654307723045349, 1e-7])
    text = [np.format_float_positional(v, unique=True) for v in original]
    parsed = np.array([float(t) for t in text])
    assert np.array_equal(A.exact_scores(parsed), original.astype(np.float64))
    # float64 values that are not float32 spellings are left alone
    wide = np.array([0.1234567890123, 0.5])
    assert np.array_equal(A.exact_scores(wide), wide)


def test_macro_names_are_letters_only_and_stable():
    assert A.word("A5") == "AFive" and A.word("bilstm_attn") == "BilstmAttn"
    assert A.word("MLH1") == "MLHOne" and A.word("neg_log_af") == "NegLogAf"
    with pytest.raises(ValueError):
        A.macro("bad1", 1)
