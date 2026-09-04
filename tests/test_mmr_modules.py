"""Offline unit tests for the MMR project-plan modules.

Covers (all network-free):
* src/gnomad.py        — HGVSp parsing, frequency flags, joins
* src/mmr_dataset.py   — PMS2 gate, evidence tiers, balanced subsets, LOPO
* src/eval_utils.py    — bootstrap CIs, MCC-optimal threshold tuning
* src/fusion.py        — head shapes incl. GateWave ablation switches
* src/mvmamba_features.py — centred-window bounds + pooling math
* src/transfer.py      — prior matrices, row alignment, stage row selection

Run:  python tests/test_mmr_modules.py   (or pytest tests/test_mmr_modules.py)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval_utils import bootstrap_ci, optimal_threshold_by_mcc, safe_mcc  # noqa: E402
from src.fusion import (  # noqa: E402
    BranchHead,
    ConcatFusionHead,
    GateWaveFusionHead,
    build_fusion_head,
)
from src.gnomad import (  # noqa: E402
    add_frequency_flags,
    join_gnomad_features,
    parse_hgvsp,
    validate_against_sequence,
)
from src.mmr_dataset import (  # noqa: E402
    MMR_UNIPROT,
    PMS2_PSEUDOGENE_CODON_RANGE,
    add_evidence_tiers,
    apply_pms2_homology_gate,
    make_balanced_subset,
    variant_key,
    write_leave_one_gene_out_manifest,
    load_leave_one_gene_out_manifest,
)
from src.mvmamba_features import centered_window_bounds  # noqa: E402
from src.transfer import (  # noqa: E402
    ABLATABLE_PRIOR_GROUPS,
    GENE_CONSTANT_PRIOR_COLS,
    PRIOR_FEATURE_GROUPS,
    TRANSFER_PRIOR_COLS,
    drop_prior_groups,
    prior_columns_of,
    add_within_gene_rank_features,
    FeatureBundle,
    align_rows,
    prior_impute_values,
    prior_matrix,
    select_stage_rows,
)


# --------------------------------------------------------------------------- #
# gnomAD parsing / flags / join
# --------------------------------------------------------------------------- #
def test_parse_hgvsp():
    assert parse_hgvsp("p.Ser2Ala") == ("S", 2, "A")
    assert parse_hgvsp("p.(Arg273His)") == ("R", 273, "H")
    assert parse_hgvsp("p.Arg273HisfsTer12") is None     # frameshift
    assert parse_hgvsp("p.Arg273Ter") is None            # stop codon
    assert parse_hgvsp("") is None


def _gnomad_frame():
    return pd.DataFrame({
        "gene": ["MLH1"] * 4,
        "position": [2, 10, 20, 30],
        "wt_aa": list("MAGT"),
        "mut_aa": ["L", "V", "E", "R"],
        "hgvs_p": ["p.Met2Leu", "p.Ala10Val", "p.Gly20Glu", "p.Thr30Arg"],
        "gnomad_af_joint": [0.07, 1e-6, np.nan, 5e-4],
        "gnomad_af_genome": [0.06, 0.0, np.nan, 1e-4],
        "gnomad_af_exome": [0.05, 1e-6, np.nan, 4e-4],
        "gnomad_ac_joint": [1000, 1, 0, 7],
        "gnomad_an_joint": [10000, 10000, 10000, 10000],
    })


def test_frequency_flags_acmg_thresholds():
    df = add_frequency_flags(_gnomad_frame())
    # BA1 (>5%): only position 2.
    assert df["acmg_ba1"].tolist() == [1, 0, 0, 0]
    # BS1 (>1e-3): only position 2 (position 30 AF=5e-4 is below threshold).
    assert df["acmg_bs1"].tolist() == [1, 0, 0, 0]
    # PM2 (absent or <1e-5): positions 10 (rare) and 20 (absent).
    assert df["acmg_pm2"].tolist() == [0, 1, 1, 0]
    # Absent AF maps to the log-floor, below every observed value.
    log_af = df["gnomad_log10_af"]
    assert log_af.iloc[2] == -9.0
    assert (log_af.drop(index=[2]) > -9.0).all()


def test_join_gnomad_features_alignment():
    from src.gnomad import add_frequency_flags as _flags
    target = pd.DataFrame({
        "gene": ["MLH1", "MLH1", "MSH2"],
        "position": [2, 999, 5],
        "wt_aa": ["M", "G", "A"],
        "mut_aa": ["L", "C", "V"],
        "score": [0.1, 0.2, 0.3],
    })
    merged = join_gnomad_features(target, _flags(_gnomad_frame()))
    assert len(merged) == len(target)                      # left join preserved
    assert merged.loc[0, "acmg_ba1"] == 1                  # matched row flagged
    assert merged.loc[1, "acmg_pm2"] == 1                  # unmatched -> PM2-consistent
    assert pd.isna(merged.loc[1, "gnomad_af_joint"])
    assert merged.loc[2, "acmg_ba1"] == 0                  # other gene unaffected


def test_validate_against_sequence_drops_isoform_mismatch():
    # 30-aa reference agreeing with the fixture at every fixture position
    # (2=M, 10=A, 20=G, 30=T); only the isoform-mismatch decoy must drop.
    ref = ["A"] * 30
    ref[1], ref[19], ref[29] = "M", "G", "T"
    seq = "".join(ref)
    df = _gnomad_frame()
    bad = pd.DataFrame([{"gene": "MLH1", "position": 3, "wt_aa": "W",
                         "mut_aa": "R", "hgvs_p": "p.Trp3Arg",
                         "gnomad_af_joint": 1e-5}])
    out = validate_against_sequence(pd.concat([df, bad], ignore_index=True), seq)
    assert out["hgvs_p"].tolist() == ["p.Met2Leu", "p.Ala10Val",
                                      "p.Gly20Glu", "p.Thr30Arg"]
    assert not (out["wt_aa"] == "W").any()


# --------------------------------------------------------------------------- #
# mmr_dataset: PMS2 gate / tiers / balance / manifest
# --------------------------------------------------------------------------- #
def _mmr_master():
    rows = []
    for gene, n in (("PMS2", 6), ("MSH2", 4)):
        for i in range(n):
            rows.append({
                "gene": gene,
                "position": 400 + i,
                "wt_aa": "A",
                "mut_aa": "V" if i % 2 else "T",
                "label": 1 if i % 2 else 0,
                "review_status": ("reviewed by expert panel"
                                  if i % 3 == 0 else
                                  "criteria provided, single submitter"),
                "_stars": 3 if i % 3 == 0 else 1,
                "label_weight": 1.0,
            })
    return pd.DataFrame(rows)


def test_variant_key_dtype_stability():
    k1 = variant_key("PMS2", np.int64(500), "A", "V")
    k2 = variant_key("pms2", 500.0, "A", "V")
    assert k1 == k2 == "PMS2|500|A|V"


def test_pms2_gate_fail_closed():
    with tempfile.TemporaryDirectory() as td:
        homology_csv = Path(td) / "conf.csv"
        pd.DataFrame([{
            "gene": "PMS2", "position": 400, "wt_aa": "A", "mut_aa": "T",
            "orthogonally_confirmed": False}]).to_csv(homology_csv, index=False)
        try:
            apply_pms2_homology_gate(_mmr_master(), homology_csv=homology_csv)
            raised = False
        except ValueError:
            raised = True
        assert raised, "gate must refuse a confirmation table without any confirmed row"

    try:
        apply_pms2_homology_gate(_mmr_master())
        raised = False
    except ValueError:
        raised = True
    assert raised, "gate must be fail-closed with no inputs at all"


def test_pms2_gate_withholds_unconfirmed_labels():
    master = _mmr_master()
    gated = apply_pms2_homology_gate(
        master, codon_range=(398, 406))
    pms2 = gated[gated["gene"] == "PMS2"]
    assert (pms2["label"].isna()).all()
    assert (gated[gated["gene"] == "MSH2"]["label"].notna()).all()
    assert gated["pms2_homology_excluded"].sum() == 6


def test_pms2_gate_confirmed_rows_keep_labels():
    master = _mmr_master()
    with tempfile.TemporaryDirectory() as td:
        homology_csv = Path(td) / "conf.csv"
        keys = master[(master["gene"] == "PMS2") & (master["label"] == 0)] \
            .head(2)[["gene", "position", "wt_aa", "mut_aa"]]
        conf = keys.assign(orthogonally_confirmed=True)
        # one decoy unconfirmed row must not break the gate
        conf = pd.concat([conf, keys.head(1).assign(
            orthogonally_confirmed=False)], ignore_index=True)
        conf.to_csv(homology_csv, index=False)
        gated = apply_pms2_homology_gate(master, homology_csv=homology_csv)
    pms2 = gated[gated["gene"] == "PMS2"]
    kept = pms2[pms2["label"].notna()]
    assert len(kept) == 2                       # exactly the confirmed rows
    assert set(kept["label"]) == {0}


def test_pms2_pseudogene_range_matches_pinned_reference():
    """The published exon 11-15 span must stay inside the pinned protein.

    ``PMS2_PSEUDOGENE_CODON_RANGE`` is derived from the Ensembl exon table by
    ``scripts/derive_pms2_homology_range.py``, but it is checked in as a
    literal so the offline gate does not need network access. This pins the
    two invariants that make the literal safe to reuse: it ends exactly at
    the canonical protein length recorded for P54278, and it leaves a
    non-empty N-terminal region for variants that are usable without
    orthogonal confirmation. A change to MMR_UNIPROT that silently
    invalidated the range would otherwise go unnoticed until PMS2 rows were
    already being trained on.
    """
    start, end = PMS2_PSEUDOGENE_CODON_RANGE
    _acc, length = MMR_UNIPROT["PMS2"]
    assert 1 < start < end, "range must be a non-degenerate interval"
    assert end == length, (
        f"exon 15 closes the CDS, so the range must end at PMS2's {length} aa")
    assert start == 382, "exon 11 opens at c.1145 -> codon 382"


def test_pms2_gate_with_derived_range_keeps_n_terminal_labels():
    """Using the derived range must gate the region and nothing else."""
    master = _mmr_master()
    safe = pd.DataFrame([{
        "gene": "PMS2", "position": 100, "wt_aa": "A", "mut_aa": "T",
        "label": 1.0, "label_weight": 1.0}])
    master = pd.concat([master, safe], ignore_index=True)
    gated = apply_pms2_homology_gate(
        master, codon_range=PMS2_PSEUDOGENE_CODON_RANGE)
    pms2 = gated[gated["gene"] == "PMS2"]
    kept = pms2[pms2["label"].notna()]
    assert len(kept) == 1 and int(kept.iloc[0]["position"]) == 100
    assert (pms2[pms2["position"] >= 382]["label"].isna()).all()
    assert (gated[gated["gene"] == "MSH2"]["label"].notna()).all()


def test_evidence_tiers():
    tiered = add_evidence_tiers(_mmr_master())
    expert = tiered[tiered["expert_panel"] == 1]
    assert len(expert) > 0
    assert (expert["tier_weight"] == 1.0).all()
    assert set(tiered["evidence_tier"]) >= {"expert", "moderate"}


def test_balanced_subset_per_gene():
    rows = []
    for gene in ("MLH1", "MSH2"):
        for i in range(30):
            rows.append({"gene": gene, "position": i + 1,
                         "wt_aa": "A", "mut_aa": "V",
                         "label": int(i < 25)})     # 25 pos / 5 neg per gene
    balanced = make_balanced_subset(pd.DataFrame(rows), seed=1)
    counts = balanced.groupby(["gene", "label"]).size().unstack(fill_value=0)
    assert counts.loc["MLH1", 0] == counts.loc["MLH1", 1] == 5
    assert counts.loc["MSH2", 0] == counts.loc["MSH2", 1] == 5


def test_lopo_manifest_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        path = write_leave_one_gene_out_manifest(Path(td))
        splits = load_leave_one_gene_out_manifest(path)
        assert {h for h, _ in splits} == {"MLH1", "MSH2", "MSH6", "PMS2"}
        for holdout, train_genes in splits:
            assert holdout not in train_genes
            assert len(train_genes) == 3


# --------------------------------------------------------------------------- #
# eval_utils
# --------------------------------------------------------------------------- #
def test_bootstrap_ci_recovers_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    score = y * 0.8 + rng.normal(0, 0.35, 400).clip(-0.2, 0.9)
    ci = bootstrap_ci(y, score, metric="roc_auc", n_bootstrap=300, seed=1)
    assert ci["lower"] <= ci["point"] <= ci["upper"]
    assert ci["point"] > 0.75                     # informative signal recovered


def test_bootstrap_ci_stratified_two_class():
    y = np.array([0] * 50 + [1] * 10)
    score = np.linspace(0, 1, 60)
    ci = bootstrap_ci(y, score, metric="roc_auc", n_bootstrap=200, seed=3)
    assert np.isfinite(ci["point"])


def test_optimal_threshold_by_mcc_separates_classes():
    y = np.array([0] * 20 + [1] * 20)
    scores = np.concatenate([np.linspace(0.05, 0.45, 20),
                             np.linspace(0.55, 0.95, 20)])
    thr, mcc = optimal_threshold_by_mcc(y, scores)
    assert 0.45 <= thr <= 0.55
    assert mcc == 1.0


def test_safe_mcc_degenerate_returns_zero():
    assert safe_mcc([1, 1, 1], [1, 0, 1]) == 0.0
    assert safe_mcc([0, 1], [1, 0]) != 0.0


# --------------------------------------------------------------------------- #
# fusion heads
# --------------------------------------------------------------------------- #
def test_branch_and_concat_shapes():
    x1 = torch.randn(16, 64)
    x2 = torch.randn(16, 32)
    b = BranchHead(64, hidden_dim=32)
    assert b(x1).shape == (16,)
    f = ConcatFusionHead((64, 32), shared_dim=24)
    assert f(x1, x2).shape == (16,)
    try:
        f(x1)
        wrong_arity = False
    except ValueError:
        wrong_arity = True
    assert wrong_arity


def test_gatewave_forward_and_ablations():
    x1, x2 = torch.randn(8, 128), torch.randn(8, 48)
    full = GateWaveFusionHead((128, 48), shared_dim=32)
    no_gate = GateWaveFusionHead((128, 48), shared_dim=32, use_gate=False)
    no_glu = GateWaveFusionHead((128, 48), shared_dim=32, use_glu=False)
    for m in (full, no_gate, no_glu):
        assert m(x1, x2).shape == (8,)
    try:
        GateWaveFusionHead((128,))
        raised = False
    except ValueError:
        raised = True
    assert raised                                  # requires exactly two branches
    assert isinstance(build_fusion_head("gatewave", (4, 4)), GateWaveFusionHead)


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


# --------------------------------------------------------------------------- #
# mvmamba windowing math (no model download needed)
# --------------------------------------------------------------------------- #
def test_centered_window_bounds_short_sequence():
    # Sequence shorter than capacity: window covers everything.
    s, e = centered_window_bounds(800, 400)
    assert (s, e) == (0, 800)


def test_centered_window_bounds_long_sequence_centered():
    s, e = centered_window_bounds(1360, 700)
    span = e - s
    assert span <= 1022
    assert s < 700 < e                              # mutation strictly inside


def test_centered_window_bounds_near_termini():
    s, e = centered_window_bounds(1360, 5)          # near N-term -> terminus fallback
    assert (s, e) == (0, min(1022, 1360))
    s, e = centered_window_bounds(1360, 1359)       # near C-term
    assert e == 1360
    assert 1359 >= s                                # mutated residue inside


def test_local_pool_math_on_synthetic_extractor():
    from src.mvmamba_features import MVmambaFeatureExtractor
    ext = MVmambaFeatureExtractor.__new__(MVmambaFeatureExtractor)
    ext.local_window = 3
    ext.include_deltas = True
    hidden = np.arange(40, dtype=np.float32).reshape(10, 4)   # residues x dim
    pooled = ext._pool(hidden)
    assert np.allclose(pooled, hidden.mean(axis=0))
    local = ext._local_pool(hidden, pos0=4, window_span=(0, 10))  # pos 5 +-3
    assert np.allclose(local, hidden[1:8].mean(axis=0))
    # For an encoded sub-span the hidden array starts at `start_enc`; pooling
    # pos0=1 inside span [3,8) must average encoded positions max(3,1-3)..min(8,1+4)
    # i.e. residues 3..4 == hidden[3:5].
    span_view = hidden[3:8]
    clipped = ext._local_pool(span_view, pos0=1, window_span=(3, 8))
    assert np.allclose(clipped, hidden[3:5].mean(axis=0))


# --------------------------------------------------------------------------- #
# transfer helpers
# --------------------------------------------------------------------------- #
def _stage_df():
    return pd.DataFrame({
        "gene": ["TP53", "TP53", "MSH2", "MSH2", "BRCA1", "MLH1"],
        "uniprot_id": ["P04637"] * 2 + ["P43246"] * 2 + ["P38398", "P40692"],
        "position": [10, 11, 12, 13, 14, 15],
        "wt_aa": list("AAAAAA"),
        "mut_aa": list("VVVVVV"),
        "am_pathogenicity": [0.1, 0.9, np.nan, 0.4, 0.2, 0.7],
        "zs_eve": [1.0, np.nan, 0.5, 0.2, np.nan, np.nan],
        "dms_bin_median": [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],   # must never be used
        "clinvar_label": [1.0, 0.0, 1.0, np.nan, 0.0, 1.0],
        "clinical_label": [np.nan] * 6,
        "label_source": ["clinvar", "clinvar", "clinvar", "", "", "clinvar"],
        "label_weight": [1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        "label": [1.0, 0.0, 1.0, np.nan, 0.0, 1.0],
    })


def test_masked_marginal_pathogenicity_score_is_oriented():
    """`pathogenicity_score` must run higher-is-worse; `score` must not.

    The raw PLLR is negative for damaging substitutions, so feeding it
    straight into a metric against a pathogenic=1 label yields ``1 - AUC``.
    That is how the backbone comparison came to report ROC-AUC 0.03-0.18 for
    two strong 650M models -- numbers that read as "the protein language model
    is useless here" when the real figures were 0.82-0.97.
    """
    from src.mvmamba_features import MaskedMarginalScorer

    scorer = MaskedMarginalScorer.__new__(MaskedMarginalScorer)
    raw = np.array([-8.0, -1.0, 0.5])          # first is the most damaging
    scorer.score = lambda df, sequence: raw    # type: ignore[method-assign]

    oriented = MaskedMarginalScorer.pathogenicity_score(scorer, None, "")
    assert np.allclose(oriented, -raw)
    # The most damaging variant must rank highest once oriented, and lowest
    # before orientation -- that flip is the whole point.
    assert int(np.argmax(oriented)) == 0
    assert int(np.argmax(raw)) == 2


def test_mvmamba_local_features_differ_between_wt_and_vt():
    """The windowed local WT/VT contrast must not be identically zero.

    For chains longer than the model's positional capacity the local features
    come from a mutation-centred window. Slicing that window out of the
    wild-type chain for both sides makes ``l_vt`` equal ``l_wt``, which zeroes
    the ``l_vt - l_wt`` and ``|l_vt - l_wt|`` blocks -- a quarter of the
    MVmamba feature vector, and the part the recipe is actually built around.
    Only MSH6 (1360 aa) crosses the limit on this panel, so the failure showed
    up as one gene having a different feature space from the other three.

    Uses a stub backend so no checkpoint download is needed: hidden states are
    a deterministic function of the residues, which is all this contract needs.
    """
    from src.mvmamba_features import MVmambaFeatureExtractor

    seq_len, dim = 1100, 8               # > MAX_RESIDUES, so windowing applies
    rng = np.random.default_rng(0)
    sequence = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), seq_len))

    def encode(s: str) -> np.ndarray:
        # One distinct row per residue identity, so a substitution genuinely
        # changes the local window and nothing else does.
        return np.stack([np.full(dim, float(ord(c)), dtype=np.float32) for c in s])

    class StubBackend:
        hidden_dim = dim
        _aa_to_id = {a: i for i, a in enumerate("ACDEFGHIKLMNPQRSTVWY")}

        def embed_sequences(self, seqs):
            longest = max(len(x) for x in seqs)
            h = np.zeros((len(seqs), longest, dim), dtype=np.float32)
            for i, x in enumerate(seqs):
                h[i, :len(x)] = encode(x)
            lp = np.zeros((len(seqs), longest, len(self._aa_to_id)), dtype=np.float32)
            return h, lp

        def _embed_spans(self, jobs):
            return [encode(sub) for _, sub in jobs], [None] * len(jobs)

    ext = MVmambaFeatureExtractor.__new__(MVmambaFeatureExtractor)
    ext.backend = StubBackend()
    ext.local_window = 3
    ext.include_deltas = True
    ext.vt_chunk_size = 4

    positions = [5, 400, 900, 1098]
    df = pd.DataFrame({
        "gene": ["X"] * len(positions), "position": positions,
        "wt_aa": [sequence[p - 1] for p in positions],
        "mut_aa": ["W" if sequence[p - 1] != "W" else "Y" for p in positions],
    })
    feats, _ = ext.extract(df, sequence)

    d = feats.shape[1] // 8
    l_wt, l_vt = feats[:, 2 * d:3 * d], feats[:, 3 * d:4 * d]
    delta, abs_delta = feats[:, 5 * d:6 * d], feats[:, 7 * d:8 * d]
    assert not np.allclose(l_wt, l_vt), "VT local window collapsed onto the WT one"
    assert np.abs(delta).max() > 0, "l_vt - l_wt is identically zero"
    assert np.abs(abs_delta).max() > 0, "|l_vt - l_wt| is identically zero"


def test_gene_constant_priors_are_droppable():
    """The gnomAD gene-level constraint columns must be removable by name.

    They hold one value per gene, so under leave-one-gene-out they are a gene
    identifier rather than variant evidence: a head can memorise each training
    gene's base rate and then meets an unseen vector on the held-out gene.
    Measured cost of leaving them in a scratch-trained run: MLH1 collapsed to
    ROC-AUC 0.500 / MCC 0.000 in every seed tried, against 0.965 +/- 0.007
    with them dropped.
    """
    df = pd.DataFrame({
        "gene": ["MLH1", "MLH1", "MSH2", "MSH2"],
        "am_pathogenicity": [0.9, 0.1, 0.8, 0.2],
        "zs_revel": [0.9, 0.1, 0.7, 0.3],
        "gnomad_pli": [1.0, 1.0, 0.99, 0.99],      # constant within gene
        "gnomad_mis_z": [3.1, 3.1, 2.7, 2.7],
        "dms_score_median": [0.5, 0.4, 0.3, 0.2],  # must never be a prior
    })
    kept = prior_columns_of(df)
    assert "gnomad_pli" in kept and "gnomad_mis_z" in kept

    dropped = prior_columns_of(df, drop_gene_constant=True)
    assert not (set(GENE_CONSTANT_PRIOR_COLS) & set(dropped))
    assert "am_pathogenicity" in dropped and "zs_revel" in dropped
    # The non-circularity contract still holds either way.
    assert "dms_score_median" not in kept
    assert "dms_score_median" not in dropped

    # Every listed column really is gene-constant in this fixture, which is
    # the property that makes dropping them correct rather than arbitrary.
    for col in ("gnomad_pli", "gnomad_mis_z"):
        assert (df.groupby("gene")[col].nunique() == 1).all()


# --------------------------------------------------------------------------- #
# Feature-family ablation (MISSING_EVIDENCE item 3, Table 4 rows 4-7)
# --------------------------------------------------------------------------- #
def _ablation_fixture():
    """One column from every ablatable group, plus a DMS column that is never
    a prior and a label the ablation must not touch."""
    return pd.DataFrame({
        "gene": ["MLH1", "MLH1", "MSH2", "MSH2"],
        "am_pathogenicity": [0.9, 0.1, 0.8, 0.2],
        "zs_esm1b": [-8.0, -1.0, -7.0, -2.0],
        "rank_am_pathogenicity": [1.0, 0.0, 1.0, 0.0],
        "consensus_rank": [0.9, 0.1, 0.8, 0.2],
        "in_domain": [1, 0, 1, 0],
        "in_interpro_domain": [1, 0, 0, 1],
        "is_functional_site": [0, 0, 1, 0],
        "af_plddt": [95.0, 40.0, 88.0, 35.0],
        "af_disordered": [0, 1, 0, 1],
        "gnomad_log10_af": [-5.0, -2.0, -4.5, -2.5],
        "acmg_ba1": [0, 1, 0, 1],
        "acmg_bs1": [0, 1, 0, 1],
        "acmg_pm2": [1, 0, 1, 0],
        "gnomad_pli": [1.0, 1.0, 0.99, 0.99],
        "gnomad_mis_z": [3.1, 3.1, 2.7, 2.7],
        "dms_score_median": [0.5, 0.4, 0.3, 0.2],
        "label": [1, 0, 1, 0],
    })


def test_every_advertised_ablation_group_removes_something():
    """A group name the CLI accepts must map to real columns.

    A group that silently matches nothing would report an "ablation" identical
    to the full model, and the table row would be a lie rather than an error.
    """
    df = _ablation_fixture()
    full = prior_columns_of(df)
    for group in ABLATABLE_PRIOR_GROUPS:
        kept = prior_columns_of(df, drop_groups=[group],
                                allow_proxy_leak=True)
        assert len(kept) < len(full), f"group {group!r} dropped no columns"
        assert set(kept) < set(full), f"group {group!r} invented columns"


def test_ablation_groups_partition_the_prior_columns():
    """Dropping every group at once must leave nothing behind.

    A prior column belonging to no group is invisible to the ablation table:
    it can never be removed, so no row of Table 4 accounts for it.
    """
    df = _ablation_fixture()
    kept = prior_columns_of(df, drop_groups=list(ABLATABLE_PRIOR_GROUPS),
                            allow_proxy_leak=True)
    assert kept == [], f"prior columns in no ablation group: {kept}"

    # Checked against the constant as well as the fixture, so a prior column
    # added later without an ablation group fails here rather than in review.
    union = set().union(*PRIOR_FEATURE_GROUPS.values())
    assert union == set(TRANSFER_PRIOR_COLS)


def test_prior_scores_group_catches_prefix_families():
    """zs_*, rank_* and the consensus column go with AlphaMissense.

    They are derived from the same published predictors; removing only the
    literal am_pathogenicity column would leave the score signal in place.
    """
    df = _ablation_fixture()
    kept = prior_columns_of(df, drop_groups=["prior_scores"])
    assert "am_pathogenicity" not in kept
    assert not [c for c in kept if c.startswith(("zs_", "rank_"))]
    assert "consensus_rank" not in kept
    # The other families survive untouched.
    assert {"af_plddt", "in_domain", "gnomad_log10_af"} <= set(kept)


def test_dropping_gnomad_while_keeping_prior_scores_is_refused():
    """The proxy guard, which is the point of the whole mechanism.

    AlphaMissense and the zero-shot models were trained on population data, so
    a "without gnomAD" arm that keeps them has not removed population
    information -- it has only removed the legible copy. The mistake shows up
    in no metric, so it is refused in code.
    """
    df = _ablation_fixture()
    with pytest.raises(ValueError, match="proxy"):
        prior_columns_of(df, drop_groups=["gnomad"])

    # Dropping the proxy alongside it is the valid form of the experiment.
    kept = prior_columns_of(df, drop_groups=["gnomad", "prior_scores"])
    assert not (set(PRIOR_FEATURE_GROUPS["gnomad"]) & set(kept))
    assert "am_pathogenicity" not in kept
    assert "af_plddt" in kept

    # And the escape hatch works, for the comparison where the proxy is the
    # subject rather than a contaminant.
    leaky = prior_columns_of(df, drop_groups=["gnomad"], allow_proxy_leak=True)
    assert "am_pathogenicity" in leaky
    assert "gnomad_log10_af" not in leaky


def test_unknown_ablation_group_is_an_error():
    """A typo must fail loudly, not quietly ablate nothing."""
    with pytest.raises(ValueError, match="Unknown prior feature group"):
        drop_prior_groups(["am_pathogenicity"], ["structrue"])


def test_ablation_preserves_column_order_and_ignores_non_priors():
    """Order matters: the checkpoint schema check compares column order."""
    df = _ablation_fixture()
    full = prior_columns_of(df)
    kept = prior_columns_of(df, drop_groups=["structure"])
    assert kept == [c for c in full if c not in ("af_plddt", "af_disordered")]
    assert "dms_score_median" not in kept
    assert "label" not in kept


def test_within_gene_rank_features_use_no_labels():
    """Ranks must depend only on scores and gene, never on the label column."""
    base = pd.DataFrame({
        "gene": ["MLH1"] * 4 + ["MSH2"] * 4,
        "am_pathogenicity": [0.1, 0.4, 0.6, 0.9, 0.2, 0.3, 0.7, 0.8],
        "zs_gemme": [-1.0, -2.0, -3.0, -4.0, -1.5, -2.5, -3.5, -4.5],
    })
    a = add_within_gene_rank_features(base.assign(label=[0, 0, 1, 1] * 2))
    b = add_within_gene_rank_features(base.assign(label=[1, 1, 0, 0] * 2))
    for col in ("rank_am_pathogenicity", "rank_zs_gemme", "consensus_rank"):
        pd.testing.assert_series_equal(a[col], b[col], check_names=False)

    # Ranks are per gene, so each gene spans the full 0-1 range independently.
    # np.isclose keeps this file runnable as a plain script: the suite is
    # invoked directly by run_mmr_pipeline.py and pytest is not a dependency.
    for gene in ("MLH1", "MSH2"):
        sub = a[a["gene"] == gene]["rank_am_pathogenicity"]
        assert np.isclose(sub.min(), 0.25) and np.isclose(sub.max(), 1.0)

    # GEMME is sign-flipped (higher raw = more fit), so its most negative raw
    # value must rank as the MOST pathogenic within its gene.
    mlh1 = a[a["gene"] == "MLH1"]
    assert np.isclose(mlh1.loc[mlh1["zs_gemme"].idxmin(), "rank_zs_gemme"], 1.0)


def test_prior_matrix_excludes_dms_and_adds_missingness():
    df = _stage_df()
    X, cols = prior_matrix(df)
    assert not any(c.startswith("dms") for c in cols)
    assert "is_missing_am_pathogenicity" in cols
    assert X.shape == (6, len(cols))
    # NaN imputed to the column median; flag column marks it.
    idx = cols.index("am_pathogenicity")
    mid = float(df["am_pathogenicity"].median())
    assert np.isclose(X[2, idx], mid)
    assert X[2, cols.index("is_missing_am_pathogenicity")] == 1.0
    # zs_eve missing on the MLH1 row -> imputed + flagged.
    zidx = cols.index("zs_eve")
    zmid = float(df["zs_eve"].median())
    assert np.isclose(X[5, zidx], zmid)
    assert X[5, cols.index("is_missing_zs_eve")] == 1.0


def test_align_rows_matches_on_variant_key():
    meta = _stage_df()
    target = meta.iloc[[0, 2]].copy()
    mask = align_rows(meta, target)
    assert mask.tolist() == [True, False, True, False, False, False]


def test_select_stage_rows_pretrain_excludes_all_mmr():
    sel = select_stage_rows(_stage_df(), stage="pretrain",
                            mode="leave_gene_out", holdout_gene=None)
    assert not {"MSH2", "MLH1"} & set(sel["gene"])
    assert {"TP53", "BRCA1"} <= set(sel["gene"])


def test_select_stage_rows_finetune_clinical_only_and_holdout():
    sel = select_stage_rows(_stage_df(), stage="finetune",
                            mode="leave_gene_out", holdout_gene="MSH2")
    assert "MSH2" not in set(sel["gene"])
    # Fine-tuning keeps only clinical-label MMR rows (BRCA1 is dms-sourced,
    # TP53 is not an MMR gene at all).
    assert set(sel["gene"]) == {"MLH1"}


def test_feature_bundle_stack_order_esm_then_priors():
    bundle = FeatureBundle(
        X_esm=np.zeros((3, 4), dtype=np.float32),
        X_prior=np.ones((3, 2), dtype=np.float32),
        prior_cols=["a", "b"], meta=pd.DataFrame({"x": range(3)}))
    stacked = bundle.stack()
    assert stacked.shape == (3, 6)
    assert (stacked[:, :4] == 0).all() and (stacked[:, 4:] == 1).all()


def test_prior_matrix_reuses_supplied_impute_values():
    """Stage 2 must impute with stage 1's constants, not its own medians.

    ``zs_*`` priors are missing for the overwhelming majority of rows, so the
    imputed constant is what the head mostly sees. Recomputing it per stage
    silently shifts the feature the pretrained weights were fitted on.
    """
    df = _stage_df()
    cols = prior_matrix(df)[1]
    idx = cols.index("am_pathogenicity")
    own_median = float(df["am_pathogenicity"].median())

    # Sanity: without a supplied value, the frame's own median is used.
    X_own, _ = prior_matrix(df)
    assert np.isclose(X_own[2, idx], own_median)

    # With one supplied, that exact value fills the gap instead.
    pinned = own_median + 0.25
    X_pinned, cols_pinned = prior_matrix(
        df, impute_values={"am_pathogenicity": pinned})
    assert cols_pinned == cols, "supplying fill values must not reorder columns"
    assert np.isclose(X_pinned[2, idx], pinned)
    # Observed values are untouched; only the missing cell moves.
    observed = df["am_pathogenicity"].notna().to_numpy()
    assert np.allclose(X_pinned[observed, idx], X_own[observed, idx])
    # The missingness flag still marks it as imputed.
    assert X_pinned[2, cols.index("is_missing_am_pathogenicity")] == 1.0

    # Columns not mentioned keep falling back to their own median.
    zidx = cols.index("zs_eve")
    assert np.isclose(X_pinned[5, zidx], float(df["zs_eve"].median()))


def test_prior_impute_values_round_trip():
    """What a stage reports as its fill values must reproduce its matrix."""
    df = _stage_df()
    X_ref, cols = prior_matrix(df)
    values = prior_impute_values(df, cols)
    assert "am_pathogenicity" in values and "zs_eve" in values
    assert not any(k.startswith("is_missing_") for k in values)
    X_replay, cols_replay = prior_matrix(df, columns=cols, impute_values=values)
    assert cols_replay == cols
    assert np.allclose(X_ref, X_replay)


def test_assemble_features_reports_impute_values():
    from src.transfer import assemble_features
    df = _stage_df()
    bundle = assemble_features(
        df, sequence_by_gene={}, model_name="unused",
        processed_dir=Path(tempfile.gettempdir()), device=torch.device("cpu"),
        features_mode="priors")
    assert bundle.impute_values, "priors mode must report its fill values"
    assert set(bundle.impute_values) <= set(bundle.prior_cols)


def test_feature_cache_rejects_a_stale_hit():
    """A rebuilt variant table with the same row count must not silently
    reuse another table's cached embeddings."""
    from src.esm_extractor import _assert_cache_matches
    df = pd.DataFrame({"position": [10, 20, 30],
                       "wt_aa": list("AAA"), "mut_aa": list("CDE")})
    _assert_cache_matches(df.copy(), df, Path("cache.npz"))          # identical: fine

    reordered = df.iloc[::-1].reset_index(drop=True)
    for bad, why in [(reordered, "reordered rows"),
                     (df.assign(position=[10, 20, 31]), "a different residue"),
                     (df.assign(mut_aa=list("CDF")), "a different substitution")]:
        try:
            _assert_cache_matches(bad, df, Path("cache.npz"))
        except RuntimeError as exc:
            assert "overwrite_cache" in str(exc)
        else:
            raise AssertionError(f"stale cache with {why} was accepted")

    # A row-count change is still caught, with the older message.
    try:
        _assert_cache_matches(df.iloc[:2], df, Path("cache.npz"))
    except RuntimeError as exc:
        assert "rows" in str(exc)
    else:
        raise AssertionError("row-count mismatch was accepted")


def test_build_model_registry_covers_architectures():
    from src.transfer import build_model
    assert build_model("esm", [16]).__class__ is BranchHead
    assert build_model("concat", [16, 8]).__class__ is ConcatFusionHead
    assert build_model("gatewave", [16, 8]).__class__ is GateWaveFusionHead


def test_transfer_head_checkpoint_roundtrips_weights_scalers_and_threshold():
    """A saved stage-2 head must reload to bit-identical predictions.

    Before save_transfer_head existed, run_mmr_transfer.py kept only the
    metrics row -- the trained head and its per-view StandardScaler were both
    lost on exit, so no held-out probability could be reproduced.
    """
    import numpy as np
    import torch
    from src.transfer import (
        build_model, load_transfer_head, predict_logits, save_transfer_head,
    )

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dev = torch.device("cpu")
    d_esm, d_prior = 6, 4
    model = build_model("concat", dims=[d_esm, d_prior], hidden_dim=8, dropout=0.1)
    with torch.no_grad():                         # perturb off the init
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)

    raws = [rng.normal(2.0, 3.0, size=(5, d_esm)).astype(np.float32),
            rng.normal(-1.0, 0.5, size=(5, d_prior)).astype(np.float32)]
    scalers = [(x.mean(0), x.std(0) + 1e-6) for x in raws]
    scaled = [(x - m) / s for x, (m, s) in zip(raws, scalers)]
    before = predict_logits(model, scaled, dev)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "head.pt"
        save_transfer_head(
            path, model, arch="concat", dims=[d_esm, d_prior], hidden_dim=8,
            dropout=0.1, scalers=scalers, prior_cols=["a", "b", "c", "d"],
            feature_mode="esm+priors", esm_model="stub", esm_dim=d_esm,
            threshold=0.42, metrics={"roc_auc": 0.9},
            extra={"holdout_gene": "MSH2"})
        model2, scale_views, payload = load_transfer_head(path, device=dev)

    after = predict_logits(model2, scale_views(raws), dev)
    assert np.allclose(before, after, atol=1e-6)
    assert payload["threshold"] == 0.42
    assert payload["metrics"] == {"roc_auc": 0.9}
    assert payload["holdout_gene"] == "MSH2"
    assert payload["config"]["arch"] == "concat"
    assert payload["feature_columns"] == ["a", "b", "c", "d"]

    # Wrong number of feature views is a loud error, not a silent misalign.
    try:
        scale_views(raws[:1])
    except ValueError as exc:
        assert "views" in str(exc)
    else:
        raise AssertionError("mismatched view count was accepted")


# --------------------------------------------------------------------------- #
# gnomAD GraphQL retry classification
#
# gnomAD reports transient overload as HTTP 200 with an ``errors`` array, so
# ``raise_for_status`` never sees it. Before this was classified, a single
# "Service overloaded" reply aborted a ten-minute build on its first attempt
# without using any of the five configured retries.
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    """Replays a queued list of payloads and counts the POSTs it served."""

    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.calls = 0

    def post(self, *_args, **_kwargs):
        self.calls += 1
        return _FakeResponse(self._payloads.pop(0))


def test_gnomad_service_overloaded_is_retryable():
    from src.gnomad import _gql_error_is_retryable
    assert _gql_error_is_retryable([{"message": "Service overloaded"}])
    assert _gql_error_is_retryable([{"message": "Too Many Requests"}])
    assert _gql_error_is_retryable([{"message": "upstream timed out"}])


def test_gnomad_permanent_error_is_not_retryable():
    from src.gnomad import _gql_error_is_retryable
    assert not _gql_error_is_retryable([{"message": "Cannot query field 'nope'"}])
    assert not _gql_error_is_retryable([])
    assert not _gql_error_is_retryable(None)


def test_gnomad_mixed_errors_are_not_retryable():
    """A real bug alongside an overload must not be retried into silence."""
    from src.gnomad import _gql_error_is_retryable
    assert not _gql_error_is_retryable([
        {"message": "Service overloaded"},
        {"message": "Cannot query field 'nope'"},
    ])


def test_gql_retries_overload_then_returns_data():
    import src.gnomad as gnomad_mod

    session = _FakeSession([
        {"errors": [{"message": "Service overloaded"}]},
        {"errors": [{"message": "Service overloaded"}]},
        {"data": {"gene": {"gnomad_constraint": {"pLI": 0.99}}}},
    ])
    slept = []
    original_sleep = gnomad_mod.time.sleep
    gnomad_mod.time.sleep = slept.append
    try:
        data = gnomad_mod._gql(session, "query", {}, backoff_base=0.0)
    finally:
        gnomad_mod.time.sleep = original_sleep
    assert data["gene"]["gnomad_constraint"]["pLI"] == 0.99
    assert session.calls == 3, "should have retried twice before succeeding"
    assert len(slept) == 2, "should have backed off between attempts"


def test_gql_does_not_retry_a_permanent_error():
    import src.gnomad as gnomad_mod

    session = _FakeSession([{"errors": [{"message": "Cannot query field 'nope'"}]}])
    original_sleep = gnomad_mod.time.sleep
    gnomad_mod.time.sleep = lambda _s: (_ for _ in ()).throw(
        AssertionError("must not sleep on a permanent error"))
    try:
        try:
            gnomad_mod._gql(session, "query", {}, backoff_base=0.0)
        except RuntimeError as exc:
            assert "Cannot query field" in str(exc)
        else:
            raise AssertionError("a permanent GraphQL error must raise")
    finally:
        gnomad_mod.time.sleep = original_sleep
    assert session.calls == 1, "a permanent error must fail on the first attempt"


def test_gql_exhausts_retries_and_mentions_the_cache():
    import src.gnomad as gnomad_mod

    session = _FakeSession([{"errors": [{"message": "Service overloaded"}]}] * 3)
    original_sleep = gnomad_mod.time.sleep
    gnomad_mod.time.sleep = lambda _s: None
    try:
        try:
            gnomad_mod._gql(session, "query", {}, max_retries=3, backoff_base=0.0)
        except RuntimeError as exc:
            # The operator needs to know a re-run resumes rather than restarts.
            assert "cached" in str(exc).lower()
        else:
            raise AssertionError("exhausted retries must raise")
    finally:
        gnomad_mod.time.sleep = original_sleep
    assert session.calls == 3


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed.append((fn.__name__, exc))
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - len(failed)}/{len(fns)} tests passed.")
    if failed:
        raise SystemExit(1)
