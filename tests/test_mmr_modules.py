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
    add_evidence_tiers,
    apply_pms2_homology_gate,
    make_balanced_subset,
    variant_key,
    write_leave_one_gene_out_manifest,
    load_leave_one_gene_out_manifest,
)
from src.mvmamba_features import centered_window_bounds  # noqa: E402
from src.transfer import (  # noqa: E402
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
