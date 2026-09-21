"""Regression suite: every defect that has already been paid for once.

Each test encodes a real bug found in the v1 codebase between 2026-08 and
2026-09, most of which silently produced *plausible* numbers rather than
crashing. They are written against the v2 ``vpdl`` API before that API exists:
this file is the specification the rewrite has to satisfy, not an afterthought
bolted on once it is working.

Provenance is cited per test. Do not delete or weaken a test here without
reading the incident that produced it -- every one of these cost days.

Source documents: docs/CODE_REVIEW.md (B1-B3, D1-D5),
docs/RUNLOG.md (2026-08-23, 08-28, 08-30, 09-06, 09-12), MISSING_EVIDENCE.md
(items 2, 3, 7, 12, 13, 14).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# L1. ProteinGym DMS_score_bin orientation
# ---------------------------------------------------------------------------
# RUNLOG 2026-08-23. ProteinGym's DMS_score_bin == 1 marks the TOP fitness half,
# i.e. the variant is TOLERATED. v1's builder mapped bin==1 straight to
# "pathogenic", inverting roughly 185,000 labels. Nothing failed; the model
# trained happily on backwards supervision. Caught only by anchoring the merged
# labels against AlphaMissense and the ProteinGym clinical benchmark.

def test_proteingym_score_bin_1_is_benign():
    from vpdl.sources.proteingym_dms import to_label

    assert to_label(dms_score_bin=1) == 0, (
        "DMS_score_bin==1 is the top fitness half = tolerated = benign. "
        "Mapping it to pathogenic inverts ~185k labels (RUNLOG 2026-08-23)."
    )
    assert to_label(dms_score_bin=0) == 1


def test_proteingym_labels_agree_with_alphamissense_anchor():
    """The check that actually caught the inversion: cross-source agreement.

    If a source's labels are inverted, agreement with an independent
    pathogenicity prior collapses below chance. Any new label-producing source
    must clear this before it is allowed to supply supervision.
    """
    from vpdl.assemble import label_anchor_agreement

    rng = np.random.default_rng(0)
    n = 500
    truth = rng.integers(0, 2, n)
    # AlphaMissense-like prior: high for pathogenic, low for benign, noisy.
    prior = np.where(truth == 1, rng.normal(0.8, 0.15, n), rng.normal(0.2, 0.15, n))

    assert label_anchor_agreement(truth, prior) > 0.7
    assert label_anchor_agreement(1 - truth, prior) < 0.3, (
        "An inverted label set must be detectable as sub-chance agreement "
        "with an independent prior, not silently accepted."
    )


# ---------------------------------------------------------------------------
# L2. Assay-derived feature that is the label in disguise
# ---------------------------------------------------------------------------
# RUNLOG 2026-08-23. `dms_bin_median` equalled the flipped label on 97.3% of
# rows. It was fed to the model as a feature and produced ROC-AUC 0.9987 --
# a number good enough that it was nearly written up before anyone asked why.

def test_feature_matrix_rejects_label_proxy_columns():
    from vpdl.features import assert_no_label_proxy

    n = 1000
    rng = np.random.default_rng(1)
    label = rng.integers(0, 2, n)
    df = pd.DataFrame({
        "honest_feature": rng.normal(size=n),
        "dms_bin_median": 1 - label,          # the actual v1 leak
    })

    with pytest.raises(ValueError, match="label proxy|leak"):
        assert_no_label_proxy(df, label, max_agreement=0.95)

    assert_no_label_proxy(df[["honest_feature"]], label, max_agreement=0.95)


# ---------------------------------------------------------------------------
# L3. Leakage groups must be protein-qualified
# ---------------------------------------------------------------------------
# CODE_REVIEW D2. The CV grouping key was the bare residue `position`. Pooling
# multiple proteins collapsed e.g. TP53:175 and BRCA1:175 into one "group",
# so folds were neither leakage-free nor the sizes they claimed to be.

def test_split_groups_are_protein_qualified():
    from vpdl.splits import group_keys

    df = pd.DataFrame({
        "uniprot_id": ["P40692", "P43246"],
        "position": [175, 175],
    })
    keys = group_keys(df)
    assert keys[0] != keys[1], (
        "Equal residue positions in different proteins must not share a "
        "leakage group (CODE_REVIEW D2)."
    )


def test_no_group_straddles_train_and_test():
    """The invariant the grouping exists to protect, asserted directly."""
    from vpdl.splits import group_keys, lopo_splits

    df = pd.DataFrame({
        "gene": ["MLH1"] * 6 + ["MSH2"] * 6,
        "uniprot_id": ["P40692"] * 6 + ["P43246"] * 6,
        "position": [10, 10, 11, 12, 13, 14] * 2,
        "label": [0, 1, 0, 1, 0, 1] * 2,
    })
    for train_idx, test_idx in lopo_splits(df):
        train_groups = set(group_keys(df.iloc[train_idx]))
        test_groups = set(group_keys(df.iloc[test_idx]))
        assert not (train_groups & test_groups)


# ---------------------------------------------------------------------------
# L4. Amino-acid validation must not be vacuous
# ---------------------------------------------------------------------------
# CODE_REVIEW B1. v1 checked `np.char.str_len(np.asarray(df["mut_aa"],
# dtype="U1")) == 1`. Casting to dtype="U1" truncates every value to one
# character first, so the length check was always true. The validation ran,
# passed, and validated nothing.

@pytest.mark.parametrize("bad", ["XX", "Z", "", "1", "ala"])
def test_invalid_mutant_residue_is_rejected(bad):
    from vpdl.sources.base import validate_substitution

    with pytest.raises(ValueError):
        validate_substitution(wt_aa="A", mut_aa=bad, position=1)


def test_valid_substitution_passes():
    from vpdl.sources.base import validate_substitution

    validate_substitution(wt_aa="A", mut_aa="V", position=1)


def test_wildtype_residue_checked_against_canonical_sequence():
    """Numbering safety: a row whose wt_aa disagrees with UniProt is dropped.

    This is what keeps RefSeq/ProteinGym coordinates from silently landing on
    the wrong residue after an isoform mismatch.
    """
    from vpdl.sources.base import validate_against_sequence

    sequence = "MAVQ"
    assert validate_against_sequence(sequence, position=2, wt_aa="A") is True
    assert validate_against_sequence(sequence, position=2, wt_aa="Q") is False
    assert validate_against_sequence(sequence, position=99, wt_aa="A") is False


# ---------------------------------------------------------------------------
# L5. Caches must key on the feature schema they hold
# ---------------------------------------------------------------------------
# CODE_REVIEW B3. Features were cached as `{gene}_{model}_features.npz`
# regardless of which prior columns had been appended, so a run with priors
# would silently load a cache built without them.

def test_feature_cache_key_changes_with_schema():
    from vpdl.features import cache_key

    base = cache_key(gene="MLH1", model="esm2_t33_650M_UR50D", columns=["a", "b"])
    more = cache_key(gene="MLH1", model="esm2_t33_650M_UR50D", columns=["a", "b", "c"])
    reordered = cache_key(gene="MLH1", model="esm2_t33_650M_UR50D", columns=["b", "a"])

    assert base != more, "A different feature schema must not reuse a cache."
    assert base == reordered, "Column order alone is not a schema difference."


# ---------------------------------------------------------------------------
# L6. PLLR orientation
# ---------------------------------------------------------------------------
# RUNLOG 2026-08-30. Raw pseudo-log-likelihood-ratio is NEGATIVE for damaging
# variants. compare_backbones.py compared it directly against a pathogenic=1
# label and reported ROC-AUC 0.03-0.18 for two strong 650M models -- output
# that reads as "protein language models are useless here" and was very nearly
# believed.

def test_pllr_pathogenicity_score_is_higher_for_damaging():
    from vpdl.features import pllr_to_pathogenicity

    damaging_raw_pllr = -4.2
    tolerated_raw_pllr = 0.3

    assert pllr_to_pathogenicity(damaging_raw_pllr) > pllr_to_pathogenicity(
        tolerated_raw_pllr
    ), "Higher score must mean more pathogenic (RUNLOG 2026-08-30)."


def test_score_orientation_guard_catches_inverted_predictor():
    """A predictor scoring below chance is a sign error, not a weak model.

    Any scorer wired into evaluation must pass this, so an orientation bug
    surfaces as a loud failure instead of a publishable-looking bad result.
    """
    from vpdl.evaluate import assert_orientation

    y = np.array([0, 0, 1, 1])
    correct = np.array([0.1, 0.2, 0.8, 0.9])
    inverted = 1.0 - correct

    assert_orientation(y, correct)
    with pytest.raises(ValueError, match="orientation|inverted"):
        assert_orientation(y, inverted)


# ---------------------------------------------------------------------------
# L7. Seed before model construction, on every split
# ---------------------------------------------------------------------------
# MISSING_EVIDENCE item 12 / RUNLOG 2026-09-06. v1 built the model (and so
# initialised the head) before anything seeded the RNG, so the FIRST split of
# every process drew its weights from process-start entropy. MMR_GENES is
# ordered (MLH1, ...), so MLH1 -- and only MLH1 -- was unseeded in every run
# ever done. Two runs of one cell at one commit disagreed by 0.021 AUROC on
# MLH1 while the other three genes came back bit-identical.

def test_every_split_is_seeded_including_the_first():
    from vpdl.models.mlp import MLPClassifier

    def first_layer_weights(seed, split_index):
        model = MLPClassifier(n_features=8, seed=seed, split_index=split_index)
        return model.initial_weights()

    # Same seed + same split -> identical init, for the FIRST split too.
    for split_index in range(4):
        a = first_layer_weights(seed=42, split_index=split_index)
        b = first_layer_weights(seed=42, split_index=split_index)
        np.testing.assert_array_equal(
            a, b,
            err_msg=f"Split {split_index} is not deterministically seeded. "
                    "Split 0 is the one that was broken in v1.",
        )

    # Different splits must not share an initialisation.
    assert not np.array_equal(
        first_layer_weights(42, 0), first_layer_weights(42, 1)
    )


def test_split_seed_is_keyed_on_gene_not_loop_position():
    """A gene dropping out must not re-seed every other gene's split.

    Found 2026-09-20 by review, not by the test above — which passes happily
    while the caller feeds it a positional index. `lopo_splits` skips genes with
    no rows without incrementing, so losing PMS2 (which this project did, to a
    build flag) shifts MSH6 from index 3 to index 2 and silently changes its
    initialisation. Same cell, same seed, different numbers.
    """
    from vpdl.models.base import derive_seed

    four_genes = {g: derive_seed(42, g) for g in ("MLH1", "MSH2", "MSH6", "PMS2")}
    three_genes = {g: derive_seed(42, g) for g in ("MLH1", "MSH2", "MSH6")}

    for gene, seed in three_genes.items():
        assert seed == four_genes[gene], (
            f"{gene}'s seed changed when PMS2 left the panel — the split key is "
            "positional, not stable."
        )

    assert len(set(four_genes.values())) == 4, "gene keys must not collide"


def test_string_split_keys_are_stable_across_processes():
    """Must not use builtin hash(): its salt changes between interpreters.

    A per-process salt would make every run irreproducible across invocations,
    which is the exact failure derive_seed exists to prevent.
    """
    import subprocess
    import sys

    code = (
        "import sys; sys.path.insert(0, '.');"
        "from vpdl.models.base import derive_seed;"
        "print(derive_seed(42, 'MLH1'))"
    )
    runs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, check=True).stdout.strip()
        for _ in range(3)
    }
    assert len(runs) == 1, f"derive_seed is not stable across processes: {runs}"


# ---------------------------------------------------------------------------
# L8. Feature schema must never silently truncate
# ---------------------------------------------------------------------------
# RUNLOG 2026-08-28. The stage-1 checkpoint carried a 19-column prior schema.
# Stage 2 pinned its feature order to the checkpoint's -- correct for weight
# transfer -- and silently discarded the 8 richer columns the MMR table
# actually had, including gnomad_log10_af, the one feature PROJECT_PLAN Phase 3
# explicitly requires as an input. Fixing the flag moved mean ROC-AUC
# 0.9229 -> 0.9445.

def test_checkpoint_schema_mismatch_raises_rather_than_dropping_columns():
    from vpdl.models.base import align_features_to_checkpoint

    checkpoint_columns = ["a", "b", "c"]
    available_columns = ["a", "b", "c", "gnomad_log10_af", "acmg_pm2"]

    with pytest.raises(ValueError) as exc:
        align_features_to_checkpoint(available_columns, checkpoint_columns)

    assert "gnomad_log10_af" in str(exc.value), (
        "The error must name the columns being lost, not just report a count "
        "(RUNLOG 2026-08-28)."
    )


# ---------------------------------------------------------------------------
# L9. PMS2 homology gate is fail-closed
# ---------------------------------------------------------------------------
# PROJECT_PLAN Phase 1 / MISSING_EVIDENCE item 14. PMS2CL pseudogene homology
# makes short-read clinical calls in exons 11-15 untrustworthy. The gate must
# REFUSE to proceed when given no confirmation policy rather than defaulting to
# trusting the calls. (In 2026-09-13 the opposite error occurred: --exclude_pms2
# dropped the gene entirely and a whole grid trained on a 3-gene panel without
# anyone noticing until the split counts were checked.)

def test_pms2_gate_refuses_without_explicit_policy():
    from vpdl.sources.clinvar import apply_homology_gate

    df = pd.DataFrame({"gene": ["PMS2", "MLH1"], "position": [500, 100]})

    with pytest.raises(ValueError, match="PMS2|homology"):
        apply_homology_gate(df, policy=None)


def test_pms2_gate_with_codon_range_keeps_the_rest_of_the_gene():
    """Excluding the homology region must not exclude the gene.

    The 2026-09-13 rebuild passed --exclude_pms2 and produced a silently
    3-gene dataset; 16 grid cells trained on it before the zero-row split was
    noticed. A codon-range policy must retain out-of-region PMS2 variants.
    """
    from vpdl.sources.clinvar import apply_homology_gate

    df = pd.DataFrame({
        "gene": ["PMS2"] * 3,
        "position": [100, 500, 800],   # 500 and 800 fall inside 382-862
    })
    out = apply_homology_gate(df, policy={"codon_range": (382, 862)})

    assert len(out) == 3, "Gating withholds supervision; it does not drop rows."
    assert out.loc[out["position"] == 100, "homology_excluded"].iloc[0] == 0
    assert out.loc[out["position"] == 500, "homology_excluded"].iloc[0] == 1
    assert (out["gene"] == "PMS2").sum() > 0, "PMS2 must survive the gate."


def test_panel_gene_count_is_asserted_after_build():
    """A gene vanishing from the built table must fail the build, not the paper.

    MISSING_EVIDENCE item 14: PMS2 went from 21 usable holdout variants to 0,
    and the grid script logged a warning that nobody read while 16 cells
    trained on three genes.
    """
    from vpdl.assemble import assert_genes_present

    df = pd.DataFrame({"gene": ["MLH1", "MSH2", "MSH6"]})

    with pytest.raises(ValueError, match="PMS2"):
        assert_genes_present(df, expected=["MLH1", "MSH2", "MSH6", "PMS2"])


# ---------------------------------------------------------------------------
# L10. Manifest must describe the file actually on disk
# ---------------------------------------------------------------------------
# MISSING_EVIDENCE item 2. The two-phase build wrote the manifest, then rewrote
# the CSV, so the recorded SHA-256 was the pre-join one: the manifest named a
# 37.8 MB file that was 48.4 MB on disk, and flagged gnomAD as disabled on a
# table carrying gnomAD columns for 6,492 variants.

def test_manifest_checksum_matches_file_after_every_write(tmp_path):
    from vpdl.provenance import write_manifest, verify_manifest

    data = tmp_path / "dataset.csv"
    data.write_text("a,b\n1,2\n")
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, artefacts=[data])

    verify_manifest(manifest)                  # clean

    data.write_text("a,b\n1,2\n3,4\n")         # post-manifest mutation
    with pytest.raises(ValueError, match="checksum|stale"):
        verify_manifest(manifest)


# ---------------------------------------------------------------------------
# L11. Ablations must refuse to leave a proxy behind
# ---------------------------------------------------------------------------
# MISSING_EVIDENCE item 3. AlphaMissense was trained on population and clinical
# data, so a "without gnomAD" ablation is uninterpretable while AlphaMissense
# remains in the feature set: the ablation removes the legible copy of the
# signal and nothing else.

def test_ablation_refuses_when_a_proxy_for_the_removed_family_remains():
    from vpdl.features import resolve_ablation

    with pytest.raises(ValueError, match="proxy"):
        resolve_ablation(
            columns=["gnomad_log10_af", "alphamissense_score", "plddt"],
            drop_groups=["gnomad"],
        )

    resolved = resolve_ablation(
        columns=["gnomad_log10_af", "alphamissense_score", "plddt"],
        drop_groups=["gnomad"],
        allow_proxy_leak=True,
    )
    assert "gnomad_log10_af" not in resolved


# ---------------------------------------------------------------------------
# L12. Windowed wild-type / variant contrast must be non-degenerate
# ---------------------------------------------------------------------------
# RUNLOG 2026-08-30. For chains past the positional capacity, the variant-type
# local window was sliced from the WILD-TYPE sequence, so l_vt == l_wt exactly
# and the (l_vt - l_wt) / |l_vt - l_wt| blocks were identically zero. Only MSH6
# (1360 aa) exceeded the limit, so exactly one gene had a different feature
# space from the other three -- inside a leave-one-gene-out design.

@pytest.mark.parametrize("seq_len", [200, 1360])
def test_variant_window_differs_from_wildtype_window(seq_len):
    from vpdl.features import local_windows

    sequence = "A" * seq_len
    position = seq_len - 10
    wt_window, vt_window = local_windows(
        sequence=sequence, position=position, wt_aa="A", mut_aa="V", radius=3
    )

    assert wt_window != vt_window, (
        f"At length {seq_len} the variant window is identical to the "
        "wild-type window -- the WT/VT contrast is dead (RUNLOG 2026-08-30)."
    )
    assert vt_window[len(vt_window) // 2] == "V"


# ---------------------------------------------------------------------------
# L15. Gene-constant features must not survive a leave-one-gene-out split
# ---------------------------------------------------------------------------
# gnomAD's gene-level constraint columns (pLI, o/e missense, missense Z) take a
# single value across every variant in a gene. Under leave-one-gene-out that is
# gene identity and nothing else: constant across the training genes, one unseen
# value on the holdout. v1 guarded this with
# `prior_columns_of(df, drop_gene_constant=(eval == "lopo"))`. The trap is
# invisible in any within-gene split, where such a column looks perfectly
# well-behaved.

def test_gene_constant_features_are_dropped_for_lopo():
    from vpdl.features import drop_gene_constant

    df = pd.DataFrame({
        "gene": ["MLH1", "MLH1", "MSH2", "MSH2"],
        "feature_gnomad_log10_af": [-3.0, -5.0, -2.0, -4.0],   # varies within gene
        "feature_gnomad_pli": [0.99, 0.99, 0.71, 0.71],        # gene-constant
        "feature_gnomad_mis_z": [3.1, 3.1, 2.4, 2.4],          # gene-constant
    })
    columns = ["feature_gnomad_log10_af", "feature_gnomad_pli", "feature_gnomad_mis_z"]

    kept = drop_gene_constant(df, columns)

    assert kept == ["feature_gnomad_log10_af"], (
        "Gene-level constraint is constant within a gene and encodes gene "
        "identity under LOPO — it must not reach the model."
    )


def test_globally_constant_feature_is_also_dropped():
    from vpdl.features import drop_gene_constant

    df = pd.DataFrame({
        "gene": ["MLH1", "MLH1", "MSH2", "MSH2"],
        "feature_useless": [1.0, 1.0, 1.0, 1.0],
        "feature_real": [0.1, 0.9, 0.2, 0.8],
    })
    assert drop_gene_constant(df, ["feature_useless", "feature_real"]) == ["feature_real"]


def test_gnomad_ablation_group_actually_matches_gnomad_columns():
    """The ablation machinery must not silently no-op on a real feature family.

    `PRIOR_GROUPS` declared a 'gnomad' group and `PROXY_FOR` refused ablations
    leaving AlphaMissense behind, while no source emitted a gnomAD column at
    all — so the ablation would have removed nothing and reported success.
    """
    from vpdl.features import group_of
    from vpdl.sources.gnomad import provides

    for column in provides().feature_columns:
        assert group_of(column) == "gnomad", (
            f"{column} is produced by the gnomAD source but does not resolve to "
            "the 'gnomad' ablation group, so dropping that group would leave it in."
        )


def test_acmg_pm2_fires_when_allele_frequency_is_absent():
    """Unobserved in gnomAD IS the PM2 case, not a missing value to impute."""
    from vpdl.sources.gnomad import acmg_frequency_flags

    flags = acmg_frequency_flags(np.array([np.nan, 1e-6, 0.02, 0.10]))

    assert flags["feature_acmg_pm2"][0] == 1.0, "absent from gnomAD => PM2"
    assert flags["feature_acmg_pm2"][1] == 1.0, "vanishingly rare => PM2"
    assert flags["feature_acmg_bs1"][2] == 1.0, "above 0.1% => BS1"
    assert flags["feature_acmg_ba1"][3] == 1.0, "above 5% => BA1"
    assert flags["feature_acmg_ba1"][2] == 0.0, "2% is BS1 but not BA1"


def test_variants_absent_from_gnomad_are_pm2_after_assembly():
    """The flag function handled absence; the ASSEMBLED table did not.

    gnomAD returns only observed variants, so after the merge ~90% of
    substitutions had NaN in every gnomAD column and median imputation made them
    look like typical observed variants. The previous test covered
    acmg_frequency_flags in isolation and passed while this was broken — the
    same shape of gap as L7's seed test.
    """
    from vpdl.assemble import assemble
    from vpdl.sources.gnomad import UNOBSERVED_LOG10_AF, acmg_frequency_flags

    common = {"uniprot_id": "P40692", "gene": "MLH1"}
    observed = {**common, "position": 2, "wt_aa": "A", "mut_aa": "V"}
    absent = {**common, "position": 3, "wt_aa": "V", "mut_aa": "A"}

    clinvar = pd.DataFrame([
        {**observed, "label": 1.0, "label_source": "clinvar", "evidence_tier": "2"},
        {**absent, "label": 0.0, "label_source": "clinvar", "evidence_tier": "2"},
    ])
    flags = {k: float(v) for k, v in acmg_frequency_flags(0.02).items()}
    gnomad = pd.DataFrame([{**observed, "label": np.nan, "label_source": "gnomad",
                            "evidence_tier": "population",
                            "feature_gnomad_log10_af": np.log10(0.02), **flags}])

    table, _ = assemble({"clinvar": clinvar, "gnomad": gnomad},
                        sequences={"P40692": "MAVQ"}, anchor_column=None)
    by_position = table.set_index("position")

    assert by_position.loc[3, "feature_acmg_pm2"] == 1.0, "absent => PM2"
    assert by_position.loc[3, "feature_acmg_bs1"] == 0.0
    assert by_position.loc[3, "feature_gnomad_observed"] == 0.0
    assert by_position.loc[3, "feature_gnomad_log10_af"] == UNOBSERVED_LOG10_AF
    assert by_position.loc[2, "feature_acmg_pm2"] == 0.0, "2% is observed, not PM2"
    assert by_position.loc[2, "feature_gnomad_observed"] == 1.0


def test_acmg_thresholds_are_the_lynch_appropriate_ones():
    """BS1 at 1% and PM2 at 1e-4 are generic, and 10x too permissive here.

    v1 chose BS1 = 1e-3 and PM2 = 1e-5 for an autosomal-dominant early-onset
    cancer syndrome. A draft of the v2 gnomAD source silently shipped the
    generic values instead; a variant at AF 5e-3 would then fail to earn BS1.
    """
    from vpdl.sources.gnomad import ACMG_THRESHOLDS, acmg_frequency_flags

    assert ACMG_THRESHOLDS["bs1"] <= 1e-3
    assert ACMG_THRESHOLDS["pm2"] <= 1e-5
    flags = acmg_frequency_flags(np.array([5e-3, 5e-5]))
    assert flags["feature_acmg_bs1"][0] == 1.0, "0.5% must earn BS1 for Lynch"
    assert flags["feature_acmg_pm2"][1] == 0.0, "5e-5 is not rare enough for PM2"


# ---------------------------------------------------------------------------
# L16. Assembly must keep features from feature-only sources
# ---------------------------------------------------------------------------
# Found by review 2026-09-21, before any real run. assemble() stacked each
# source's rows and then applied label precedence to WHOLE rows with
# drop_duplicates(keep="first"). For every variant ClinVar labelled, the ClinVar
# row won and the AlphaMissense and gnomAD rows — the ones carrying features —
# were discarded. Every labelled variant reached training with all-NaN
# features, and the orientation check silently never ran because the label and
# the anchor were never on the same row.

def test_assembly_keeps_features_from_feature_only_sources():
    from vpdl.assemble import assemble

    variant = {"uniprot_id": "P40692", "position": 2, "wt_aa": "A",
               "mut_aa": "V", "gene": "MLH1"}
    clinvar = pd.DataFrame([{**variant, "label": 1.0, "label_source": "clinvar",
                             "evidence_tier": "2 star"}])
    alphamissense = pd.DataFrame([{**variant, "label": np.nan,
                                   "label_source": "alphamissense",
                                   "evidence_tier": "prior",
                                   "feature_alphamissense_score": 0.97}])
    gnomad = pd.DataFrame([{**variant, "label": np.nan, "label_source": "gnomad",
                            "evidence_tier": "population",
                            "feature_gnomad_log10_af": -5.2}])

    table, _ = assemble(
        {"clinvar": clinvar, "alphamissense": alphamissense, "gnomad": gnomad},
        sequences={"P40692": "MAVQ"}, anchor_column=None,
    )

    assert len(table) == 1, "one variant in, one row out"
    row = table.iloc[0]
    assert row["label"] == 1.0 and row["label_source"] == "clinvar"
    assert row["feature_alphamissense_score"] == 0.97, (
        "AlphaMissense feature lost — precedence was applied to the whole row."
    )
    assert row["feature_gnomad_log10_af"] == -5.2, "gnomAD feature lost"


def test_contradictory_labels_are_quarantined_not_resolved():
    """Two sources disagreeing is evidence of a problem, not a tie to break."""
    from vpdl.assemble import assemble

    variant = {"uniprot_id": "P40692", "position": 2, "wt_aa": "A",
               "mut_aa": "V", "gene": "MLH1"}
    clinvar = pd.DataFrame([{**variant, "label": 1.0, "label_source": "clinvar",
                             "evidence_tier": "2 star"}])
    dms = pd.DataFrame([{**variant, "label": 0.0, "label_source": "pg_dms",
                         "evidence_tier": "assay"}])

    table, report = assemble({"clinvar": clinvar, "pg_dms": dms},
                             sequences={"P40692": "MAVQ"}, anchor_column=None)

    assert np.isnan(table.iloc[0]["label"])
    assert table.iloc[0]["label_source"] == "conflict_quarantined"
    assert report.conflicts_quarantined == 1


def test_orientation_check_runs_once_sources_are_merged():
    """The anchor must reach the labelled rows, or the check is decorative."""
    from vpdl.assemble import assemble

    rng = np.random.default_rng(3)
    n = 60
    labels = rng.integers(0, 2, n).astype(float)
    base = pd.DataFrame({
        "uniprot_id": "P40692",
        "position": np.arange(1, n + 1),
        "wt_aa": "A", "mut_aa": "V", "gene": "MLH1",
    })
    clinvar = base.assign(label=1.0 - labels, label_source="clinvar",
                          evidence_tier="2 star")                 # INVERTED
    anchor = base.assign(label=np.nan, label_source="alphamissense",
                         evidence_tier="prior",
                         feature_alphamissense_score=np.where(labels == 1, 0.9, 0.1))

    with pytest.raises(ValueError, match="INVERTED|inverted|below chance"):
        assemble({"clinvar": clinvar, "alphamissense": anchor},
                 sequences={"P40692": "A" * n})


# ---------------------------------------------------------------------------
# L17. Arms differ in what they train on, never in what they are scored on
# ---------------------------------------------------------------------------
# Found by review 2026-09-21. The first design built one table per arm and
# scored each on its own labels, so the pooled arm's MSH2 fold was judged
# against thousands of DMS labels while the ClinVar arm's was judged against
# ~335 clinical ones. That compares two test sets, not two training regimes.
# The fix: one table, per-source labels kept (label__<source>), each arm picks
# its TRAINING sources, and every arm is scored on the same ClinVar variants.

def test_training_labels_depend_only_on_the_chosen_sources():
    from vpdl.assemble import resolve_labels

    table = pd.DataFrame({
        "label__clinvar": [1.0, np.nan, 1.0],
        "label__pg_dms": [0.0, 0.0, 1.0],
    })

    clinvar_only = resolve_labels(table, ["clinvar"])
    assert clinvar_only[0] == 1.0
    assert np.isnan(clinvar_only[1]), "a DMS-only variant is unlabelled for this arm"

    pooled = resolve_labels(table, ["clinvar", "pg_dms"])
    assert np.isnan(pooled[0]), "contradiction within the chosen sources is withheld"
    assert pooled[1] == 0.0
    assert pooled[2] == 1.0

    dms_only = resolve_labels(table, ["pg_dms"])
    assert dms_only[0] == 0.0, "no conflict when ClinVar is not being trained on"


def test_training_on_an_absent_source_is_an_error_not_an_empty_arm():
    from vpdl.assemble import resolve_labels

    table = pd.DataFrame({"label__clinvar": [1.0, 0.0]})
    with pytest.raises(ValueError, match="mavedb"):
        resolve_labels(table, ["mavedb"])


def test_assembly_keeps_each_sources_own_label():
    from vpdl.assemble import assemble

    variant = {"uniprot_id": "P40692", "position": 2, "wt_aa": "A",
               "mut_aa": "V", "gene": "MLH1"}
    clinvar = pd.DataFrame([{**variant, "label": 1.0, "label_source": "clinvar",
                             "evidence_tier": "2 star"}])
    dms = pd.DataFrame([{**variant, "label": 0.0, "label_source": "pg_dms",
                         "evidence_tier": "assay"}])

    table, _ = assemble({"clinvar": clinvar, "pg_dms": dms},
                        sequences={"P40692": "MAVQ"}, anchor_column=None)

    # The resolved label is quarantined, but the per-source labels survive —
    # so an arm training on ClinVar alone still sees this variant as pathogenic.
    assert table.iloc[0]["label__clinvar"] == 1.0
    assert table.iloc[0]["label__pg_dms"] == 0.0


# ---------------------------------------------------------------------------
# L13. Provenance gate blocks incomparable pooling
# ---------------------------------------------------------------------------
# MISSING_EVIDENCE items 7 and 14. Runs on two different dataset builds are
# indistinguishable from their metrics alone. In 2026-09-15 a 28-cell set was
# pooled from 16 cells on one dataset and 12 on another; the metric deltas
# reached 0.47 MCC at identical seeds and looked like model variance.

def test_pooling_runs_from_different_datasets_is_refused():
    from vpdl.provenance import assert_comparable

    run_a = {"dataset_sha256": "aaa", "split_hash": "s1", "feature_schema": "f1"}
    run_b = {"dataset_sha256": "bbb", "split_hash": "s1", "feature_schema": "f1"}
    run_c = {"dataset_sha256": "aaa", "split_hash": "s1", "feature_schema": "f2"}

    assert_comparable([run_a, run_a])

    with pytest.raises(ValueError, match="dataset"):
        assert_comparable([run_a, run_b])

    # Feature schema may differ across arms -- that IS the ablation -- but not
    # among seeds of one arm.
    assert_comparable([run_a, run_c])
    with pytest.raises(ValueError, match="schema"):
        assert_comparable([run_a, run_c], as_replicates=True)


def test_run_without_provenance_is_flagged_not_assumed_matching():
    from vpdl.provenance import assert_comparable

    with pytest.raises(ValueError, match="missing|provenance"):
        assert_comparable([{"dataset_sha256": "aaa"}, {}])


# ---------------------------------------------------------------------------
# L14. Checkpoint format tags are validated on load
# ---------------------------------------------------------------------------
# RUNLOG 2026-09-12. save() stamped a format tag that load() never checked, so
# a foreign or future-format file was accepted and failed later, deep inside
# model construction, with an error naming neither the file nor its format.
# Note the deliberate asymmetry: a MISSING tag is accepted (legacy
# checkpoints predate the field); a PRESENT but WRONG tag is rejected.

def test_checkpoint_with_foreign_format_tag_is_rejected(tmp_path):
    from vpdl.models.base import save_checkpoint, load_checkpoint

    path = tmp_path / "model.pt"
    save_checkpoint(path, state={"w": [1.0]}, fmt="vpdl-v2")

    load_checkpoint(path)  # matching tag

    save_checkpoint(path, state={"w": [1.0]}, fmt="something-else")
    with pytest.raises(ValueError, match="format"):
        load_checkpoint(path)


def test_checkpoint_without_format_tag_still_loads(tmp_path):
    from vpdl.models.base import save_checkpoint, load_checkpoint

    path = tmp_path / "legacy.pt"
    save_checkpoint(path, state={"w": [1.0]}, fmt=None)
    load_checkpoint(path)  # must not raise
