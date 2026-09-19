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
