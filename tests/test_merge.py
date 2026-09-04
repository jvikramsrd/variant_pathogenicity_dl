"""Offline unit tests for the multi-source merge (src/extended_builder.py).

``assemble_master`` decides which source defines every training label and is
the single point where a silent join failure corrupts everything downstream,
yet it had no direct test coverage. These tests pin the behaviours that are
easy to break and hard to notice:

* label precedence and the ProteinGym DMS bin flip
* per-source and cross-source conflict quarantine
* the row universe being a union (unlabelled rows survive to be scored)
* the zero-shot join key surviving a float ``position`` column
* metadata backfill happening before de-duplication

Run:  python tests/test_merge.py   (or pytest tests/test_merge.py)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.extended_builder import (  # noqa: E402
    BuildStats,
    assemble_master,
    refresh_manifest,
    validate_master_for_export,
)

ACC = "P00001"
GENE = "TESTG"
STARS2 = "criteria provided, multiple submitters, no conflicts"
STARS1 = "criteria provided, single submitter"
STARS3 = "reviewed by expert panel"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _row(pos, wt, mut, gene=GENE):
    return dict(gene=gene, uniprot_id=ACC, position=pos, wt_aa=wt, mut_aa=mut,
                hgvs_p=None)


def _empty(cols):
    return pd.DataFrame(columns=cols)


EMPTY_CLIN = _empty(["gene", "uniprot_id", "position", "wt_aa", "mut_aa",
                     "hgvs_p", "clinical_label", "np_accession"])
EMPTY_AM = _empty(["gene", "uniprot_id", "position", "wt_aa", "mut_aa",
                   "hgvs_p", "am_pathogenicity", "am_class"])
EMPTY_ZS = _empty(["uniprot_id", "mutant"])
DOMAINS = pd.DataFrame([{"uniprot_id": ACC, "feature_type": "Domain",
                         "start": 1, "end": 1000, "description": "TestDomain"}])


def _merge(labeled=None, vus=None, dms=None, clin=None, am=None, zs=None):
    empty_cv = _empty(["gene", "uniprot_id", "position", "wt_aa", "mut_aa",
                       "hgvs_p", "label", "review_status"])
    empty_dms = _empty(["gene", "uniprot_id", "position", "wt_aa", "mut_aa",
                        "hgvs_p", "dms_score", "dms_bin", "dms_id",
                        "selection_type"])
    return assemble_master(
        labeled if labeled is not None else empty_cv,
        vus if vus is not None else empty_cv.drop(columns="label"),
        dms if dms is not None else empty_dms,
        clin if clin is not None else EMPTY_CLIN,
        am if am is not None else EMPTY_AM,
        zs if zs is not None else EMPTY_ZS,
        DOMAINS, BuildStats(), gene_of_uniprot={ACC: GENE})


def _at(master, pos):
    hit = master[master["position"] == pos]
    assert len(hit) == 1, f"expected exactly 1 row at position {pos}, got {len(hit)}"
    return hit.iloc[0]


def _dms(pos, wt, mut, bin_, assay, score=-1.0):
    return {**_row(pos, wt, mut), "dms_score": score, "dms_bin": bin_,
            "dms_id": assay, "selection_type": "x"}


# --------------------------------------------------------------------------- #
# Label precedence
# --------------------------------------------------------------------------- #
def test_label_precedence_clinvar_beats_clinical_beats_dms():
    labeled = pd.DataFrame([
        {**_row(10, "A", "C"), "label": 1.0, "review_status": STARS3},   # 3 stars
        {**_row(20, "A", "D"), "label": 0.0, "review_status": STARS2},   # 2 stars
        {**_row(50, "A", "G"), "label": 1.0, "review_status": STARS1},   # 1 star
    ])
    clin = pd.DataFrame([
        # Agrees with the ClinVar assertion at 20 (both benign), so this row
        # tests precedence rather than quarantine.
        {**_row(20, "A", "D"), "clinical_label": 0, "np_accession": "NP_1"},
        {**_row(30, "A", "E"), "clinical_label": 1, "np_accession": "NP_1"},
    ])
    clin["clinical_label"] = clin["clinical_label"].astype("Int64")
    dms = pd.DataFrame([_dms(40, "A", "F", 0, "A1")])
    m = _merge(labeled=labeled, clin=clin, dms=dms)

    assert _at(m, 10)["label_source"] == "clinvar"
    assert _at(m, 30)["label_source"] == "pg_clinical"
    assert _at(m, 40)["label_source"] == "dms"
    # Position 20 has both a ClinVar and a PG-clinical assertion. They agree
    # here, so nothing is quarantined and ClinVar must win precedence.
    assert _at(m, 20)["label_source"] == "clinvar", "ClinVar must outrank PG-clinical"

    # The weight ladder: stars 1/2/3+ -> 0.50/0.75/1.00, PG-clinical 0.75,
    # DMS 0.20. Note a 2-star ClinVar row deliberately *ties* PG-clinical --
    # precedence and confidence are separate axes, and only 3+ stars outweighs
    # an independent curated clinical label.
    assert _at(m, 10)["label_weight"] == 1.00
    assert _at(m, 20)["label_weight"] == 0.75
    assert _at(m, 30)["label_weight"] == 0.75
    assert _at(m, 50)["label_weight"] == 0.50
    assert _at(m, 40)["label_weight"] == 0.20
    assert _at(m, 10)["label_weight"] > _at(m, 50)["label_weight"] \
        > _at(m, 40)["label_weight"]


def test_dms_bin_is_flipped_into_pathogenicity():
    """ProteinGym DMS_score_bin=1 is the TOP fitness half, i.e. TOLERATED.

    Mapping it straight through inverted ~185k labels once. AUC is
    flip-invariant, so no metric caught it -- only this assertion can.
    """
    dms = pd.DataFrame([_dms(10, "A", "C", 0, "A1"),    # deleterious
                        _dms(20, "A", "D", 1, "A1")])   # tolerated
    m = _merge(dms=dms)
    assert _at(m, 10)["label"] == 1.0, "dms_bin=0 (low fitness) must be pathogenic"
    assert _at(m, 20)["label"] == 0.0, "dms_bin=1 (high fitness) must be benign"


def test_multi_assay_dms_yields_no_label():
    """Current policy: only single-assay DMS rows become labels.

    Note this discards *agreeing* multi-assay evidence too (position 30), which
    is a deliberate conservatism, not an oversight -- see docs/CODE_GUIDE.md §8.
    """
    dms = pd.DataFrame([
        _dms(20, "A", "D", 0, "A1"), _dms(20, "A", "D", 1, "A2"),   # disagree
        _dms(30, "A", "E", 0, "A1"), _dms(30, "A", "E", 0, "A2"),   # agree
    ])
    m = _merge(dms=dms)
    assert pd.isna(_at(m, 20)["label"])
    assert pd.isna(_at(m, 30)["label"]), "agreeing multi-assay rows are still unlabelled"
    assert _at(m, 30)["n_dms_assays"] == 2


# --------------------------------------------------------------------------- #
# Conflict quarantine
# --------------------------------------------------------------------------- #
def test_contradictory_clinvar_assertions_are_quarantined():
    """A key with both labels is never resolved by star count or row order."""
    labeled = pd.DataFrame([
        {**_row(10, "A", "C"), "label": 1.0, "review_status": STARS1},
        {**_row(10, "A", "C"), "label": 0.0, "review_status": STARS3},
    ])
    m = _merge(labeled=labeled)
    row = _at(m, 10)
    assert row["label_conflict"] == 1
    assert pd.isna(row["label"])
    assert pd.isna(row["clinvar_label"]), "conflicting evidence must not resolve"
    assert row["label_weight"] == 0.0


def test_cross_source_disagreement_is_quarantined():
    labeled = pd.DataFrame([{**_row(10, "A", "C"), "label": 1.0,
                             "review_status": STARS2}])
    clin = pd.DataFrame([{**_row(10, "A", "C"), "clinical_label": 0,
                          "np_accession": "NP_1"}])
    clin["clinical_label"] = clin["clinical_label"].astype("Int64")
    m = _merge(labeled=labeled, clin=clin)
    row = _at(m, 10)
    assert row["cross_source_conflict"] == 1 and row["label_conflict"] == 1
    assert pd.isna(row["label"]) and row["label_weight"] == 0.0
    # Raw evidence is preserved for investigation...
    assert row["clinvar_label"] == 1.0 and row["clinical_label"] == 0
    # ...but the row must not look like usable supervision from any angle.
    assert row["label_source"] == "", \
        "a quarantined row must not advertise a label source"


def test_quarantined_rows_cannot_reach_the_training_export():
    labeled = pd.DataFrame([
        {**_row(10, "A", "C"), "label": 1.0, "review_status": STARS2},
        {**_row(20, "A", "D"), "label": 1.0, "review_status": STARS2},
    ])
    clin = pd.DataFrame([{**_row(20, "A", "D"), "clinical_label": 0,
                          "np_accession": "NP_1"}])
    clin["clinical_label"] = clin["clinical_label"].astype("Int64")
    m = _merge(labeled=labeled, clin=clin)
    validate_master_for_export(m)                      # must not raise
    trainable = m[m["label"].notna()]
    assert set(trainable["position"]) == {10}
    assert (trainable["label_conflict"] == 0).all()


# --------------------------------------------------------------------------- #
# Row universe and metadata
# --------------------------------------------------------------------------- #
def test_row_universe_is_a_union_not_a_join():
    """Unlabelled rows must survive: ~966k of them exist only to be scored."""
    labeled = pd.DataFrame([{**_row(10, "A", "C"), "label": 1.0,
                             "review_status": STARS2}])
    vus = pd.DataFrame([{**_row(20, "A", "D"), "review_status": STARS2}])
    am = pd.DataFrame([{"uniprot_id": ACC, "position": 30, "wt_aa": "A",
                        "mut_aa": "E", "am_pathogenicity": 0.9,
                        "am_class": "pathogenic", "gene": None, "hgvs_p": None}])
    m = _merge(labeled=labeled, vus=vus, am=am)
    assert set(m["position"]) == {10, 20, 30}
    assert pd.isna(_at(m, 20)["label"]) and pd.isna(_at(m, 30)["label"])


def test_metadata_is_backfilled_before_deduplication():
    """AlphaMissense rows carry no gene; backfilling after dedupe leaves twins."""
    labeled = pd.DataFrame([{**_row(10, "A", "C"), "label": 1.0,
                             "review_status": STARS2}])
    am = pd.DataFrame([{"uniprot_id": ACC, "position": 10, "wt_aa": "A",
                        "mut_aa": "C", "am_pathogenicity": 0.9,
                        "am_class": "pathogenic", "gene": None, "hgvs_p": None}])
    m = _merge(labeled=labeled, am=am)
    assert len(m) == 1, "the same variant from two sources must collapse to one row"
    row = _at(m, 10)
    assert row["gene"] == GENE and row["hgvs_p"] == "p.Ala10Cys"
    assert row["am_pathogenicity"] == 0.9 and row["label"] == 1.0
    validate_master_for_export(m)


# --------------------------------------------------------------------------- #
# The silent-join regression
# --------------------------------------------------------------------------- #
def test_zeroshot_join_survives_a_float_position_column():
    """A float ``position`` used to render as "A10.0C" and match nothing.

    One source frame with a single NaN position makes the whole column float,
    so all 17 published zero-shot scores would join zero rows -- no error, no
    warning, just the strongest priors silently missing.
    """
    zs = pd.DataFrame([{"uniprot_id": ACC, "mutant": "A10C", "zs_eve": 0.5}])
    for pos in (10, 10.0):
        labeled = pd.DataFrame([{**_row(pos, "A", "C"), "label": 1.0,
                                 "review_status": STARS2}])
        m = _merge(labeled=labeled, zs=zs)
        assert m["zs_eve"].notna().sum() == 1, \
            f"zero-shot join lost its rows for position dtype {type(pos).__name__}"
        assert _at(m, 10)["zs_eve"] == 0.5


def test_key_dtype_is_normalised_across_sources():
    """Mixed int/float keys across frames must not depend on pandas upcasting."""
    labeled = pd.DataFrame([{**_row(10.0, "A", "C"), "label": 1.0,
                             "review_status": STARS2}])
    dms = pd.DataFrame([_dms(10, "A", "C", 0, "A1")])
    m = _merge(labeled=labeled, dms=dms)
    assert len(m) == 1, "the same variant keyed as int and float must not split"
    row = _at(m, 10)
    assert row["label_source"] == "clinvar" and row["n_dms_assays"] == 1


# --------------------------------------------------------------------------- #
# Robustness: inputs that used to abort or corrupt a whole build
# --------------------------------------------------------------------------- #
def test_gene_alias_from_one_source_does_not_split_a_variant():
    """The panel owns the gene symbol; a source's alias never wins.

    `gene` is part of the de-duplication key, so a source spelling it
    differently produced two rows for one variant, which then collided on
    MASTER_KEY and aborted the build at export.
    """
    labeled = pd.DataFrame([{**_row(10, "A", "C"), "label": 1.0,
                             "review_status": STARS2}])
    dms = pd.DataFrame([_dms(10, "A", "C", 0, "A1")])
    dms["gene"] = "TESTG_ALIAS"
    m = _merge(labeled=labeled, dms=dms)
    assert len(m) == 1, "a gene alias must not split one variant into two rows"
    assert _at(m, 10)["gene"] == GENE
    validate_master_for_export(m)


def test_hgvs_p_is_derived_not_taken_from_the_source():
    """Two renderings of one substitution must not survive as twin rows."""
    labeled = pd.DataFrame([{**_row(10, "A", "C"), "label": 1.0,
                             "review_status": STARS2, "hgvs_p": "p.A10C"}])
    dms = pd.DataFrame([_dms(10, "A", "C", 0, "A1")])
    m = _merge(labeled=labeled, dms=dms)
    assert len(m) == 1
    assert _at(m, 10)["hgvs_p"] == "p.Ala10Cys", "hgvs_p must be canonical"
    validate_master_for_export(m)


def test_non_standard_residues_do_not_abort_the_build():
    """UniProt canonical sequences really do contain U (selenocysteine).

    ONE_TO_THREE returns NaN for those, and a NaN hgvs_p aborts export -- so
    one odd residue in one gene used to kill an entire 80-gene panel build.
    """
    for wt, mut, expected in [("U", "C", "p.Sec10Cys"),
                              ("A", "X", "p.Ala10Xaa"),
                              ("B", "C", "p.Asx10Cys")]:
        m = _merge(dms=pd.DataFrame([_dms(10, wt, mut, 0, "A1")]))
        assert _at(m, 10)["hgvs_p"] == expected
        validate_master_for_export(m)          # must not raise


def test_residue_case_is_normalised():
    labeled = pd.DataFrame([{**_row(10, "A", "C"), "label": 1.0,
                             "review_status": STARS2}])
    dms = pd.DataFrame([_dms(10, "a", "c", 0, "A1")])
    m = _merge(labeled=labeled, dms=dms)
    assert len(m) == 1, "lowercase residues must not split a variant"
    assert _at(m, 10)["wt_aa"] == "A" and _at(m, 10)["mut_aa"] == "C"
    validate_master_for_export(m)


def test_synonymous_rows_are_dropped():
    """wt_aa == mut_aa is not a missense variant.

    Left in, a single-assay DMS row like this acquires a pathogenicity label.
    """
    dms = pd.DataFrame([_dms(10, "A", "A", 0, "A1"),     # synonymous
                        _dms(20, "A", "C", 0, "A1")])    # real
    stats = BuildStats()
    m = assemble_master(
        _empty(["gene", "uniprot_id", "position", "wt_aa", "mut_aa", "hgvs_p",
                "label", "review_status"]),
        _empty(["gene", "uniprot_id", "position", "wt_aa", "mut_aa", "hgvs_p",
                "review_status"]),
        dms, EMPTY_CLIN, EMPTY_AM, EMPTY_ZS, DOMAINS, stats,
        gene_of_uniprot={ACC: GENE})
    assert set(m["position"]) == {20}
    assert stats.master_dropped_synonymous == 1, "the drop must be counted"


def test_provenance_records_every_contributing_source():
    labeled = pd.DataFrame([{**_row(10, "A", "C"), "label": 1.0,
                             "review_status": STARS2}])
    am = pd.DataFrame([{"uniprot_id": ACC, "position": 10, "wt_aa": "A",
                        "mut_aa": "C", "am_pathogenicity": 0.9,
                        "am_class": "pathogenic", "gene": None, "hgvs_p": None}])
    zs = pd.DataFrame([{"uniprot_id": ACC, "mutant": "A10C", "zs_eve": 0.5}])
    m = _merge(labeled=labeled, am=am, zs=zs)
    row = _at(m, 10)
    tokens = set(str(row["sources"]).split("|"))
    assert {"clinvar", "alphamissense", "zeroshot_models", "domains"} <= tokens
    assert row["n_sources"] == len([t for t in tokens if t])


# --------------------------------------------------------------------------- #
# Manifest refresh
#
# build_extended_dataset writes manifest.json immediately after the CSV, but
# build_mmr_dataset.py then joins gnomAD and rewrites the CSV. Before the
# refresh existed, the manifest recorded include_gnomad=false and the digest of
# a file that no longer existed -- a provenance record that actively misleads.
# --------------------------------------------------------------------------- #
def _seed_manifest(tmp: Path) -> Path:
    (tmp / "extended_dataset.csv").write_text("a,b\n1,2\n")
    (tmp / "manifest.json").write_text(json.dumps({
        "built_at_utc": "2026-01-01T00:00:00+00:00",
        "parameters": {"genes": ["MLH1"], "include_gnomad": False, "min_stars": 2},
        "sources": {"clinvar": {"url": "https://example/clinvar"}},
        "stats": {"master_rows": 1, "gnomad_rows_panel": 0},
        "artefacts": {"extended_dataset.csv": {"sha256": "stale", "bytes": 0}},
    }, indent=2))
    return tmp / "manifest.json"


def test_refresh_manifest_recomputes_stale_checksums():
    from src.external_datasets import sha256_of

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        path = _seed_manifest(tmp)
        # The table changes after the manifest was written, as the gnomAD join
        # does; the recorded digest must follow the file, not the other way.
        (tmp / "extended_dataset.csv").write_text("a,b,gnomad_af_joint\n1,2,0.5\n")
        out = refresh_manifest(tmp)
        rec = out["artefacts"]["extended_dataset.csv"]
        assert rec["sha256"] != "stale"
        assert rec["sha256"] == sha256_of(tmp / "extended_dataset.csv")
        assert rec["bytes"] == (tmp / "extended_dataset.csv").stat().st_size
        assert json.loads(path.read_text())["artefacts"] == out["artefacts"]


def test_refresh_manifest_merges_updates_one_level_deep():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _seed_manifest(tmp)
        out = refresh_manifest(tmp, updates={
            "parameters": {"include_gnomad": True},
            "stats": {"gnomad_rows_panel": 4321},
        })
        # Corrected...
        assert out["parameters"]["include_gnomad"] is True
        assert out["stats"]["gnomad_rows_panel"] == 4321
        # ...without discarding sibling keys the caller did not mention.
        assert out["parameters"]["min_stars"] == 2
        assert out["parameters"]["genes"] == ["MLH1"]
        assert out["stats"]["master_rows"] == 1
        assert out["sources"]["clinvar"]["url"] == "https://example/clinvar"


def test_refresh_manifest_marks_the_second_write():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _seed_manifest(tmp)
        out = refresh_manifest(tmp)
        # The two-phase build must stay visible, not look like one atomic write.
        assert out["built_at_utc"] == "2026-01-01T00:00:00+00:00"
        assert "refreshed_at_utc" in out


def test_refresh_manifest_requires_an_existing_manifest():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        try:
            refresh_manifest(Path(d))
        except FileNotFoundError:
            return
        raise AssertionError("refreshing a directory with no manifest must raise")


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
