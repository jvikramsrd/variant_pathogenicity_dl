"""Variant normalisation, reference validation, PMS2 QC, duplicates, canonical schema."""

from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest

from dl_helpers import PANEL, assembled_table, clinvar_record


# -- HGVS / identity ------------------------------------------------------------------

@pytest.mark.parametrize("text", ["p.Arg123His", "p.(Arg123His)", "p.R123H", "R123H"])
def test_hgvs_p_spellings_normalise_to_one_identity(text):
    from vpdl.dl.hgvs import format_hgvs_p, normalize_hgvs_p

    assert normalize_hgvs_p(text) == ("R", 123, "H")
    assert format_hgvs_p("R", 123, "H") == "p.Arg123His"


@pytest.mark.parametrize("text", ["p.Arg123=", "p.Arg123Ter", "p.Arg123*", "p.Arg123fs",
                                  "p.Arg123_Lys125del", "p.?", "p.Arg123Arg", "", None])
def test_non_missense_hgvs_is_refused_not_coerced(text):
    from vpdl.dl.hgvs import normalize_hgvs_p

    assert normalize_hgvs_p(text) is None


def test_variant_id_matches_the_split_key():
    from vpdl.dl.hgvs import parse_variant_id, variant_id
    from vpdl.splits import variant_keys

    frame = pd.DataFrame({"uniprot_id": ["P40692"], "position": [5], "wt_aa": ["A"],
                          "mut_aa": ["V"]})
    assert variant_id("P40692", 5, "A", "V") == variant_keys(frame)[0]
    assert parse_variant_id("P40692:5:A>V") == ("P40692", 5, "A", "V")


def test_clinvar_name_parts_and_codon_consistency():
    from vpdl.dl.hgvs import hgvs_consistency, parse_clinvar_name, transcript_status

    name = parse_clinvar_name("NM_000249.4(MLH1):c.367C>T (p.Arg123His)")
    assert (name.transcript, name.gene, name.hgvs_c, name.hgvs_p) == \
        ("NM_000249.4", "MLH1", "c.367C>T", "p.Arg123His")
    assert hgvs_consistency("c.367C>T", 123) == "consistent"      # codon 123 = c.367-369
    assert hgvs_consistency("c.367C>T", 124) == "inconsistent"
    assert hgvs_consistency("c.367+1G>A", 123) == "not_checkable"
    assert transcript_status("MLH1", "NM_000249.4") == "mane"
    assert transcript_status("MLH1", "NM_000249.3") == "mane_version_mismatch"
    assert transcript_status("MLH1", "NM_001167617.3") == "non_mane"


# -- ClinVar iterator: load() output must not change ------------------------------------

_HEADER = ["#AlleleID", "Type", "Name", "GeneID", "GeneSymbol", "ClinicalSignificance",
           "ReviewStatus", "Assembly", "Chromosome", "PositionVCF", "ReferenceAlleleVCF",
           "AlternateAlleleVCF", "VariationID"]


def _write_summary(path, rows):
    with gzip.open(path, "wt") as handle:
        handle.write("\t".join(_HEADER) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(c, "")) for c in _HEADER) + "\n")


def _old_load(path, genes, uniprot_by_gene, min_stars=2):
    """Verbatim copy of the pre-refactor loop — the oracle."""
    from vpdl.sources.clinvar import _review_stars, parse_hgvs_p, significance_to_label

    rows = []
    with gzip.open(path, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        index = {name: position for position, name in enumerate(header)}
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != len(header):
                continue
            gene = fields[index.get("GeneSymbol", 0)]
            if gene not in set(genes):
                continue
            if _review_stars(fields[index.get("ReviewStatus", 0)]) < min_stars:
                continue
            parsed = parse_hgvs_p(fields[index.get("Name", 0)])
            if parsed is None:
                continue
            wt, position, mut = parsed
            rows.append({"uniprot_id": uniprot_by_gene.get(gene), "position": position,
                         "wt_aa": wt, "mut_aa": mut, "gene": gene,
                         "label": significance_to_label(fields[index.get("ClinicalSignificance", 0)]),
                         "label_source": "clinvar",
                         "evidence_tier": fields[index.get("ReviewStatus", 0)]})
    frame = pd.DataFrame(rows).drop_duplicates(
        subset=["uniprot_id", "position", "wt_aa", "mut_aa"], keep="first")
    return frame


def test_clinvar_load_output_is_unchanged_by_the_iterator_refactor(tmp_path):
    from vpdl.sources.clinvar import iter_variant_summary, load

    good = "criteria provided, multiple submitters, no conflicts"
    rows = [
        {"GeneSymbol": "MLH1", "Name": "NM_000249.4(MLH1):c.2T>C (p.Met1Thr)",
         "ClinicalSignificance": "Pathogenic", "ReviewStatus": good, "Assembly": "GRCh37",
         "VariationID": 1},
        {"GeneSymbol": "MLH1", "Name": "NM_000249.4(MLH1):c.2T>C (p.Met1Thr)",
         "ClinicalSignificance": "Pathogenic", "ReviewStatus": good, "Assembly": "GRCh38",
         "Chromosome": "3", "PositionVCF": 36993350, "VariationID": 1},
        {"GeneSymbol": "MLH1", "Name": "NM_000249.4(MLH1):c.5G>A (p.Ser2Asn)",
         "ClinicalSignificance": "Benign", "ReviewStatus": "criteria provided, single submitter",
         "Assembly": "GRCh38", "VariationID": 2},
        {"GeneSymbol": "MSH2", "Name": "NM_000251.3(MSH2):c.4del (p.Ala2fs)",
         "ClinicalSignificance": "Pathogenic", "ReviewStatus": good, "VariationID": 3},
        {"GeneSymbol": "BRCA1", "Name": "NM_007294.4(BRCA1):c.5G>A (p.Asp2Asn)",
         "ClinicalSignificance": "Benign", "ReviewStatus": good, "VariationID": 4},
    ]
    path = tmp_path / "variant_summary.txt.gz"
    _write_summary(path, rows)
    uniprot = {g: a for g, a in PANEL.items()}
    new = load(path, list(PANEL), uniprot, pms2_policy={"codon_range": (382, 862)})
    old = _old_load(path, list(PANEL), uniprot).reset_index(drop=True)
    pd.testing.assert_frame_equal(new.drop(columns="homology_excluded"), old)

    records = list(iter_variant_summary(path, list(PANEL), min_stars=0))
    assert len(records) == 3                       # both assemblies + the 1-star record
    grch38 = [r for r in records if r["assembly"] == "GRCh38" and r["position"] == 1][0]
    assert grch38["position_vcf"] == "36993350" and grch38["stars"] == 2


# -- canonical table ------------------------------------------------------------------

def _canonical(table, sequences, records=(), **kwargs):
    from vpdl.dl.canonical import build_canonical
    return build_canonical(table, sequences, clinvar_records=list(records), **kwargs)


def test_discordant_clinvar_records_withhold_the_label(table, sequences):
    row = table[table["gene"] == "MLH1"].iloc[0]
    key = ("MLH1", int(row.position), row.wt_aa, row.mut_aa)
    records = [clinvar_record(*key, label=1, variation="10"),
               clinvar_record(*key, label=0, variation="11")]
    canonical, report, _ = _canonical(table, sequences, records)
    hit = canonical[canonical["variant_id"] == f"P40692:{row.position}:{row.wt_aa}>{row.mut_aa}"]
    assert hit["qc_status"].iloc[0] == "withheld"
    assert "clinvar_discordant" in hit["qc_flags"].iloc[0]
    assert np.isnan(hit["label__clinvar"].iloc[0]) and np.isnan(hit["clinical_label"].iloc[0])
    assert report.clinvar_discordant_changes == 1


def test_concordant_multi_records_keep_the_label_and_count(table, sequences):
    row = table[table["gene"] == "MSH2"].iloc[0]
    key = ("MSH2", int(row.position), row.wt_aa, row.mut_aa)
    records = [clinvar_record(*key, label=1, variation="20"),
               clinvar_record(*key, label=1, variation="21", ref="G", alt="A", pos_vcf="7")]
    canonical, report, _ = _canonical(table, sequences, records)
    hit = canonical[canonical["variant_id"] == f"P43246:{row.position}:{row.wt_aa}>{row.mut_aa}"]
    assert hit["clinical_label"].iloc[0] == 1.0 and hit["clinvar_n_records"].iloc[0] == 2
    assert "multi_nucleotide" in hit["qc_flags"].iloc[0]
    assert hit["qc_status"].iloc[0] == "pass"


def test_pms2_homology_region_is_withheld_unless_confirmed():
    from dl_helpers import random_sequences

    sequences = random_sequences(length=900)
    table = assembled_table(sequences, per_position=1)
    table = table[(table["gene"] != "PMS2") | table["position"].isin([100, 500, 800])]
    pms2 = table[table["gene"] == "PMS2"]
    confirmed = pms2[pms2["position"] == 800][["gene", "position", "wt_aa", "mut_aa"]].assign(
        method="long-range PCR")
    canonical, report, _ = _canonical(table, sequences, confirmations=confirmed)
    by_pos = canonical[canonical["gene"] == "PMS2"].set_index("position")
    assert not by_pos.loc[100, "pms2_pseudogene_risk"]
    assert by_pos.loc[100, "qc_status"] == "pass"
    assert by_pos.loc[500, "pms2_pseudogene_risk"] and by_pos.loc[500, "qc_status"] == "withheld"
    assert "PMS2CL" in by_pos.loc[500, "exclusion_reason"]
    assert np.isnan(by_pos.loc[500, "label__clinvar"])
    assert by_pos.loc[800, "orthogonal_confirmation"] == "long-range PCR"
    assert by_pos.loc[800, "qc_status"] == "pass"
    assert not by_pos.loc[500, "population_reliable"]
    assert report.pms2["in_homology_region"] == 2 and report.pms2["confirmed"] == 1


def test_reference_residue_mismatch_is_flagged_never_repaired(table, sequences):
    wrong = dict(sequences)
    acc = PANEL["MSH6"]
    wrong[acc] = "W" + wrong[acc][1:] if wrong[acc][0] != "W" else "Y" + wrong[acc][1:]
    canonical, _, _ = _canonical(table, wrong)
    first = canonical[(canonical["uniprot_id"] == acc) & (canonical["position"] == 1)]
    assert (first["qc_status"] == "withheld").all()
    assert first["qc_flags"].str.contains("reference_mismatch").all()


def test_clinvar_records_off_the_reference_are_listed_with_a_reason(table, sequences):
    acc = PANEL["MLH1"]
    wt = sequences[acc][4]
    bad_wt = "W" if wt != "W" else "Y"
    records = [clinvar_record("MLH1", 5, bad_wt, "A", label=1)]
    _, report, flagged = _canonical(table, sequences, records)
    assert report.flagged_records == 1
    assert flagged["flag_reason"].iloc[0].startswith("reference_mismatch")


def test_duplicate_variant_rows_fail_the_build(table, sequences):
    doubled = pd.concat([table, table.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate variant"):
        _canonical(doubled, sequences)


def test_canonical_schema_is_complete_and_validates(table, sequences, tmp_path):
    from vpdl.dl.canonical import CANONICAL_SCHEMA, validate_canonical, write_canonical
    from vpdl.provenance import verify_manifest

    canonical, report, flagged = _canonical(table, sequences)
    assert list(canonical.columns[:len(CANONICAL_SCHEMA)]) == list(CANONICAL_SCHEMA)
    assert validate_canonical(canonical) == []
    assert set(canonical["protein_cluster"]) == {"MutL", "MutS"}
    assert (canonical["split_logo"] == canonical["gene"]).all()
    assert canonical["split_random_debug"].between(0, 4).all()
    paths = write_canonical(canonical, report, flagged, sequences, tmp_path / "c.csv")
    verify_manifest(paths["manifest"])                      # L10: describes the files on disk
    reread = pd.read_csv(paths["table"])
    assert validate_canonical(reread) == []


def test_expert_label_conflict_is_withheld(table, sequences):
    row = table[table["gene"] == "MLH1"].iloc[3]
    key = ("MLH1", int(row.position), row.wt_aa, row.mut_aa)
    expert = pd.DataFrame([{"gene": key[0], "position": key[1], "wt_aa": key[2],
                            "mut_aa": key[3], "label": 1.0 - row.label__clinvar,
                            "expert_source": "insight:test.csv"}])
    records = [clinvar_record(*key, label=int(row.label__clinvar))]
    canonical, _, _ = _canonical(table, sequences, records, expert=expert)
    hit = canonical[canonical["variant_id"] == f"P40692:{key[1]}:{key[2]}>{key[3]}"]
    assert hit["qc_status"].iloc[0] == "withheld"
    assert "expert_conflict" in hit["qc_flags"].iloc[0]


def test_protein_level_gnomad_frequency_sums_alleles():
    from vpdl.dl.canonical import gnomad_protein_frequencies

    records = [{"gene": "MLH1", "position": 3, "wt_aa": "A", "mut_aa": "V", "ac": 2, "an": 1000,
                "af": 0.002},
               {"gene": "MLH1", "position": 3, "wt_aa": "A", "mut_aa": "V", "ac": 1, "an": 800,
                "af": 0.00125}]
    freq = gnomad_protein_frequencies(records).iloc[0]
    assert freq["gnomad_af"] == pytest.approx(0.00325)
    assert freq["gnomad_ac"] == 3 and freq["gnomad_an"] == 1000 and freq["gnomad_n_alleles"] == 2
