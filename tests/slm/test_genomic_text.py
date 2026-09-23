"""Clinical text: sentences, conclusion masking, ACMG codes, evidence units, duplicates, tiers.

Every narrative in these tests is invented for the test (no ClinVar text is
copied) and deliberately written in the shapes real laboratory reports use.
"""

from __future__ import annotations

import pytest

from vpdl.slm.text.acmg import (CRITERIA, GENE_SPECS, applicable_codes, find_codes, mask_codes,
                                task_policy)
from vpdl.slm.text.conclusion import classify_sentence, mask_conclusions, residual_assertions
from vpdl.slm.text.dedup import (cluster_documents, exact_group, jaccard, MinHasher, shingles,
                                 template_text)
from vpdl.slm.text.evidence import RULE_PROVENANCE, extract_units
from vpdl.slm.text.quality import assign_tier, restriction
from vpdl.slm.text.sentences import split_sentences


# -- sentences ---------------------------------------------------------------------

def test_variant_notation_does_not_end_a_sentence():
    text = ("The c.199G>A (p.Gly67Arg) change in NM_000249.4 was seen once. "
            "It was reported by Smith et al. 2020 in two families.")
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert sentences[0].text.startswith("The c.199G>A")
    assert "Smith et al. 2020" in sentences[1].text


def test_offsets_point_back_at_the_original_text():
    text = "First sentence here. Second one follows."
    for sentence in split_sentences(text):
        assert text[sentence.start:sentence.end] == sentence.text


# -- conclusion leakage --------------------------------------------------------------

@pytest.mark.parametrize("sentence", [
    "Therefore, this variant is classified as pathogenic.",
    "Based on the evidence above, the variant was classified as likely pathogenic.",
    "We classify this variant as a variant of uncertain significance.",
    "This variant is likely benign.",
    "In summary, the available evidence is currently insufficient to determine the role of this variant in disease.",
    "This variant meets ACMG criteria to be classified as likely pathogenic.",
    "Classification: Pathogenic",
    "It was reclassified from VUS to likely benign.",
])
def test_verdict_sentences_are_found(sentence):
    assert classify_sentence(sentence) is not None


@pytest.mark.parametrize("sentence", [
    "The variant was absent from population databases.",
    "The variant segregated with disease in three affected family members.",
    "It was observed in trans with a pathogenic variant in a patient.",
    "Pathogenic variants in MLH1 cause Lynch syndrome.",
    "In silico tools predict a benign effect on protein function.",
    "Other missense changes at this residue have been reported as pathogenic.",
    "This sequence change replaces glycine with arginine at codon 67.",
])
def test_evidence_sentences_are_kept(sentence):
    assert classify_sentence(sentence) is None


def test_other_laboratories_verdicts_are_their_own_category():
    assert classify_sentence("This has been classified as pathogenic by other laboratories.") == \
        "external_classification"


def test_masking_removes_the_verdict_and_nothing_else():
    text = ("This variant is absent from gnomAD. It segregated with disease in three relatives. "
            "Therefore, this variant has been classified as Pathogenic.")
    result = mask_conclusions(text)
    assert "absent from gnomAD" in result.text
    assert "segregated with disease" in result.text
    assert "classified" not in result.text
    assert result.categories() == {"classification_statement": 1}
    assert residual_assertions(result.text) == []


def test_external_verdicts_can_be_kept_for_an_ablation():
    text = "This has been reported as pathogenic by other laboratories. Therefore it is pathogenic."
    assert mask_conclusions(text, keep_external=True).n_masked == 1
    assert mask_conclusions(text).n_masked == 2


# -- ACMG codes ----------------------------------------------------------------------

def test_codes_are_parsed_with_their_strength_and_met_status():
    text = "Criteria applied: PM2_Supporting, PP3; not met: BS1, BP4."
    codes = {m.normalized: m.met for m in find_codes(text)}
    assert codes == {"PM2_Supporting": True, "PP3": True, "BS1": False, "BP4": False}


def test_a_lone_code_like_word_is_not_a_code():
    assert find_codes("PS1 is another name for presenilin 1 in this paper.") == []
    assert find_codes("Particulate matter PM2 was measured.") == []


def test_codes_are_removed_from_an_input_but_the_words_stay():
    masked, count = mask_codes("PM2: the variant is absent from gnomAD.")
    assert count == 1
    assert "absent from gnomAD" in masked and "PM2" not in masked


def test_gene_specifications_switch_codes_off_and_say_where_they_come_from():
    assert "PP5" not in applicable_codes("BRCA1")
    assert "PM1" not in applicable_codes("MLH1")          # InSiGHT specification
    assert "PM1" in applicable_codes("BRCA1")
    assert all(not spec.verified and spec.source for spec in GENE_SPECS)


def test_every_criterion_has_a_direction_and_a_default_strength():
    assert len(CRITERIA) == 28
    assert {c.direction for c in CRITERIA.values()} == {"pathogenic", "benign"}
    assert all(c.default_strength for c in CRITERIA.values())


def test_each_task_declares_what_it_masks():
    assert task_policy("classify").mask_acmg_codes is True
    assert task_policy("classify_from_codes").mask_acmg_codes is False
    with pytest.raises(KeyError):
        task_policy("invented_task")


# -- evidence units --------------------------------------------------------------------

def test_sentences_become_units_with_types_polarity_and_role():
    text = ("This variant is absent from the gnomAD population database. "
            "In silico tools predict the change to be tolerated. "
            "Therefore, this variant is classified as benign.")
    units = extract_units(text)
    assert [u.sentence_role for u in units] == ["evidence", "evidence", "conclusion"]
    assert "population" in units[0].evidence_types and units[0].evidence_polarity == "pathogenic"
    assert "computational" in units[1].evidence_types and units[1].evidence_polarity == "benign"
    assert units[2].evidence_types == ()


def test_a_sentence_no_rule_recognises_gets_no_type_rather_than_a_guess():
    units = extract_units("The sample was received in good condition.")
    assert units[0].evidence_types == ()
    assert units[0].evidence_polarity == "neutral"


def test_units_point_back_into_the_text_and_are_labelled_by_rules():
    text = "The variant segregated with disease in four relatives (PMID: 9912345)."
    unit = extract_units(text)[0]
    assert RULE_PROVENANCE.startswith("rule:")      # what records.py stamps on every unit
    assert text[unit.start:unit.end] == unit.text
    assert unit.pmids == ("9912345",)


def test_a_code_only_sentence_is_marked_so_it_can_be_dropped_when_codes_are_masked():
    units = extract_units("The following criteria were applied: PM2_Supporting, PP3.")
    assert units[0].sentence_role == "code_list"


# -- duplicates and templates -----------------------------------------------------------

def test_the_same_template_about_two_variants_shares_a_template_but_not_a_near_duplicate():
    a = ("This sequence change replaces glycine with arginine at codon 67 of the MLH1 protein "
         "(p.Gly67Arg). The residue is highly conserved. It is absent from gnomAD.")
    b = ("This sequence change replaces serine with leucine at codon 120 of the MSH2 protein "
         "(p.Ser120Leu). The residue is highly conserved. It is absent from gnomAD.")
    assert jaccard(shingles(template_text(a, ["MLH1"])), shingles(template_text(b, ["MSH2"]))) == 1.0
    assert jaccard(shingles(a), shingles(b)) < 0.5


def test_near_duplicates_cluster_and_unrelated_text_does_not():
    a = "The variant is absent from gnomAD and segregated with disease in three relatives."
    b = a.replace("three", "four")
    c = "Long QT syndrome is diagnosed by a corrected QT interval above 480 milliseconds."
    labels = cluster_documents([a, b, c], 0.5).labels
    assert labels[0] == labels[1] != labels[2]


def test_minhash_estimates_the_real_overlap():
    a = "the variant is absent from population databases and segregated with disease"
    b = a + " in three affected relatives"
    hasher = MinHasher(seed=3)
    estimate = MinHasher.estimate(hasher.signature(shingles(a)), hasher.signature(shingles(b)))
    assert abs(estimate - jaccard(shingles(a), shingles(b))) < 0.15


def test_exact_groups_ignore_spacing_and_case():
    assert exact_group("The  variant is Absent.") == exact_group("the variant is absent.")


def test_clustering_is_the_same_in_a_fresh_process(tmp_path):
    import subprocess
    import sys
    code = ("import json;"
            "from vpdl.slm.text.dedup import cluster_documents;"
            "texts=['a b c d e f g h', 'a b c d e f g x', 'z y x w v u t s'];"
            "print(json.dumps(cluster_documents(texts, 0.4).labels.tolist()))")
    first = cluster_documents(["a b c d e f g h", "a b c d e f g x", "z y x w v u t s"], 0.4)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip().endswith(str(first.labels.tolist()))


# -- tiers and restrictions ---------------------------------------------------------------

def test_tiers_follow_review_status_and_narrative_length():
    narrative = "The variant is absent from gnomAD and segregated with disease in a family."
    assert assign_tier("clinvar_submission_summary", "reviewed by expert panel", "curation", narrative) == 1
    assert assign_tier("clinvar_submission_summary", "criteria provided, single submitter",
                       "clinical testing", narrative) == 2
    assert assign_tier("clinvar_submission_summary", "criteria provided, single submitter",
                       "literature only", narrative) == 3
    assert assign_tier("pubmed", None, None, narrative) == 4
    assert assign_tier("clinvar_submission_summary", "no assertion criteria provided", "", "") == 5


def test_omim_text_is_flagged_restricted_with_a_reason():
    assert restriction("OMIM") and "licence" in restriction("OMIM")
    assert restriction("Synthetic Lab Alpha") is None
