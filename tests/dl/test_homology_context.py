"""Aligner, homology groups/twins, sequence-context policies (MSH6), feature groups."""

from __future__ import annotations

import numpy as np
import pytest


def test_blosum62_is_symmetric_with_known_values():
    from vpdl.dl.homology import BLOSUM62

    assert BLOSUM62[("W", "W")] == 11 and BLOSUM62[("A", "A")] == 4
    assert BLOSUM62[("I", "V")] == 3 and BLOSUM62[("W", "D")] == -4
    assert all(BLOSUM62[(a, b)] == BLOSUM62[(b, a)] for a, b in BLOSUM62)


def test_alignment_maps_through_an_insertion():
    from vpdl.dl.homology import align

    a = "MKTAYIAKQRQISFVKSHFSRQ"
    b = "MKTAYIAKQRWWWQISFVKSHFSRQ"
    alignment = align(a, b)
    assert alignment.identity_shorter == 1.0
    assert alignment.map_position(10) == 10          # before the insertion
    assert alignment.map_position(11) == 14          # after it, shifted by three


def test_homology_groups_and_twins():
    from vpdl.dl.homology import align, homologous_twins, homology_groups

    seqs = {"P40692": "MKTAYIAKQRQISFVKSHFSRQ", "P54278": "MKTAYIAKQRQISFVKSHFSRQ",
            "P43246": "GGGGPPPPWWWWCCCC"}
    groups = homology_groups(seqs)
    assert groups[("P40692", 5)] == groups[("P54278", 5)]           # same family, aligned
    assert groups[("P43246", 5)] != groups[("P40692", 5)]
    alignments = {("P40692", "P54278"): align(seqs["P40692"], seqs["P54278"])}
    twins = homologous_twins([("P54278", 5, "Y", "A")], [("P40692", 5, "Y", "A")], alignments)
    assert twins == {"test_variants": 1, "position_twins": 1, "exact_twins": 1}


def test_identity_clustering_separates_unrelated_sequences():
    from vpdl.dl.homology import cluster_by_identity, pairwise_identity

    rng = np.random.default_rng(0)
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    base = "".join(rng.choice(aa, 200))
    mutated = "".join(c if rng.random() > 0.3 else rng.choice(aa) for c in base)
    unrelated = "".join(rng.choice(aa, 200))
    alignments = pairwise_identity({"a": base, "b": mutated, "c": unrelated})
    clusters = cluster_by_identity(alignments, ["a", "b", "c"], threshold=0.25)
    assert clusters["a"] == clusters["b"] != clusters["c"]


def test_sliding_spans_cover_msh6_with_overlap():
    from vpdl.dl.context import sliding_spans

    spans = sliding_spans(1360)
    assert spans[0] == (0, 1022) and spans[-1][1] == 1360
    covered = np.zeros(1360, dtype=int)
    for start, end in spans:
        covered[start:end] += 1
    assert covered.min() >= 1 and all(e - s <= 1022 for s, e in spans)


@pytest.mark.parametrize("pos0", [0, 400, 700, 1359])
def test_centered_and_asymmetric_windows_contain_the_mutation(pos0):
    from vpdl.dl.context import asymmetric_span, centered_span

    for start, end in (centered_span(1360, pos0), asymmetric_span(1360, pos0)):
        assert start <= pos0 < end and end - start == 1022 and 0 <= start and end <= 1360


def test_asymmetric_window_prefers_the_nearest_terminus():
    from vpdl.dl.context import asymmetric_span

    assert asymmetric_span(1360, 500) == (0, 1022)
    assert asymmetric_span(1360, 1300) == (338, 1360)
    assert asymmetric_span(900, 500) == (0, 900)


def test_full_context_is_refused_for_a_positionally_limited_backbone():
    from vpdl.dl.context import ContextError, site_span

    assert site_span("full", 1360, 1200, supports_full_length=True) == (0, 1360)
    with pytest.raises(ContextError, match="ESM-1b"):
        site_span("full", 1360, 1200, supports_full_length=False)
    assert site_span("full", 934, 10, supports_full_length=False) == (0, 934)


def test_new_dl_feature_columns_resolve_to_their_ablation_groups():
    from vpdl.dl.genomic import GENOMIC_COLUMNS
    from vpdl.dl.structure import VT_STRUCTURE_COLUMNS, WT_STRUCTURE_COLUMNS
    from vpdl.features import group_of

    for column in (*WT_STRUCTURE_COLUMNS, *VT_STRUCTURE_COLUMNS):
        assert group_of(column) == "structure", column
    for column in (*GENOMIC_COLUMNS, "feature_genomic_transition", "feature_genomic_codon_pos1"):
        assert group_of(column) == "genomic", column
    # Existing columns keep their groups.
    assert group_of("feature_alphamissense_score") == "prior_scores"
    assert group_of("feature_gnomad_log10_af") == "gnomad"
    assert group_of("feature_in_domain") == "domains"
