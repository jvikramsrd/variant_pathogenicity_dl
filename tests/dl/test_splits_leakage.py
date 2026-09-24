"""Split isolation for every scheme, and the leakage gate that stops training."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dl_helpers import PANEL


def _work(table):
    return table.assign(_train=table["label__clinvar"], _eval=table["label__clinvar"])


def test_logo_folds_are_the_original_run_cell_folds(table):
    from vpdl.dl.splits import make_folds

    work = _work(table)
    folds = make_folds(work, "logo")
    assert [f.name for f in folds] == sorted(PANEL)
    for fold in folds:
        gene = fold.name
        expected_train = np.where((work["gene"] != gene) & work["_train"].notna())[0]
        expected_test = np.where((work["gene"] == gene) & work["_eval"].notna())[0]
        np.testing.assert_array_equal(fold.train_positions, expected_train)
        np.testing.assert_array_equal(fold.test_positions, expected_test)


def test_mandatory_leave_one_gene_out_rotations(table):
    from vpdl.dl.splits import make_folds

    work = _work(table)
    for fold in make_folds(work, "logo"):
        train_genes = set(work.iloc[fold.train_positions]["gene"])
        assert train_genes == set(PANEL) - {fold.name}      # e.g. MLH1+MSH2+MSH6 -> PMS2


def test_family_split_is_gene_and_family_disjoint(table):
    from vpdl.dl.splits import make_folds

    work = _work(table)
    folds = make_folds(work, "family")
    assert {f.name for f in folds} == {"family:MutL", "family:MutS"}
    for fold in folds:
        train, test = work.iloc[fold.train_positions], work.iloc[fold.test_positions]
        assert not set(train["gene"]) & set(test["gene"])
    mutl = next(f for f in folds if f.name == "family:MutL")
    assert set(mutl.test_genes) == {"MLH1", "PMS2"}


def test_random_debug_split_is_residue_group_disjoint(table):
    from vpdl.dl.splits import make_folds
    from vpdl.splits import group_keys

    work = _work(table)
    groups = group_keys(work)
    for fold in make_folds(work, "random_debug"):
        assert not set(groups[fold.train_positions]) & set(groups[fold.test_positions])


def test_logo_purged_removes_homologous_training_rows(table, sequences):
    from vpdl.dl.canonical import build_canonical
    from vpdl.dl.splits import make_folds

    canonical, _, _ = build_canonical(table, sequences)
    work = _work(canonical)
    plain = {f.name: f for f in make_folds(work, "logo")}
    purged = {f.name: f for f in make_folds(work, "logo_purged")}
    clusters = work["cluster_id"].to_numpy()
    for name, fold in purged.items():
        assert fold.purged == len(plain[name].train_positions) - len(fold.train_positions)
        assert not set(clusters[fold.train_positions]) & set(clusters[fold.test_positions])


def test_gate_refuses_a_variant_on_both_sides(table):
    from vpdl.dl.leakage import LeakageError, leakage_gate
    from vpdl.dl.splits import Fold

    work = _work(table)
    leaky = Fold("MLH1", np.arange(0, 20), np.arange(10, 30), ("MLH1",))
    with pytest.raises(LeakageError, match="both sides"):
        leakage_gate(work, [leaky], "logo", ["feature_in_domain"], ["clinvar"])


def test_gate_refuses_functional_values_as_features(table):
    from vpdl.dl.leakage import LeakageError, leakage_gate
    from vpdl.dl.splits import make_folds

    work = _work(table).assign(dms_score=0.0, feature_dms_score=0.0)
    with pytest.raises(LeakageError, match="functional"):
        leakage_gate(work, make_folds(work), "logo", ["feature_dms_score"], ["clinvar"])


def test_gate_refuses_functional_validation_under_a_leaky_split(table):
    from vpdl.dl.leakage import LeakageError, leakage_gate
    from vpdl.dl.splits import make_folds

    work = _work(table)
    folds = make_folds(work, "random_debug")
    with pytest.raises(LeakageError, match="gene-disjoint"):
        leakage_gate(work, folds, "random_debug", ["feature_in_domain"], ["clinvar"],
                     validating_functional=True, functional_keys={"P40692:1:A>V"})


def test_gate_refuses_label_derived_features_for_the_training_source(table):
    from vpdl.dl.leakage import LeakageError, leakage_gate
    from vpdl.dl.splits import make_folds

    work = _work(table).assign(feature_dms_bin=0.0)
    with pytest.raises(LeakageError):
        leakage_gate(work, make_folds(work), "logo", ["feature_dms_bin"], ["clinvar", "pg_dms"])


def test_full_report_measures_paralog_similarity_and_twins(table, sequences):
    from vpdl.dl.leakage import run_checks
    from vpdl.dl.splits import make_folds

    work = _work(table)
    # Make MLH1 and PMS2 identical in their first 40 residues, so PMS2 variants
    # have homologous twins in MLH1 under leave-one-gene-out.
    seqs = dict(sequences)
    seqs[PANEL["PMS2"]] = seqs[PANEL["MLH1"]][:40] + seqs[PANEL["PMS2"]][40:]
    report = run_checks(work, make_folds(work), "logo", ["feature_in_domain"], ["clinvar"],
                        sequences=seqs)
    assert not report.critical
    similarity = [f for f in report.findings if f.check == "sequence_similarity"
                  and f.details.get("fold") == "PMS2"]
    assert similarity and similarity[0].details["position_twins"] > 0
    assert "sequence_similarity" in report.to_markdown()


def test_existing_style_table_passes_the_gate(table):
    """The gate must not break the arms that already have published numbers."""
    from vpdl.dl.leakage import leakage_gate
    from vpdl.dl.splits import make_folds

    work = _work(table)
    columns = [c for c in table.columns if c.startswith("feature_")]
    report = leakage_gate(work, make_folds(work), "logo", columns, ["clinvar"])
    assert not report.critical
