"""Unit tests for dataset acquisition/assembly logic (no network needed).

Run with:  python -m pytest tests/ -v   (or python tests/test_datasets.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (  # noqa: E402
    FINAL_COLUMNS,
    classify_clinical_significance,
    parse_protein_substitution,
    stars_for_review,
)
from src.dataset import (  # noqa: E402
    check_no_position_leakage,
    make_position_group_folds,
)
from src.esm_extractor import validate_and_align  # noqa: E402
from src.external_datasets import model_col_name, parse_mutant_token  # noqa: E402


# --------------------------------------------------------------------------- #
# ClinVar parsing
# --------------------------------------------------------------------------- #
def test_parse_protein_substitution_one_letter():
    assert parse_protein_substitution("R273H", "x") == ("R", 273, "H")


def test_parse_protein_substitution_three_letter_from_name():
    assert parse_protein_substitution("", "TP53 c.818G>A (p.Arg273His)") == ("R", 273, "H")


def test_parse_rejects_synonymous_and_frameshift():
    assert parse_protein_substitution("R273R", "") is None
    assert parse_protein_substitution("", "p.Arg273HisfsTer12") is None
    assert parse_protein_substitution("", "") is None


def test_classify_labelled_paths():
    assert classify_clinical_significance("Pathogenic", stars_for_review(
        "criteria provided, multiple submitters, no conflicts"), min_stars=1) == ("labelled", 1.0)
    assert classify_clinical_significance("Benign/Likely benign", 1, 1) == ("labelled", 0.0)


def test_classify_excludes_conflicting_and_lowstars():
    conflicting = "Criteria provided, conflicting interpretations"
    assert classify_clinical_significance(conflicting, 0, 1) is None
    assert classify_clinical_significance("Pathogenic", 0, 1) is None          # 0 stars
    assert classify_clinical_significance("Uncertain significance", 0, 1)[0] == "vus"


# --------------------------------------------------------------------------- #
# ProteinGym token parsing / column naming
# --------------------------------------------------------------------------- #
def test_parse_mutant_token():
    assert parse_mutant_token("A119D") == ("A", 119, "D")
    assert parse_mutant_token("")is None
    assert parse_mutant_token("A119A") is None            # synonymous
    assert parse_mutant_token("A119*") is None            # stop codon
    assert parse_mutant_token("*119A") is None            # stop codon wt
    assert parse_mutant_token("AC119D") is None           # not a single substitution


def test_model_col_name_normalisation():
    assert model_col_name("PolyPhen2 (HVAR)") == "zs_polyphen2_hvar"
    assert model_col_name("BayesDel (addAF)") == "zs_bayesdel_addaf"


# --------------------------------------------------------------------------- #
# Leakage-safe splitting incl. multi-protein groups
# --------------------------------------------------------------------------- #
def test_folds_are_position_disjoint_single_protein():
    positions = np.repeat(np.arange(1, 51), 4)      # 50 residues x 4 variants
    labels = np.random.default_rng(0).integers(0, 2, len(positions))
    folds = make_position_group_folds(positions, labels, k_folds=5, seed=42)
    assert len(folds) == 5
    all_val = np.concatenate([v for _, v in folds])
    assert sorted(all_val) == list(range(len(positions)))     # partition
    for tr, va in folds:
        check_no_position_leakage(positions[tr], positions[va])


def test_folds_respect_explicit_multi_protein_groups():
    # Same position numbers in two proteins must be independent groups.
    pos = np.array([1, 1, 2, 2, 3, 3, 4, 4] * 2)
    prot = np.array(["P1"] * 8 + ["P2"] * 8)
    labels = np.array([0, 1] * 8)
    groups = np.array([f"{p}:{q}" for p, q in zip(prot, pos)])
    folds = make_position_group_folds(pos, labels, k_folds=2, seed=7, groups=groups)
    for tr, va in folds:
        overlap = set(groups[tr]) & set(groups[va])
        assert not overlap, f"leaked groups: {overlap}"
    # With only 4 distinct raw positions but 8 groups, grouping matters:
    assert len(np.unique(groups)) == 8


# --------------------------------------------------------------------------- #
# Alignment validation
# --------------------------------------------------------------------------- #
def test_validate_and_align_drops_mismatches():
    df = pd.DataFrame({
        "position": [1, 2, 3, 99, -3],
        "wt_aa": ["M", "X", "K", "M", "M"],   # pos2 wrong residue, pos3 valid
        "mut_aa": ["A", "A", "A", "A", "A"],
    })
    out = validate_and_align(df, "MGK")
    assert list(out["position"]) == [1, 3]
    assert set(FINAL_COLUMNS) == {"gene", "position", "wt_aa", "mut_aa",
                                  "hgvs_p", "label", "review_status"}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
