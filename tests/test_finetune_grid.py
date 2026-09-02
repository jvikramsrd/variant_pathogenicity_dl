"""Grid-cell identity, tier composition, and the AF label/feature quarantine."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.finetune_grid import TIERS, GridCell, cells_for  # noqa: E402
from src.transfer import AF_DERIVED_PRIOR_COLS, assert_af_quarantine  # noqa: E402


def test_af_quarantine_raises_when_labels_and_features_are_both_active():
    """Minting benign labels from allele frequency while feeding allele
    frequency as a feature makes acmg_bs1 == label by construction -- the
    same target leakage as Finding 2, in the paper that reports Finding 2."""
    with pytest.raises(ValueError, match="quarantine"):
        assert_af_quarantine(["am_pathogenicity", "acmg_bs1"], af_labels_active=True)


def test_af_quarantine_passes_when_af_columns_are_removed():
    assert_af_quarantine(["am_pathogenicity", "in_domain"], af_labels_active=True)


def test_af_quarantine_allows_af_features_without_af_labels():
    assert_af_quarantine(list(AF_DERIVED_PRIOR_COLS), af_labels_active=False)


def test_af_derived_cols_cover_every_frequency_column():
    assert set(AF_DERIVED_PRIOR_COLS) == {
        "gnomad_log10_af", "acmg_ba1", "acmg_bs1", "acmg_pm2"}


def test_cell_slug_is_deterministic_and_distinguishes_every_axis():
    base = dict(branch="esm+priors", n_unfrozen_layers=-1,
                pllr_mode="residual", seed=42, fusion="concat")
    a = GridCell(**base)
    assert a.slug() == GridCell(**base).slug(), "slug must be deterministic"
    for field, other in [("branch", "esm"), ("n_unfrozen_layers", 0),
                         ("pllr_mode", "off"), ("seed", 43),
                         ("fusion", "gatewave")]:
        assert GridCell(**{**base, field: other}).slug() != a.slug(), \
            f"slug must vary with {field}"


def test_cell_slug_is_filesystem_safe():
    slug = GridCell(branch="esm+priors", n_unfrozen_layers=-1,
                    pllr_mode="residual", seed=42).slug()
    assert all(c.isalnum() or c in "-_" for c in slug), slug


def test_unknown_branch_raises():
    with pytest.raises(ValueError, match="branch"):
        GridCell(branch="priors_only", n_unfrozen_layers=0, pllr_mode="off")


def test_tier_one_is_the_headline_fair_fight_with_three_seeds():
    cells = TIERS["1"]
    assert {c.seed for c in cells} == {42, 43, 44}
    assert {c.branch for c in cells} == {"esm+priors"}
    assert {c.n_unfrozen_layers for c in cells} == {-1, 0}
    assert len(cells) == 6


def test_cells_for_deduplicates_and_preserves_tier_order():
    cells = cells_for(["1", "2", "1"])
    assert len(cells) == len({c.slug() for c in cells})
    assert cells[0].tier == "1"


def test_cells_for_rejects_an_unknown_tier():
    with pytest.raises(ValueError, match="unknown tier"):
        cells_for(["1", "99"])


def test_every_tier_cell_has_a_unique_slug():
    everything = cells_for(sorted(TIERS))
    assert len(everything) == len({c.slug() for c in everything})
