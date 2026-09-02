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


# --------------------------------------------------------------------------- #
# prior inputs for the fine-tune script
# --------------------------------------------------------------------------- #
import importlib.util  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def load_script(name: str):
    """Import a `scripts/` file as a module.

    The module must be registered in ``sys.modules`` *before* exec_module:
    @dataclass resolves its own module through sys.modules, and fails with an
    opaque AttributeError if it is not there yet.
    """
    spec = importlib.util.spec_from_file_location(
        name, PROJECT_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_finetune_script():
    return load_script("finetune_esm_mmr")


def prior_frame(n=8):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "gene": ["MLH1"] * n,
        "am_pathogenicity": rng.random(n),
        "gnomad_pli": np.full(n, 0.99),      # gene-constant
        "gnomad_log10_af": rng.random(n),    # AF-derived
        "acmg_bs1": rng.integers(0, 2, n),   # AF-derived
        "label": rng.integers(0, 2, n),
    })


def test_prior_inputs_standardise_on_train_only():
    """Val and holdout must be transformed with the train partition's
    constants, never their own -- otherwise each split is centred differently
    and the head reads shifted inputs."""
    mod = load_finetune_script()
    ft, ho = prior_frame(8), prior_frame(4)
    out = mod.build_prior_inputs(ft, ho, [0, 1, 2, 3, 4, 5], [6, 7],
                                 drop_gene_constant=True,
                                 af_labels_active=False)
    assert out.train.shape[0] == 6 and out.val.shape[0] == 2
    assert out.holdout.shape[0] == 4
    np.testing.assert_allclose(out.train.mean(axis=0), 0.0, atol=1e-5)
    assert out.mean.shape[0] == out.train.shape[1]


def test_prior_inputs_drop_gene_constant_columns_under_lopo():
    mod = load_finetune_script()
    ft, ho = prior_frame(8), prior_frame(4)
    out = mod.build_prior_inputs(ft, ho, [0, 1, 2, 3], [4, 5],
                                 drop_gene_constant=True,
                                 af_labels_active=False)
    assert "gnomad_pli" not in out.columns


def test_prior_inputs_enforce_the_af_quarantine():
    mod = load_finetune_script()
    ft, ho = prior_frame(8), prior_frame(4)
    with pytest.raises(ValueError, match="quarantine"):
        mod.build_prior_inputs(ft, ho, [0, 1, 2, 3], [4, 5],
                               drop_gene_constant=True, af_labels_active=True)


def test_prior_inputs_column_order_is_identical_across_partitions():
    """A different column order between partitions would silently permute the
    head's inputs at evaluation time."""
    mod = load_finetune_script()
    ft, ho = prior_frame(8), prior_frame(4)
    out = mod.build_prior_inputs(ft, ho, [0, 1, 2, 3], [4, 5],
                                 drop_gene_constant=True,
                                 af_labels_active=False)
    assert out.train.shape[1] == out.val.shape[1] == out.holdout.shape[1]
    assert out.train.shape[1] == len(out.columns)


def test_prior_inputs_are_finite_even_with_all_nan_columns():
    mod = load_finetune_script()
    ft, ho = prior_frame(8), prior_frame(4)
    ft["am_pathogenicity"] = np.nan
    ho["am_pathogenicity"] = np.nan
    out = mod.build_prior_inputs(ft, ho, [0, 1, 2, 3], [4, 5],
                                 drop_gene_constant=True,
                                 af_labels_active=False)
    assert np.isfinite(out.train).all() and np.isfinite(out.holdout).all()


# --------------------------------------------------------------------------- #
# grid driver
# --------------------------------------------------------------------------- #
import json  # noqa: E402


def load_grid_script():
    return load_script("run_stage2b_grid")


def test_cell_is_incomplete_before_anything_runs(tmp_path):
    mod = load_grid_script()
    cell = GridCell("esm+priors", -1, "residual", 42)
    assert not mod.cell_is_complete(tmp_path, cell, "siamese", "lopo")


def test_cell_is_complete_only_with_results_and_predictions(tmp_path):
    """Resumability must not skip a cell whose run died between writing the
    summary and writing the predictions."""
    mod = load_grid_script()
    cell = GridCell("esm+priors", -1, "residual", 42)
    tag = f"siamese_lopo_{cell.slug()}"
    (tmp_path / f"esm_finetune_summary_{tag}.json").write_text(
        json.dumps({"cell": cell.slug()}))
    assert not mod.cell_is_complete(tmp_path, cell, "siamese", "lopo")
    (tmp_path / f"esm_finetune_results_{tag}.csv").write_text("holdout_gene\nMLH1\n")
    assert not mod.cell_is_complete(tmp_path, cell, "siamese", "lopo")
    (tmp_path / f"esm_finetune_predictions_{tag}.csv").write_text("label,prob\n1,0.9\n")
    assert mod.cell_is_complete(tmp_path, cell, "siamese", "lopo")


def test_build_cell_argv_carries_every_axis():
    mod = load_grid_script()
    args = mod.parse_args(["--tiers", "1", "--esm_model", "facebook/esm2_t6_8M_UR50D"])
    cell = GridCell("esm+priors", 0, "off", 43, fusion="gatewave")
    argv = mod.build_cell_argv(args, cell)
    assert argv[argv.index("--branch") + 1] == "esm+priors"
    assert argv[argv.index("--n_unfrozen_layers") + 1] == "0"
    assert argv[argv.index("--seed") + 1] == "43"
    assert argv[argv.index("--fusion") + 1] == "gatewave"
    assert argv[argv.index("--cell_slug") + 1] == cell.slug()
    assert "--no-use_pllr" in argv, "pllr_mode 'off' must disable the term"


def test_build_cell_argv_uses_pllr_mode_when_the_term_is_on():
    mod = load_grid_script()
    args = mod.parse_args(["--tiers", "1"])
    argv = mod.build_cell_argv(args, GridCell("esm", -1, "residual", 42))
    assert "--no-use_pllr" not in argv
    assert argv[argv.index("--pllr_mode") + 1] == "residual"


def test_build_cell_argv_is_accepted_by_the_finetune_cli():
    """The driver and the script must not drift apart: every flag the driver
    emits has to parse against the script's own argument parser."""
    grid = load_grid_script()
    fine = load_finetune_script()
    args = grid.parse_args(["--tiers", "1"])
    for cell in cells_for(sorted(grid.TIERS)):
        argv = grid.build_cell_argv(args, cell)
        parsed = fine.parse_args(argv[2:])   # drop [python, script.py]
        assert parsed.branch == cell.branch
        assert parsed.n_unfrozen_layers == cell.n_unfrozen_layers
        assert parsed.seed == cell.seed
        assert parsed.cell_slug == cell.slug()
        assert parsed.use_pllr == (cell.pllr_mode != "off")


def test_aggregate_merges_every_cell_result(tmp_path):
    mod = load_grid_script()
    for slug, auc in [("a", 0.9), ("b", 0.8)]:
        (tmp_path / f"esm_finetune_results_siamese_lopo_{slug}.csv").write_text(
            f"holdout_gene,roc_auc,cell_slug\nMLH1,{auc},{slug}\n")
    df = mod.aggregate(tmp_path)
    assert len(df) == 2
    assert set(df["cell_slug"]) == {"a", "b"}


def test_aggregate_of_an_empty_dir_is_empty(tmp_path):
    mod = load_grid_script()
    assert mod.aggregate(tmp_path).empty


def test_output_tag_includes_the_holdout_gene():
    """A holdout tag without the gene collides across genes, and -- because
    the driver's resume check looks for these exact filenames -- made every
    `--eval holdout` cell re-run on restart instead of being skipped."""
    from src.finetune_grid import output_tag
    cell = GridCell("esm+priors", 0, "off", 42)
    assert output_tag("siamese", "lopo", cell.slug()) == f"siamese_lopo_{cell.slug()}"
    assert output_tag("siamese", "holdout", cell.slug(), "MSH2") == \
        f"siamese_holdout_MSH2_{cell.slug()}"
    with pytest.raises(ValueError, match="holdout_gene"):
        output_tag("siamese", "holdout", cell.slug())


@pytest.mark.parametrize("eval_mode,gene", [("lopo", None), ("holdout", "MSH2")])
def test_cell_is_complete_matches_the_names_the_finetune_script_writes(
        tmp_path, eval_mode, gene):
    """Anti-drift: the driver's resume check and the script's output naming
    must be the same construction, not two that happen to agree under lopo."""
    grid, fine = load_grid_script(), load_finetune_script()
    cell = GridCell("esm+priors", 0, "off", 42)
    tag = fine.output_tag("siamese", eval_mode, cell.slug(), gene)
    for kind, ext, body in (("summary", "json", "{}"),
                            ("results", "csv", "holdout_gene\nMSH2\n"),
                            ("predictions", "csv", "label,prob\n1,0.9\n")):
        (tmp_path / f"esm_finetune_{kind}_{tag}.{ext}").write_text(body)
    assert grid.cell_is_complete(tmp_path, cell, "siamese", eval_mode, gene)
