"""Unit tests for the new data-collection / DL modules (no network needed).

Covers:
* src/mavedb.py          -- hgvs_pro parsing
* src/cimra.py            -- OddsPath CSV loading, ACMG-strength classification
* src/mmr_dataset.py      -- functional-assay attach helpers
* src/esm_finetune.py     -- windowed example construction (pure logic, no
  model load: the actual fine-tuning forward/backward pass is exercised as a
  live smoke test, not here, since it requires downloading an ESM-2 checkpoint)

Run with:  python -m pytest tests/test_new_data_sources.py -v
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mavedb import parse_hgvs_pro  # noqa: E402
from src.cimra import (  # noqa: E402
    TAVTIGIAN_ODDSPATH_THRESHOLDS,
    attach_cimra_features,
    classify_oddspath_strength,
    load_cimra_oddspath,
)
from src.mmr_dataset import FUNCTIONAL_ASSAY_VALIDATION_ONLY_COLS  # noqa: E402
import src.esm_finetune as esm_finetune  # noqa: E402
import src.transfer as transfer  # noqa: E402
from src.esm_finetune import (  # noqa: E402
    FINETUNE_CHECKPOINT_FORMAT,
    ESMFineTuneClassifier,
    FineTuneExample,
    build_examples,
    load_finetuned_model,
    save_finetuned,
)
from scripts.build_cluster_split import (  # noqa: E402
    assign_cluster_split,
    parse_cluster_tsv,
)
from src.gnomad import (  # noqa: E402
    GENE_CONSTRAINT_COLUMNS,
    GNOMAD_FEATURE_COLUMNS,
    add_frequency_flags,
    attach_gene_constraint,
    join_gnomad_features,
)
from src.structure import ALPHAFOLD_FEATURE_COLUMNS, attach_alphafold_features, plddt_bin  # noqa: E402
from src.interpro import (  # noqa: E402
    INTERPRO_FEATURE_COLUMNS,
    attach_interpro_features,
    parse_interpro_intervals,
)
from src.external_datasets import attach_functional_site_features  # noqa: E402


# --------------------------------------------------------------------------- #
# MaveDB hgvs_pro parsing
# --------------------------------------------------------------------------- #
def test_parse_hgvs_pro_basic():
    assert parse_hgvs_pro("p.Met1Ala") == ("M", 1, "A")
    assert parse_hgvs_pro("p.Arg273His") == ("R", 273, "H")


def test_parse_hgvs_pro_rejects_synonymous_stop_and_na():
    assert parse_hgvs_pro("p.Met1=") is None
    assert parse_hgvs_pro("p.Met1Ter") is None
    assert parse_hgvs_pro("NA") is None
    assert parse_hgvs_pro("") is None
    assert parse_hgvs_pro("p.Met1Met") is None       # wt == mut


# --------------------------------------------------------------------------- #
# CIMRA OddsPath classification
# --------------------------------------------------------------------------- #
def test_classify_oddspath_strength_boundaries():
    t = TAVTIGIAN_ODDSPATH_THRESHOLDS
    assert classify_oddspath_strength(t["PS3_very_strong"]) == "PS3_very_strong"
    assert classify_oddspath_strength(t["PS3_supporting"]) == "PS3_supporting"
    assert classify_oddspath_strength(1.0) == "indeterminate"
    assert classify_oddspath_strength(t["BS3_very_strong"]) == "BS3_very_strong"
    assert classify_oddspath_strength(float("nan")) == "indeterminate"


def test_load_cimra_oddspath_excludes_splicing_and_computes_columns():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "cimra.csv"
        pd.DataFrame([
            {"gene": "pms2", "position": 134, "wt_aa": "A", "mut_aa": "V",
             "cimra_oddspath": 400.0, "mechanism": "missense"},
            {"gene": "PMS2", "position": 200, "wt_aa": "G", "mut_aa": "D",
             "cimra_oddspath": 0.02, "mechanism": "missense"},
            {"gene": "PMS2", "position": 300, "wt_aa": "L", "mut_aa": "P",
             "cimra_oddspath": 5.0, "mechanism": "splicing"},
        ]).to_csv(path, index=False)
        df = load_cimra_oddspath(path)
    assert len(df) == 2                               # splicing row dropped
    assert set(df["gene"]) == {"PMS2"}                 # normalised to upper
    assert df.loc[df["position"] == 134, "cimra_acmg_strength"].iloc[0] == "PS3_very_strong"
    assert df.loc[df["position"] == 200, "cimra_acmg_strength"].iloc[0] == "BS3_strong"
    assert np.isclose(df.loc[df["position"] == 134, "cimra_log10_oddspath"].iloc[0],
                      np.log10(400.0))


def test_load_cimra_oddspath_missing_file_raises():
    try:
        load_cimra_oddspath(Path("/nonexistent/path/cimra.csv"))
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised


def test_load_cimra_oddspath_missing_columns_raises():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "bad.csv"
        pd.DataFrame([{"gene": "PMS2", "position": 1}]).to_csv(path, index=False)
        try:
            load_cimra_oddspath(path)
            raised = False
        except ValueError:
            raised = True
        assert raised


def test_attach_cimra_features_leaves_unmatched_rows_indeterminate():
    cimra_df = pd.DataFrame([{
        "gene": "PMS2", "position": 134, "wt_aa": "A", "mut_aa": "V",
        "cimra_oddspath": 400.0, "cimra_log10_oddspath": np.log10(400.0),
        "cimra_acmg_strength": "PS3_very_strong",
    }])
    master = pd.DataFrame([
        {"gene": "PMS2", "position": 134, "wt_aa": "A", "mut_aa": "V"},
        {"gene": "PMS2", "position": 999, "wt_aa": "A", "mut_aa": "V"},
    ])
    merged = attach_cimra_features(master, cimra_df)
    assert merged.loc[0, "cimra_acmg_strength"] == "PS3_very_strong"
    assert merged.loc[1, "cimra_acmg_strength"] == "indeterminate"
    assert pd.isna(merged.loc[1, "cimra_oddspath"])


def test_functional_assay_cols_excluded_from_transfer_priors():
    # Non-circularity contract (PROJECT_PLAN.md Phase 2): none of these may
    # ever be a leakage-safe ESM-branch prior column.
    from src.transfer import TRANSFER_PRIOR_COLS
    assert not (set(FUNCTIONAL_ASSAY_VALIDATION_ONLY_COLS) & set(TRANSFER_PRIOR_COLS))


# --------------------------------------------------------------------------- #
# ESM fine-tuning: windowed example construction (pure logic)
# --------------------------------------------------------------------------- #
def test_build_examples_wt_site_window_and_position():
    seq = "M" + "A" * 2000 + "K"     # 2002 aa, forces windowing
    df = pd.DataFrame([{"gene": "TEST", "position": 1001, "wt_aa": "A",
                        "mut_aa": "V", "label": 1.0, "label_weight": 2.0}])
    examples = build_examples(df, {"TEST": seq}, mode="wt_site", max_residues=1022)
    assert len(examples) == 1
    ex = examples[0]
    assert isinstance(ex, FineTuneExample)
    assert ex.wt_seq[ex.wt_pos0] == "A"
    assert len(ex.wt_seq) <= 1022
    assert ex.mut_seq is None
    assert ex.label == 1.0
    assert ex.weight == 2.0


def test_build_examples_siamese_mutates_local_window():
    seq = "MAAAAAAAAAAAAAAK"
    df = pd.DataFrame([{"gene": "TEST", "position": 5, "wt_aa": "A", "mut_aa": "D",
                        "label": 0.0}])
    examples = build_examples(df, {"TEST": seq}, mode="siamese", max_residues=1022)
    ex = examples[0]
    assert ex.wt_seq[ex.wt_pos0] == "A"
    assert ex.mut_seq[ex.mut_pos0] == "D"
    # Every other residue in the window is unchanged.
    assert ex.mut_seq[:ex.mut_pos0] == ex.wt_seq[:ex.wt_pos0]
    assert ex.mut_seq[ex.mut_pos0 + 1:] == ex.wt_seq[ex.wt_pos0 + 1:]


def test_build_examples_drops_wt_mismatch_and_unknown_gene():
    seq = "MAAAAK"
    df = pd.DataFrame([
        {"gene": "TEST", "position": 2, "wt_aa": "A", "mut_aa": "V", "label": 1.0},
        {"gene": "TEST", "position": 2, "wt_aa": "W", "mut_aa": "V", "label": 1.0},  # wrong wt
        {"gene": "UNKNOWN", "position": 2, "wt_aa": "A", "mut_aa": "V", "label": 1.0},
    ])
    examples = build_examples(df, {"TEST": seq}, mode="wt_site")
    assert len(examples) == 1


# --------------------------------------------------------------------------- #
# Fine-tuned model persistence (stage 2b): the backbone weights must be
# saved, not discarded on process exit. Stubs stand in for the ESM download.
# --------------------------------------------------------------------------- #
def _stub_finetune_model(**overrides):
    """An ESMFineTuneClassifier shell with just the attributes config() reads."""
    m = ESMFineTuneClassifier.__new__(ESMFineTuneClassifier)
    m.model_name = overrides.get("model_name", "facebook/esm2_t6_8M_UR50D")
    m.mode = overrides.get("mode", "siamese")
    m.n_unfrozen_layers = overrides.get("n_unfrozen_layers", -1)
    m.hidden_dim = overrides.get("hidden_dim", 256)
    m.dropout = overrides.get("dropout", 0.15)
    m.pllr_mode = overrides.get("pllr_mode", "residual")
    m.use_pllr = m.pllr_mode != "off"
    return m


def test_config_roundtrips_every_constructor_arg():
    cfg = _stub_finetune_model(mode="wt_site", n_unfrozen_layers=4,
                               pllr_mode="off").config()
    assert cfg == {
        "model_name": "facebook/esm2_t6_8M_UR50D", "mode": "wt_site",
        "n_unfrozen_layers": 4, "hidden_dim": 256, "dropout": 0.15,
        "pllr_mode": "off",
    }


def test_save_finetuned_payload_is_self_describing():
    captured = {}

    def fake_save_checkpoint(path, model, mean, scale, feature_columns,
                             cfg_dict, extra=None):
        captured.update(path=path, cfg=cfg_dict, extra=extra)
        return path

    orig = transfer.save_checkpoint
    transfer.save_checkpoint = fake_save_checkpoint
    try:
        save_finetuned(Path("x.pt"), _stub_finetune_model(pllr_mode="off"),
                       threshold=0.37, max_residues=1022,
                       metrics={"roc_auc": 0.9, "mcc": 0.5},
                       extra={"holdout_gene": "MSH2", "seed": 42})
    finally:
        transfer.save_checkpoint = orig

    # Every constructor arg needed to rebuild the model, plus the crop width.
    assert captured["cfg"]["hidden_dim"] == 256
    assert captured["cfg"]["pllr_mode"] == "off"
    assert captured["cfg"]["max_residues"] == 1022
    # Threshold + metrics + provenance travel with the weights.
    assert captured["extra"]["format"] == FINETUNE_CHECKPOINT_FORMAT
    assert captured["extra"]["threshold"] == 0.37
    assert captured["extra"]["metrics"] == {"roc_auc": 0.9, "mcc": 0.5}
    assert captured["extra"]["holdout_gene"] == "MSH2"


def test_load_finetuned_model_rebuilds_from_stored_config():
    payload = {
        "config": {"model_name": "facebook/esm2_t6_8M_UR50D", "mode": "wt_site",
                   "n_unfrozen_layers": 2, "hidden_dim": 128, "dropout": 0.2,
                   "use_pllr": False, "max_residues": 512},
        "model_state_dict": {"sentinel": 1},
        "threshold": 0.41,
    }
    seen = {}

    class FakeModel:
        def __init__(self, **kw):
            seen["init"] = kw

        def load_state_dict(self, sd, strict=True):
            seen["state_dict"], seen["strict"] = sd, strict

        def to(self, dev):
            seen["device"] = dev
            return self

        def eval(self):
            seen["eval"] = True
            return self

    orig_load = transfer.load_checkpoint
    orig_cls = esm_finetune.ESMFineTuneClassifier
    transfer.load_checkpoint = lambda p: payload
    esm_finetune.ESMFineTuneClassifier = FakeModel
    try:
        model, got = load_finetuned_model(Path("x.pt"), device="cpu")
    finally:
        transfer.load_checkpoint = orig_load
        esm_finetune.ESMFineTuneClassifier = orig_cls

    assert isinstance(model, FakeModel)
    assert got is payload
    # Rebuilt with the checkpoint's own config, not defaults.
    # Rebuilt with the checkpoint's own config, and the legacy `use_pllr:
    # False` is migrated to pllr_mode="off". A pre-pllr_mode checkpoint
    # described the concat behaviour, never the residual one, so mapping a
    # legacy True to "residual" would silently change the loaded architecture.
    assert seen["init"] == {
        "model_name": "facebook/esm2_t6_8M_UR50D", "mode": "wt_site",
        "n_unfrozen_layers": 2, "hidden_dim": 128, "dropout": 0.2,
        "pllr_mode": "off",
    }
    assert seen["state_dict"] == {"sentinel": 1}
    assert seen["device"] == "cpu"
    assert seen["eval"] is True


def test_load_finetuned_model_migrates_legacy_use_pllr_true_to_concat():
    """A checkpoint predating pllr_mode described the concat architecture."""
    payload = {
        "config": {"model_name": "facebook/esm2_t6_8M_UR50D", "mode": "siamese",
                   "n_unfrozen_layers": -1, "hidden_dim": 256, "dropout": 0.15,
                   "use_pllr": True},
        "model_state_dict": {},
    }
    seen = {}

    class FakeModel:
        def __init__(self, **kw):
            seen["init"] = kw

        def load_state_dict(self, sd, strict=True):
            pass

        def eval(self):
            pass

    orig_load = transfer.load_checkpoint
    orig_cls = esm_finetune.ESMFineTuneClassifier
    transfer.load_checkpoint = lambda p: payload
    esm_finetune.ESMFineTuneClassifier = FakeModel
    try:
        load_finetuned_model(Path("x.pt"))
    finally:
        transfer.load_checkpoint = orig_load
        esm_finetune.ESMFineTuneClassifier = orig_cls

    assert seen["init"]["pllr_mode"] == "concat"
    assert "use_pllr" not in seen["init"]


# --------------------------------------------------------------------------- #
# MMseqs2 cluster split (pure-Python parsing/assignment, no binary needed)
# --------------------------------------------------------------------------- #
def test_parse_cluster_tsv():
    with tempfile.TemporaryDirectory() as td:
        tsv = Path(td) / "x_cluster.tsv"
        tsv.write_text("P1\tP1\nP1\tP2\nP3\tP3\nP4\tP4\nP4\tP5\n")
        mapping = parse_cluster_tsv(tsv)
    assert mapping == {"P1": "P1", "P2": "P1", "P3": "P3", "P4": "P4", "P5": "P4"}


def test_assign_cluster_split_keeps_clusters_together():
    mapping = {"P1": "P1", "P2": "P1", "P3": "P3", "P4": "P4", "P5": "P4"}
    split = assign_cluster_split(mapping, val_fraction=0.4, seed=0)
    assert split["P1"] == split["P2"]                  # same cluster, same side
    assert split["P4"] == split["P5"]
    assert set(split.values()) <= {"train", "val"}


# --------------------------------------------------------------------------- #
# Regression: joining real gnomAD data onto a table pre-seeded with NaN
# placeholder columns of the same name (assemble_master does this when
# include_gnomad=False) must overwrite, not _x/_y-suffix-collide.
# --------------------------------------------------------------------------- #
def test_join_gnomad_features_overwrites_preexisting_placeholder_columns():
    placeholder_master = pd.DataFrame([{
        "gene": "MLH1", "position": 5, "wt_aa": "A", "mut_aa": "V",
        **{c: float("nan") for c in GNOMAD_FEATURE_COLUMNS},
    }])
    real_gnomad = add_frequency_flags(pd.DataFrame([{
        "gene": "MLH1", "position": 5, "wt_aa": "A", "mut_aa": "V",
        "gnomad_af_joint": 0.02, "gnomad_af_genome": 0.02, "gnomad_af_exome": 0.02,
        "gnomad_ac_joint": 200, "gnomad_an_joint": 10000,
    }]))
    merged = join_gnomad_features(placeholder_master, real_gnomad)
    assert not any(c.endswith(("_x", "_y")) for c in merged.columns)
    assert merged.loc[0, "acmg_bs1"] == 1
    assert not pd.isna(merged.loc[0, "gnomad_af_joint"])


def test_attach_gene_constraint_overwrites_preexisting_placeholder_columns():
    placeholder_master = pd.DataFrame([{
        "gene": "MLH1", "position": 5,
        **{c: float("nan") for c in GENE_CONSTRAINT_COLUMNS},
    }])
    merged = attach_gene_constraint(
        placeholder_master,
        {"MLH1": {"gnomad_pli": 0.48, "gnomad_oe_lof": 0.40, "gnomad_oe_mis": 0.94,
                  "gnomad_mis_z": 0.68, "gnomad_syn_z": -0.29}})
    assert not any(c.endswith(("_x", "_y")) for c in merged.columns)
    assert merged.loc[0, "gnomad_mis_z"] == 0.68


# --------------------------------------------------------------------------- #
# gnomAD gene-level constraint
# --------------------------------------------------------------------------- #
def test_attach_gene_constraint_broadcasts_and_fills_unknown_gene():
    constraint = {"MSH2": {"gnomad_pli": 0.0003, "gnomad_oe_lof": 0.47,
                           "gnomad_oe_mis": 1.30, "gnomad_mis_z": -4.31,
                           "gnomad_syn_z": -1.81}}
    df = pd.DataFrame([{"gene": "MSH2", "position": 1}, {"gene": "MSH2", "position": 2},
                       {"gene": "OTHER", "position": 1}])
    merged = attach_gene_constraint(df, constraint)
    assert merged.loc[0, "gnomad_mis_z"] == merged.loc[1, "gnomad_mis_z"] == -4.31
    assert pd.isna(merged.loc[2, "gnomad_mis_z"])
    assert set(GENE_CONSTRAINT_COLUMNS) <= set(merged.columns)


# --------------------------------------------------------------------------- #
# AlphaFold structural confidence
# --------------------------------------------------------------------------- #
def test_plddt_bin_boundaries():
    assert plddt_bin(10.0) == "very_low"
    assert plddt_bin(49.99) == "very_low"
    assert plddt_bin(50.0) == "low"
    assert plddt_bin(70.0) == "confident"
    assert plddt_bin(90.0) == "very_high"
    assert plddt_bin(100.0) == "very_high"


def test_attach_alphafold_features_merge():
    af_panel = pd.DataFrame([
        {"uniprot_id": "P1", "position": 5, "af_plddt": 30.0,
         "af_plddt_bin": "very_low", "af_disordered": 1},
        {"uniprot_id": "P1", "position": 6, "af_plddt": 95.0,
         "af_plddt_bin": "very_high", "af_disordered": 0},
    ])
    master = pd.DataFrame([{"uniprot_id": "P1", "position": 5},
                           {"uniprot_id": "P1", "position": 999}])
    merged = attach_alphafold_features(master, af_panel)
    assert merged.loc[0, "af_disordered"] == 1
    assert pd.isna(merged.loc[1, "af_plddt"])
    assert set(ALPHAFOLD_FEATURE_COLUMNS) <= set(merged.columns)


# --------------------------------------------------------------------------- #
# InterPro domain/family intervals
# --------------------------------------------------------------------------- #
def test_parse_interpro_intervals_filters_entry_types_and_flattens_fragments():
    entries = [
        {"metadata": {"accession": "IPR001", "name": "MutS domain", "type": "domain",
                      "member_databases": {"pfam": {"PF001": "x"}}},
         "proteins": [{"entry_protein_locations": [
             {"fragments": [{"start": 10, "end": 50}]}]}]},
        {"metadata": {"accession": "IPRXXX", "name": "unintegrated signature",
                      "type": "unintegrated"},
         "proteins": [{"entry_protein_locations": [
             {"fragments": [{"start": 1, "end": 5}]}]}]},
    ]
    df = parse_interpro_intervals(entries, "P1")
    assert len(df) == 1                                # unintegrated type dropped
    assert df.iloc[0]["accession"] == "IPR001"
    assert df.iloc[0]["start"] == 10 and df.iloc[0]["end"] == 50


def test_attach_interpro_features_flags_positions_inside_interval():
    intervals = pd.DataFrame([{"uniprot_id": "P1", "accession": "IPR001",
                               "name": "MutS domain", "entry_type": "domain",
                               "member_databases": "pfam", "start": 10, "end": 50}])
    master = pd.DataFrame([{"uniprot_id": "P1", "position": 20},
                           {"uniprot_id": "P1", "position": 5}])
    merged = attach_interpro_features(master, intervals)
    assert merged.loc[0, "in_interpro_domain"] == 1
    assert merged.loc[1, "in_interpro_domain"] == 0
    assert set(INTERPRO_FEATURE_COLUMNS) <= set(merged.columns)


# --------------------------------------------------------------------------- #
# UniProt point features (active/binding site, PTM, disulfide)
# --------------------------------------------------------------------------- #
def test_attach_functional_site_features_merges_and_lists_types():
    sites = pd.DataFrame([
        {"uniprot_id": "P1", "position": 42, "feature_type": "Active site", "description": ""},
        {"uniprot_id": "P1", "position": 42, "feature_type": "Binding site", "description": ""},
    ])
    master = pd.DataFrame([{"uniprot_id": "P1", "position": 42},
                           {"uniprot_id": "P1", "position": 43}])
    merged = attach_functional_site_features(master, sites)
    assert merged.loc[0, "is_functional_site"] == 1
    assert merged.loc[0, "functional_site_types"] == "Active site|Binding site"
    assert merged.loc[1, "is_functional_site"] == 0


# --------------------------------------------------------------------------- #
# Every newly acquired source must actually reach the DL model's prior branch
# --------------------------------------------------------------------------- #
def test_new_prior_columns_are_wired_into_transfer_prior_cols():
    from src.transfer import TRANSFER_PRIOR_COLS
    expected = set(GENE_CONSTRAINT_COLUMNS) | set(ALPHAFOLD_FEATURE_COLUMNS) \
        | set(INTERPRO_FEATURE_COLUMNS) | {"is_functional_site"}
    assert expected <= set(TRANSFER_PRIOR_COLS)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
