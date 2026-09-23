"""Structure/genomic/functional inputs, feature store, interface schema, run_cell end to end,
CLI parsing and run tracking."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import PANEL

ROOT = Path(__file__).resolve().parents[2]


# -- structure -------------------------------------------------------------------------

def _helix_pdb(path, n=12):
    """An ideal alpha-helix backbone (+CB) as a PDB, n alanines."""
    lines = []
    atom = 1
    for i in range(n):
        angle = np.radians(100 * i)
        rise = 1.5 * i
        for name, radius, dz in (("N", 1.55, -0.5), ("CA", 2.3, 0.0), ("C", 1.6, 0.6),
                                 ("O", 1.3, 1.4), ("CB", 3.3, -0.3)):
            x, y, z = radius * np.cos(angle), radius * np.sin(angle), rise + dz
            lines.append(f"ATOM  {atom:5d}  {name:<3} ALA A{i + 1:4d}    "
                         f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{70.0 + i:6.2f}           {name[0]}")
            atom += 1
    Path(path).write_text("\n".join(lines) + "\nEND\n")


def test_structure_features_from_a_small_model(tmp_path):
    from vpdl.dl.structure import WT_STRUCTURE_COLUMNS, parse_pdb, residue_features

    _helix_pdb(tmp_path / "X.pdb")
    residues = parse_pdb(tmp_path / "X.pdb")
    assert len(residues) == 12 and residues[0].aa == "A"
    features = residue_features(residues)
    assert list(features["position"]) == list(range(1, 13))
    assert set(WT_STRUCTURE_COLUMNS) <= set(features.columns)
    assert features["feature_struct_plddt"].iloc[0] == pytest.approx(70.0)
    assert features["feature_struct_rsa"].between(0, 1.5).all()
    assert (features["feature_struct_contacts12"] >= features["feature_struct_contacts8"]).all()


def test_structure_model_of_another_isoform_is_refused(tmp_path):
    from vpdl.dl.structure import wt_structure_table

    _helix_pdb(tmp_path / "X.pdb")
    table, index = wt_structure_table(tmp_path, {"X": "A" * 12})
    assert len(table) == 12 and index["X"]["residues"] == set(range(1, 13))
    with pytest.raises(ValueError, match="disagree"):
        wt_structure_table(tmp_path, {"X": "G" * 12})


def test_real_alphafold_model_parses_against_uniprot_when_cached():
    """Smoke test on the locally cached MSH2 model (no computation beyond parsing)."""
    from vpdl.dl.structure import parse_pdb

    from vpdl.dl.structure import _metadata

    candidates = [ROOT / "data/mmr/raw/alphafold", ROOT / "data/raw/alphafold"]
    found = next((d for d in candidates if (d / "P43246.pdb").exists()), None)
    if found is None:
        pytest.skip("MSH2 AlphaFold model not cached here")
    residues = parse_pdb(found / "P43246.pdb")
    sequence = _metadata(found / "P43246_metadata.json")["uniprotSequence"]
    assert "".join(r.aa for r in residues) == sequence and len(sequence) == 934


def test_variant_structures_are_missing_not_zero_without_inputs(tmp_path):
    from vpdl.dl.structure import VT_STRUCTURE_COLUMNS, PrecomputedVariantStructures

    _helix_pdb(tmp_path / "X.pdb")
    pd.DataFrame([{"uniprot_id": "X", "position": 3, "wt_aa": "A", "mut_aa": "G",
                   "ddg": 1.7}]).to_csv(tmp_path / "ddg.csv", index=False)
    variants = pd.DataFrame([{"uniprot_id": "X", "position": 3, "wt_aa": "A", "mut_aa": "G"},
                             {"uniprot_id": "X", "position": 5, "wt_aa": "A", "mut_aa": "V"}])
    deltas = PrecomputedVariantStructures(tmp_path, ddg_csv=tmp_path / "ddg.csv").deltas(variants)
    assert list(deltas.columns) == list(VT_STRUCTURE_COLUMNS)
    assert deltas["feature_struct_delta_ddg"].iloc[0] == 1.7
    assert np.isnan(deltas["feature_struct_delta_ddg"].iloc[1])


# -- genomic ---------------------------------------------------------------------------

def _transcript():
    # Plus strand, three exons. CDS: 11 bp in exon 1 (30-40), 12 in exon 2, 10 in exon 3
    # (200-209) = 33 bp = 10 residues + stop. Junction codons: 4 and 8.
    return {"id": "T", "strand": 1, "Translation": {"start": 30, "end": 209},
            "Exon": [{"start": 1, "end": 40}, {"start": 100, "end": 111},
                     {"start": 200, "end": 260}]}


def test_exon_codon_table_and_junction_features():
    from vpdl.dl.genomic import exon_codon_table, genomic_features, homology_codon_range

    table = exon_codon_table(_transcript(), protein_length=10)
    assert list(table["exon"]) == [1, 2, 3]
    assert table["c_start"].tolist() == [1, 12, 24]
    variants = pd.DataFrame({"gene": ["G"] * 3, "position": [1, 4, 10]})
    features = genomic_features(variants, {"G": table}, {"G": 10})
    assert features["feature_genomic_junction_codon"].tolist() == [0.0, 1.0, 0.0]
    assert features["feature_genomic_to_junction"].iloc[1] == 0
    assert homology_codon_range(table, 10, exons=(2, 3)) == (4, 10)
    with pytest.raises(ValueError, match="implies"):
        exon_codon_table(_transcript(), protein_length=11)


# -- functional validation data ---------------------------------------------------------------

def test_cimra_loader_orients_and_classifies(tmp_path):
    from vpdl.dl.functional import load_cimra

    pd.DataFrame({"gene": ["MLH1", "MLH1", "PMS2"], "position": [5, 6, 7],
                  "wt_aa": ["A", "C", "D"], "mut_aa": ["V", "Y", "N"],
                  "cimra_oddspath": [400.0, 0.001, 1.0],
                  "mechanism": ["missense", "missense", "splicing"]}
                 ).to_csv(tmp_path / "cimra.csv", index=False)
    frame = load_cimra(tmp_path / "cimra.csv", PANEL)
    assert len(frame) == 2                                         # splicing excluded
    assert frame["cimra_strength"].tolist() == ["PS3_very_strong", "BS3_very_strong"]
    assert frame["functional_class"].tolist() == [1.0, 0.0]
    assert frame["damage_score"].iloc[0] > frame["damage_score"].iloc[1]


def test_inverted_functional_direction_is_detected():
    from vpdl.dl.functional import assert_functional_orientation

    rng = np.random.default_rng(0)
    anchor = rng.random(200)
    frame = pd.DataFrame({"assay": "x", "damage_score": -anchor + rng.normal(0, 0.1, 200)})
    with pytest.raises(ValueError, match="INVERTED"):
        assert_functional_orientation(frame, pd.Series(anchor))


def test_functional_validation_correlates_scores_with_damage():
    from vpdl.dl.functional import functional_validation

    rng = np.random.default_rng(1)
    functional = pd.DataFrame({"uniprot_id": "P43246", "position": np.arange(1, 101),
                               "wt_aa": "A", "mut_aa": "V", "gene": "MSH2",
                               "assay": "pg_dms_continuous",
                               "damage_score": rng.normal(size=100),
                               "functional_class": np.nan, "raw_score": 0.0})
    scores = pd.DataFrame({"variant_key": [f"P43246:{p}:A>V" for p in range(1, 101)],
                           "gene": "MSH2",
                           "score": functional["damage_score"] + rng.normal(0, 0.3, 100)})
    result = functional_validation(scores, functional, n_bootstrap=50)
    assert result["spearman"].iloc[0] > 0.8 and result["n"].iloc[0] == 100


# -- feature store ----------------------------------------------------------------------------

def _identity():
    return {"model": "tiny", "model_version": {"hf_id": "random"},
            "sequence_version": {"P40692": "abc"}, "dataset_version": "sha"}


def test_feature_store_round_trip_mmap_and_refusals(tmp_path):
    from vpdl.dl.feature_store import FeatureStore

    store = FeatureStore(tmp_path)
    ids = ["P40692:1:A>V", "P40692:2:C>Y", "P40692:3:D>N"]
    blocks = {"site_wt": np.arange(12, dtype=np.float32).reshape(3, 4),
              "site_vt": np.ones((3, 4), dtype=np.float32)}
    entry = store.write("esm2", "tiny@P0", _identity(), ids, blocks)
    for key in ("variant_ids_sha256", "shapes", "dtype", "generation_date", "identity", "git"):
        assert key in entry.meta
    np.testing.assert_array_equal(entry.lookup("site_wt", [ids[2], ids[0]]),
                                  blocks["site_wt"][[2, 0]])
    assert isinstance(entry.array("site_wt"), np.memmap)
    with pytest.raises(KeyError, match="never zero-filled"):
        entry.lookup("site_wt", ["P40692:9:A>V"])
    with pytest.raises(FileExistsError):
        store.write("esm2", "tiny@P0", _identity(), ids, blocks)
    assert store.find("esm2", "tiny@P0", _identity()).path == entry.path
    spec = f"esm2/tiny@P0/{entry.meta['key']}:site_wt"
    assert store.resolve(spec)[1] == "site_wt"
    (entry.path / "variant_ids.txt").write_text("\n".join(reversed(ids)) + "\n")
    with pytest.raises(ValueError, match="stale"):
        store.load("esm2", "tiny@P0", entry.meta["key"])


# -- interface ----------------------------------------------------------------------------------

def test_dl_output_records_validate_and_round_trip(tmp_path):
    from vpdl.dl.interface import (DL_OUTPUT_SCHEMA, DLRepresentation, ReasoningRepresentation,
                                   read_dl_outputs, validate_dl_record, write_dl_outputs)

    reps = [DLRepresentation("P43246:10:A>V", np.arange(4), 0.87, 0.08,
                             {"gene": "MSH2", "model_version": "fusion@abc",
                              "feature_version": "esm2/x/y", "dataset_version": "sha",
                              "fold": "MSH2", "uncertainty_method": "mc_dropout"})]
    paths = write_dl_outputs(reps, tmp_path)
    back = read_dl_outputs(paths["jsonl"])
    assert back[0].variant_id == "P43246:10:A>V" and back[0].score == pytest.approx(0.87)
    np.testing.assert_array_equal(back[0].embedding, np.arange(4, dtype=np.float32))
    record = reps[0].to_record()
    assert set(DL_OUTPUT_SCHEMA["required"]) <= set(record)
    for bad in ({**record, "pathogenicity_score": 1.3}, {**record, "variant_id": "MSH2 A10V"},
                {**record, "extra_field": 1}, {**record, "uncertainty": -0.1}):
        with pytest.raises(ValueError):
            validate_dl_record(bad)
    with pytest.raises(ValueError):
        DLRepresentation("P43246:10:A>V", np.zeros(2), score=2.0)
    future = ReasoningRepresentation("P43246:10:A>V", np.zeros(3))
    assert future.modality == "reasoning" and reps[0].modality == "dl"


# -- run_cell end to end with DL models -----------------------------------------------------------

@pytest.fixture
def cell_inputs(tmp_path, sequences):
    from conftest import assembled_table
    from vpdl.dl.canonical import build_canonical

    table, _, _ = build_canonical(assembled_table(sequences, per_position=1), sequences)
    data = tmp_path / "canonical.csv"
    table.to_csv(data, index=False)
    return table, data, sequences


def test_fusion_cell_with_an_embedding_block_runs_through_the_protocol(tmp_path, cell_inputs):
    torch = pytest.importorskip("torch")                     # noqa: F841
    from vpdl.dl.feature_store import FeatureStore
    from vpdl.dl.runner import DLContext
    from vpdl.experiment import CellConfig, run_cell
    from vpdl.splits import variant_keys

    table, data, sequences = cell_inputs
    ids = list(variant_keys(table))
    rng = np.random.default_rng(0)
    blocks = {f"{level}_{side}": rng.normal(size=(len(ids), 6)).astype(np.float32)
              for level in ("site", "local", "global") for side in ("wt", "vt")}
    entry = FeatureStore(tmp_path / "features").write("esm2", "tiny@P0", _identity(), ids,
                                                      blocks)
    spec = f"esm2/tiny@P0/{entry.meta['key']}:site.vt_minus_wt"
    context = DLContext(store_root=tmp_path / "features", export_dir=tmp_path / "export",
                        sequences=sequences)
    config = CellConfig(sources=("clinvar",), model="fusion", seed=42, n_bootstrap=20,
                        embedding_blocks=(spec,), tag="concat",
                        model_kwargs={"epochs": 2, "mc_samples": 3})
    result = run_cell(table, config, [c for c in table.columns if c.startswith("feature_")],
                      data, tmp_path / "runs", sequences=sequences, dl_context=context)
    assert [row["gene"] for row in result.per_gene] == sorted(PANEL)
    assert result.provenance["feature_version"] == spec.partition(":")[0]
    assert result.provenance["leakage"]["critical"] == 0
    assert (tmp_path / "runs" / f"predictions_{config.slug}.csv").exists()
    exported = sorted((tmp_path / "export").glob("representations_*.npz"))
    assert len(exported) == 4
    blob = np.load(exported[0], allow_pickle=True)
    assert blob["representation"].shape[1] == 128 and np.all(blob["uncertainty"] >= 0)


def test_family_split_cell_scores_every_gene_once(tmp_path, cell_inputs):
    pytest.importorskip("torch")
    from vpdl.experiment import CellConfig, run_cell

    table, data, sequences = cell_inputs
    config = CellConfig(sources=("clinvar",), model="mlp", seed=42, n_bootstrap=20,
                        split="family", model_kwargs={"epochs": 2})
    result = run_cell(table, config, ["feature_gnomad_log10_af", "feature_in_domain"], data,
                      tmp_path / "runs")
    assert "__split-family" in config.slug
    assert sorted(row["gene"] for row in result.per_gene) == sorted(PANEL)
    predicted = result.predictions["variant_key"]
    assert predicted.is_unique and len(predicted) == int(table["label__clinvar"].notna().sum())


def test_embedding_only_arm_and_window_cnn_run(tmp_path, cell_inputs):
    pytest.importorskip("torch")
    from vpdl.experiment import CellConfig, run_cell

    table, data, sequences = cell_inputs
    config = CellConfig(sources=("clinvar",), model="cnn", seed=42, n_bootstrap=20,
                        model_kwargs={"epochs": 1})
    result = run_cell(table, config, ["feature_in_domain"], data, tmp_path / "runs",
                      sequences=sequences)
    assert len(result.per_gene) == 4


# -- CLI and tracking ----------------------------------------------------------------------------

def test_cli_parses_every_command_and_applies_toml(tmp_path):
    from vpdl.dl.cli import _apply_config, build_parser

    parser = build_parser()
    for command in parser.subcommands:
        assert parser.subcommands[command].format_help()
    (tmp_path / "c.toml").write_text('seeds = [1, 2, 3]\nsplit = "family"\n')
    argv = ["train", "--data", "d.csv", "--sources", "clinvar", "--model", "mlp",
            "--config", str(tmp_path / "c.toml"), "--split", "logo"]
    args = parser.parse_args(argv)
    _apply_config(args, parser.subcommands["train"], argv)
    assert args.seeds == [1, 2, 3] and args.split == "logo"          # explicit flag wins
    for toml in (ROOT / "configs" / "dl").glob("*.toml"):
        import tomllib
        tomllib.loads(toml.read_text())


def test_modalities_map_to_complementary_drop_groups():
    from vpdl.dl.cli import _drop_for
    from vpdl.features import PRIOR_GROUPS

    assert _drop_for("population", []) == sorted(set(PRIOR_GROUPS) - {"gnomad"})
    assert _drop_for("none", []) == sorted(PRIOR_GROUPS)
    assert _drop_for(None, ["gnomad"]) == ["gnomad"]
    with pytest.raises(SystemExit):
        _drop_for("telepathy", [])


def test_run_record_has_every_required_field(tmp_path):
    from vpdl.dl.tracking import append_registry, experiment_id, run_record

    record = run_record("train", {"model": "mlp"}, seed=42, feature_version="esm2/x/y",
                        model_version="mlp", metrics={"auc": 0.9}, checkpoint="c.pt")
    for key in ("experiment_id", "git_commit", "dataset_version", "feature_version",
                "model_version", "config", "seed", "hardware", "metrics", "checkpoint"):
        assert key in record
    assert experiment_id("train", {"a": 1}, "d", 1, "c") == experiment_id("train", {"a": 1},
                                                                          "d", 1, "c")
    path = append_registry(record, tmp_path / "registry.jsonl")
    assert json.loads(path.read_text().splitlines()[0])["seed"] == 42


def test_check_hardware_script_reports_without_benchmarking():
    import subprocess
    import sys

    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_hardware.py")],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    report = json.loads(out.stdout)
    assert {"device", "host", "checks", "recommendations"} <= set(report)


def test_per_fold_embeddings_refuse_a_backbone_that_saw_the_fold(tmp_path):
    from vpdl.dl.feature_store import FeatureStore
    from vpdl.dl.leakage import LeakageError
    from vpdl.dl.runner import DLContext
    from vpdl.dl.splits import Fold

    store = FeatureStore(tmp_path / "features")
    ids = ["P40692:1:A>V", "P43246:1:A>V"]
    blocks = {"site_wt": np.zeros((2, 3), np.float32), "site_vt": np.ones((2, 3), np.float32)}
    locations = {}
    for gene, holdout in (("MLH1", "P40692"), ("MSH2", "P40692")):   # MSH2's is WRONG
        identity = {**_identity(), "adaptation": {"pretrain_corpus_mode": "strict",
                                                  "pretrain_holdout": holdout,
                                                  "pretrain_arm": "P1"}}
        entry = store.write("esm2", f"tiny@P1-{gene}", identity, ids, blocks)
        locations[gene] = f"esm2/tiny@P1-{gene}/{entry.meta['key']}"
    mapping = tmp_path / "P1_blocks.json"
    mapping.write_text(json.dumps(locations))
    context = DLContext(store_root=tmp_path / "features")
    spec = f"perfold={mapping}:site_vt"
    fold = lambda gene: Fold(gene, np.array([], int), np.array([], int), (gene,))
    assert context.embeddings(spec, ids, fold("MLH1")).shape == (2, 3)
    with pytest.raises(LeakageError, match="held out"):
        context.embeddings(spec, ids, fold("MSH2"))
    assert "perfold{" in context.feature_version([spec])


# -- results bundle for the paper ----------------------------------------------------------------

def test_results_bundle_collects_tables_checks_and_manifest(tmp_path, cell_inputs):
    pytest.importorskip("torch")
    from vpdl.dl.report import write_paper_bundle
    from vpdl.experiment import CellConfig, run_cell
    from vpdl.provenance import verify_manifest

    table, data, sequences = cell_inputs
    runs = tmp_path / "runs" / "main"
    features = ["feature_gnomad_log10_af", "feature_in_domain"]
    for model, seeds in (("mlp", (42, 43, 44)), ("gbm", (42,))):
        for seed in seeds:
            config = CellConfig(sources=("clinvar",), model=model, seed=seed, n_bootstrap=20,
                                model_kwargs={"epochs": 2} if model == "mlp" else {})
            try:
                run_cell(table, config, features, data, runs)
            except ImportError:                       # this PC blocks sklearn's _loss
                pytest.skip("no usable GBM backend here")

    bundle = write_paper_bundle(tmp_path / "runs", tmp_path / "results", n_bootstrap=20,
                                plots=False)
    cells = pd.read_csv(bundle.files["cells"])
    arms = pd.read_csv(bundle.files["arms"])
    assert set(cells["model"]) == {"mlp", "gbm"}
    assert {"roc_auc", "mcc", "brier", "ece", "dataset_sha256", "split_hash",
            "roc_auc_ci_low"} <= set(cells.columns)
    headline = arms[arms["gene"] == "mean:scoreable"]
    assert len(headline) == 2 and headline["seeds"].max() == 3
    # Three seeds of one arm give an SD; a single-seed arm reports none.
    mlp = headline[headline["model"] == "mlp"].iloc[0]
    assert np.isfinite(mlp["roc_auc_std"]) and mlp["roc_auc_mean"] > 0
    assert not np.isfinite(headline[headline["model"] == "gbm"].iloc[0]["roc_auc_std"])
    # The single-seed arm is flagged, and every file is checksummed.
    problems = " ".join(bundle.checks["problems"])
    assert "gbm" in problems and "1 seed" in problems and not bundle.checks["citable"]
    verify_manifest(bundle.files["manifest"])
    text = bundle.files["tables_md"].read_text(encoding="utf-8")
    assert "Table 2" in text and "Table 3" in text
    assert r"\begin{tabular}" in bundle.files["tables_tex"].read_text(encoding="utf-8")


def test_results_refuses_to_pool_runs_from_different_tables(tmp_path, cell_inputs):
    pytest.importorskip("torch")
    from vpdl.dl.report import collect_cells, comparability_checks, paired_against
    from vpdl.experiment import CellConfig, run_cell

    table, data, sequences = cell_inputs
    other = tmp_path / "other.csv"                    # same variants, different file/hash
    table.assign(note="second build").to_csv(other, index=False)
    for path, out in ((data, tmp_path / "runs" / "a"), (other, tmp_path / "runs" / "b")):
        config = CellConfig(sources=("clinvar",), model="mlp", seed=42, n_bootstrap=20,
                            model_kwargs={"epochs": 1})
        run_cell(table, config, ["feature_in_domain"], path, out)

    cells = collect_cells(tmp_path / "runs")
    checks = comparability_checks(cells)
    assert any("different tables" in p for p in checks["problems"])
    with pytest.raises(ValueError, match="dataset"):
        paired_against((str(tmp_path / "runs" / "a"), "train-clinvar", "mlp"),
                       [str(tmp_path / "runs" / "b")], cells, n_bootstrap=20)


def test_rerunning_a_cell_archives_the_previous_results(tmp_path, cell_inputs):
    pytest.importorskip("torch")
    from vpdl.dl.cli import _archive_superseded
    from vpdl.experiment import CellConfig, run_cell

    table, data, sequences = cell_inputs
    runs = tmp_path / "runs"
    config = CellConfig(sources=("clinvar",), model="mlp", seed=42, n_bootstrap=20,
                        model_kwargs={"epochs": 1})
    run_cell(table, config, ["feature_in_domain"], data, runs)
    first = (runs / f"predictions_{config.slug}.csv").read_text()
    archive = _archive_superseded(runs, config.slug)
    assert archive and (archive / f"predictions_{config.slug}.csv").read_text() == first
    assert not (runs / f"summary_{config.slug}.json").exists()
    run_cell(table, config, ["feature_in_domain"], data, runs)
    # The re-run is visible, the old numbers survive, and the bundle ignores them.
    from vpdl.dl.report import collect_cells
    assert collect_cells(runs)["cell"].nunique() == 1


def test_parallel_cells_give_identical_results_to_sequential(tmp_path, cell_inputs, monkeypatch):
    # --jobs changes wall-clock time only: same cells, same seeds, same files.
    pytest.importorskip("torch")
    from vpdl.dl.cli import main

    table, data, sequences = cell_inputs
    monkeypatch.chdir(tmp_path)                    # the run registry lands here
    common = ["train", "--data", str(data), "--sources", "clinvar", "--model", "mlp",
              "--modalities", "population", "--seeds", "42", "43", "--n-bootstrap", "20",
              "--model-kwargs", '{"epochs": 2}', "--tag", "t"]
    assert main(common + ["--jobs", "1", "--out", str(tmp_path / "serial")]) == 0
    assert main(common + ["--jobs", "2", "--out", str(tmp_path / "parallel")]) == 0
    serial = sorted((tmp_path / "serial").glob("predictions_*.csv"))
    parallel = sorted((tmp_path / "parallel").glob("predictions_*.csv"))
    assert [p.name for p in serial] == [p.name for p in parallel] and len(serial) == 2
    for a, b in zip(serial, parallel):
        pd.testing.assert_frame_equal(pd.read_csv(a), pd.read_csv(b))
    records = [json.loads(line) for line in
               (tmp_path / "runs" / "dl" / "registry.jsonl").read_text().splitlines()]
    assert len(records) == 4 and {r["parallel_jobs"] for r in records} == {1, 2}
    assert all("peak_gpu_memory_gib" in r for r in records)


# -- re-running embed / zeroshot on identical inputs ---------------------------------------------

@pytest.fixture
def tiny_cli(monkeypatch, tmp_path, cell_inputs):
    pytest.importorskip("transformers")
    import vpdl.dl.plm.backbones as backbones
    from vpdl.dl.cli import _write_sidecar
    from vpdl.dl.plm.backbones import tiny_backbone

    _, data, sequences = cell_inputs
    _write_sidecar(data, sequences)                # the test sequences, not UniProt's
    monkeypatch.setattr(backbones, "load_backbone", lambda *a, **k: tiny_backbone())
    monkeypatch.chdir(tmp_path)                    # the run registry lands here


def test_zeroshot_rerun_reuses_the_stored_entry_and_recompute_reports_reproduction(
        tmp_path, cell_inputs, tiny_cli, monkeypatch, capsys):
    import vpdl.dl.plm.zeroshot as zeroshot
    from vpdl.dl.cli import main

    _, data, _ = cell_inputs
    command = ["zeroshot", "--data", str(data), "--policy", "centered",
               "--store", str(tmp_path / "features"), "--table-out", str(tmp_path / "zs.csv")]
    assert main(command) == 0
    column = "zeroshot_tiny_rotary_P0_masked_marginal"
    first = pd.read_csv(tmp_path / "zs.csv")[column]

    real = zeroshot.zeroshot_frame
    monkeypatch.setattr(zeroshot, "zeroshot_frame",
                        lambda *a, **k: pytest.fail("identical inputs must not be recomputed"))
    assert main(command) == 0                                  # used to raise FileExistsError
    assert "reusing it" in capsys.readouterr().err
    pd.testing.assert_series_equal(pd.read_csv(tmp_path / "zs.csv")[column], first)

    monkeypatch.setattr(zeroshot, "zeroshot_frame", real)
    assert main(command + ["--recompute"]) == 0
    assert "max |difference| per block" in capsys.readouterr().out
    records = [json.loads(line) for line in
               (tmp_path / "runs" / "dl" / "registry.jsonl").read_text().splitlines()]
    assert [r["reused_entry"] for r in records] == [False, True, False]
    assert records[-1]["reproduction"]["max_abs_diff"]["pathogenicity"] < 1e-4
    entries = list((tmp_path / "features" / "zeroshot").glob("*/*"))
    assert len(entries) == 1 and not entries[0].name.startswith(".")   # no staging left behind


def test_embed_rerun_reuses_the_stored_entry(tmp_path, cell_inputs, tiny_cli, capsys):
    from vpdl.dl.cli import main

    _, data, _ = cell_inputs
    command = ["embed", "--data", str(data), "--policy", "centered",
               "--store", str(tmp_path / "features")]
    assert main(command) == 0
    assert main(command) == 0
    assert "reusing it" in capsys.readouterr().err
    assert len(list((tmp_path / "features" / "esm2").glob("*/*"))) == 1


def test_a_losing_writer_leaves_no_staging_directory(tmp_path):
    from vpdl.dl.feature_store import FeatureStore

    store = FeatureStore(tmp_path)
    store.write("esm2", "t@P0", _identity(), ["a", "b"], {"x": np.zeros((2, 3))})
    with pytest.raises(FileExistsError):
        store.write("esm2", "t@P0", _identity(), ["a", "b"], {"x": np.zeros((2, 3))})
    assert [p.name.startswith(".") for p in (tmp_path / "esm2" / "t@P0").iterdir()] == [False]
    assert len(store.entries()) == 1
