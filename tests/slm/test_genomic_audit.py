"""Regression tests for the local code audit of 2026-09-24 (GENOMIC_SLM_CODE_AUDIT.md).

One test per defect the audit fixed, each written to fail on the code before the fix.
Synthetic inputs and tiny random-weight models only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vpdl.slm.build.examples import model_input
from vpdl.slm.build.leakage import AuditInputs, LeakageError, run_audit
from vpdl.slm.interface import DLInput, SLMRepresentation, read_slm_outputs, validate_slm_record, \
    write_slm_outputs
from vpdl.slm.teacher import TeacherConfig, TeacherRecord, filter_outputs, run_teacher
from vpdl.slm.text.acmg import task_policy
from vpdl.slm.text.conclusion import classify_sentence, mask_conclusions, residual_assertions

ROOT = Path(__file__).resolve().parents[2]


# -- conclusion leakage: verdicts without a verb ------------------------------------------------

@pytest.mark.parametrize("sentence", [
    "Pathogenic (PVS1, PM2_Supporting, PP3)",
    "Likely benign: BS1, BP4.",
    "Assertion: Likely Pathogenic",
    "Result: Likely Pathogenic (PS3, PM2).",
    "Criteria met: PS3, PM2 -> Likely Pathogenic",
    "ACMG: LP",
    "Final call: VUS",
    "This variant is classified as LP.",
    "This variant is a variant of uncertain significance (VUS-high).",
    "LP.",
])
def test_verbless_and_abbreviated_verdicts_are_conclusions(sentence):
    assert classify_sentence(sentence) is not None


@pytest.mark.parametrize("sentence", [
    "REVEL: benign.",
    "SIFT: tolerated; PolyPhen: benign",
    "Functional assay: benign.",
    "Segregation: 3 affected carriers (PP1).",
    "c.123A>G -> p.Arg41Gly",
    "B cells from the patient showed reduced expression.",
    "It was observed in trans with a pathogenic variant (PMID: 12345).",
])
def test_evidence_that_names_a_class_word_after_a_colon_is_kept(sentence):
    assert classify_sentence(sentence) is None


def test_a_label_final_narrative_leaves_no_label_in_the_model_input():
    text = ("This variant is absent from gnomAD. Functional studies show loss of function. "
            "Pathogenic (PVS1, PM2_Supporting, PP3)")
    kept, stats = model_input(text, task_policy("classify"))
    assert "absent from gnomAD" in kept and "loss of function" in kept
    assert "Pathogenic" not in kept and stats.conclusions_removed == 1
    assert residual_assertions(mask_conclusions(text).text) == []


def test_the_tripwire_counts_verbless_labels_the_masker_might_miss():
    assert residual_assertions("Assertion: LP.")
    assert residual_assertions("The history was reviewed. Pathogenic (PVS1, PM2).")


# -- the input gate that finetune runs itself ---------------------------------------------------

def _gate_examples(text: str) -> pd.DataFrame:
    return pd.DataFrame({"example_id": ["e1", "e2"], "document_id": ["d1", "d2"],
                         "variant_id": ["clinvar:1", "clinvar:2"], "gene": ["MLH1", "BRCA1"],
                         "split": ["train", "test"],
                         "input_text": ["The variant is absent from gnomAD.", text]})


def test_training_refuses_inputs_that_still_state_the_verdict():
    from vpdl.slm.build.leakage import input_gate
    input_gate(_gate_examples("It segregated with disease."))          # clean: passes
    with pytest.raises(LeakageError, match="conclusion"):
        input_gate(_gate_examples("It segregated with disease. Classification: Pathogenic"))
    with pytest.raises(LeakageError, match="label_leakage"):
        input_gate(_gate_examples("It segregated with disease."), features=("review_status",))


def test_finetune_itself_runs_the_input_gate(tmp_path):
    from vpdl.slm.modeling.finetune import FinetuneConfig, run_finetune
    examples = _gate_examples("Assertion: Likely Pathogenic").assign(target_label="pathogenic")
    path = tmp_path / "examples.parquet"
    examples.to_parquet(path, index=False)
    config = FinetuneConfig(examples=str(path), out_dir=str(tmp_path / "run"),
                            records=str(tmp_path / "missing"), use_structured=False)
    with pytest.raises(LeakageError):
        run_finetune(config, dry_run=True)


def test_an_isolating_split_audited_without_clusters_is_not_clean():
    examples = _gate_examples("It segregated with disease.")
    report = run_audit(AuditInputs("text", examples))
    near = [f for f in report.findings if f.check == "near_duplicates"]
    assert near and near[0].severity == "critical"
    report = run_audit(AuditInputs("variant", examples))
    assert [f for f in report.findings if f.check == "near_duplicates"][0].severity == "warning"


# -- the output contract -------------------------------------------------------------------------

def _representation(probabilities=None, **kwargs):
    probabilities = probabilities or {"pathogenic": 0.5, "likely_pathogenic": 0.2, "vus": 0.2,
                                      "likely_benign": 0.05, "benign": 0.05}
    return SLMRepresentation(
        variant_id="clinvar:1", embedding=np.arange(4, dtype=np.float32), score=0.7,
        uncertainty=kwargs.pop("uncertainty", 0.3),
        metadata={"gene": "MLH1", "model_version": "m1", "feature_version": "f1"},
        class_probabilities=probabilities, **kwargs)


def test_class_probabilities_outside_zero_one_are_refused_even_when_they_sum_to_one():
    bad = {"pathogenic": 1.6, "likely_pathogenic": -0.6, "vus": 0.0, "likely_benign": 0.0,
           "benign": 0.0}
    with pytest.raises(ValueError, match="outside"):
        _representation(bad)
    record = _representation().to_record()
    record["class_probabilities"] = bad
    with pytest.raises(ValueError, match="outside"):
        validate_slm_record(record)


def test_a_nan_uncertainty_is_refused_rather_than_written_as_invalid_json():
    record = _representation().to_record()
    record["uncertainty"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_slm_record(record)


def test_referenced_slm_embeddings_are_checked_and_the_file_is_released(tmp_path):
    paths = write_slm_outputs([_representation()], tmp_path)
    back = read_slm_outputs(paths["jsonl"])
    assert back[0].embedding.flags.owndata or back[0].embedding.base is None \
        or not isinstance(back[0].embedding.base, np.memmap)
    np.save(paths["embeddings"], np.full((1, 4), np.nan, dtype=np.float32))   # overwrite: file free
    with pytest.raises(ValueError, match="non-finite"):
        read_slm_outputs(paths["jsonl"])


def _dl_records(tmp_path, folds, genes, ids=None):
    from vpdl.dl.interface import DLRepresentation, write_dl_outputs
    ids = ids or [f"P40692:{i + 1}:G>R" for i in range(len(folds))]
    representations = [DLRepresentation(i, np.ones(3, dtype=np.float32), 0.5, 0.1,
                                        {"gene": g, "fold": f, "model_version": "dl",
                                         "feature_version": "f"})
                       for i, f, g in zip(ids, folds, genes)]
    return write_dl_outputs(representations, tmp_path)["jsonl"], ids


def test_family_folds_count_as_out_of_gene_and_random_debug_folds_do_not(tmp_path):
    jsonl, ids = _dl_records(tmp_path, ["family-MutL", "family:MutS", "MSH6", "random_debug-0"],
                             ["MLH1", "MSH2", "MSH6", "PMS2"])
    folds = DLInput.from_outputs(jsonl).fold_check(ids, ["MLH1", "MSH2", "MSH6", "PMS2"])
    assert folds["out_of_fold"] == 3 and folds["in_fold_or_other"] == 1


def test_a_dl_export_with_a_variant_twice_is_refused(tmp_path):
    jsonl, _ = _dl_records(tmp_path, ["MLH1", "PMS2"], ["MLH1", "MLH1"],
                           ids=["P40692:1:G>R", "P40692:1:G>R"])
    with pytest.raises(ValueError, match="more than once"):
        DLInput.from_outputs(jsonl)


# -- inference respects the gene specifications ------------------------------------------------

def test_acmg_codes_a_gene_specification_excludes_are_never_predicted():
    torch = pytest.importorskip("torch")
    from vpdl.slm.modeling.backbone import BackboneSpec, load_backbone
    from vpdl.slm.modeling.data import assemble_documents
    from vpdl.slm.modeling.multitask import GenomicSLM, HeadConfig, applicability_mask
    from vpdl.slm.modeling.predict import predict_documents
    from vpdl.slm.text.acmg import CODES

    backbone = load_backbone(BackboneSpec("tiny-bert", max_length=16), seed=0)
    examples = _gate_examples("It segregated with disease.").assign(target_label="pathogenic")
    applicable = applicability_mask(examples["gene"].tolist()).numpy()
    tensors = assemble_documents(examples, backbone, 16, None, None, None, applicable)
    model = GenomicSLM(backbone, HeadConfig(tasks=("classify", "acmg"), embedding_dim=8))
    out = predict_documents(model, tensors, np.arange(2), torch.device("cpu"))
    mlh1_off = [CODES.index(c) for c in CODES if applicable[0, CODES.index(c)] == 0]
    assert mlh1_off and (out["acmg_probs"][0, mlh1_off] == 0).all()
    assert (out["acmg_probs"][1] > 0).any()


# -- fine-tuning: threshold and provenance ---------------------------------------------------------

def _balanced_examples(n: int = 24) -> pd.DataFrame:
    labels = ["pathogenic", "benign", "likely_pathogenic", "likely_benign"] * (n // 4)
    split = ["train"] * (n - 8) + ["val"] * 4 + ["test"] * 4
    return pd.DataFrame({
        "example_id": [f"e{i}" for i in range(n)], "document_id": [f"d{i}" for i in range(n)],
        "variant_id": [f"clinvar:{i}" for i in range(n)], "gene": ["MLH1", "BRCA1"] * (n // 2),
        "split": split, "target_label": labels, "target_soft": [None] * n,
        "target_binary": [1.0, 0.0, 1.0, 0.0] * (n // 4),
        "input_text": [f"absent from gnomAD and seen in {i} families" for i in range(n)],
        "target_category": ["five_class"] * n, "task": ["classify"] * n})


def test_the_threshold_is_chosen_on_the_scores_it_is_applied_to(tmp_path):
    torch = pytest.importorskip("torch")
    from vpdl.slm.evaluation.metrics import validation_threshold
    from vpdl.slm.modeling.finetune import FinetuneConfig, run_finetune

    path = tmp_path / "examples.parquet"
    examples = _balanced_examples()
    examples.to_parquet(path, index=False)
    run_finetune(FinetuneConfig(
        run_name="t", out_dir=str(tmp_path / "run"), examples=str(path),
        records=str(tmp_path / "missing"), backbone="tiny-bert", max_length=16,
        peft={"kind": "full", "allow_full": True}, embedding_dim=8, use_structured=False,
        train={"epochs": 1, "batch_size": 4, "lr": 1e-3}, seed=3, cache_dir=str(tmp_path / "c")))
    saved = torch.load(tmp_path / "run" / "model" / "model.pt", weights_only=False)
    assert saved["threshold_basis"] == "calibrated"
    assert saved["provenance"]["backbone_version"]["backbone"] == "tiny-bert"
    predictions = pd.read_parquet(tmp_path / "run" / "predictions.parquet")
    val = predictions[predictions["split"] == "val"].merge(examples, on="example_id")
    expected = validation_threshold(val["target_binary"], val["pathogenic_score"])
    assert saved["threshold"] == pytest.approx(expected)


# -- continued pretraining: resume == uninterrupted ---------------------------------------------

def test_a_resumed_pretraining_run_makes_the_same_dropout_draws(tmp_path):
    pytest.importorskip("torch")
    from vpdl.slm.modeling.backbone import BackboneSpec, load_backbone
    from vpdl.slm.modeling.continued import PretrainConfig, pack_for_tokenizer, run_pretrain

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    lines = [json.dumps({"id": f"d{i}", "source": "t", "text": "the variant is absent from "
                         f"population databases and segregated in {i} families"}) for i in range(60)]
    (corpus / "train-00000.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (corpus / "val.jsonl").write_text("\n".join(lines[:5]) + "\n", encoding="utf-8")
    pack_for_tokenizer(corpus, load_backbone(BackboneSpec("tiny-bert")).tokenizer, tmp_path / "tok")

    def config(out):
        return PretrainConfig(backbone="tiny-bert", corpus_dir=str(corpus),
                              token_dir=str(tmp_path / "tok"), out_dir=str(out), context=32,
                              micro_batch=2, total_steps=4, warmup_steps=1, eval_every=100,
                              checkpoint_every=1, log_every=1, keep_last=10, seed=5)

    def losses(out):
        return [json.loads(line)["loss"] for line in (out / "log.jsonl").read_text().splitlines()]

    run_pretrain(config(tmp_path / "whole"))
    whole = losses(tmp_path / "whole")
    # The same run, "interrupted" after step 2: later checkpoints and log lines removed.
    split = tmp_path / "split"
    run_pretrain(config(split))
    for checkpoint in sorted((split / "checkpoints").glob("step-*.pt"))[2:]:
        checkpoint.unlink()
    losses_before = losses(split)[:2]
    (split / "log.jsonl").write_text("".join(json.dumps({"loss": x}) + "\n" for x in losses_before))
    run_pretrain(config(split))                                      # resumes from step 2
    assert losses(split) == whole


# -- teacher ---------------------------------------------------------------------------------------

def _record(confidence):
    return TeacherRecord("e1", "clinvar:1", "qwen3:32b", "v", "h", "raw", {"units": []}, confidence,
                         True, None, "now", "train")


def test_a_teacher_answer_without_a_confidence_cannot_pass_the_threshold():
    result = filter_outputs([_record(None), _record(0.9)], TeacherConfig(min_confidence=0.6))
    assert result["counts"]["kept"] == 1 and result["kept"][0].confidence == 0.9


def test_teacher_units_must_cite_given_ids_and_known_labels():
    from vpdl.slm.teacher import unit_problems
    evidence = {"E1": {"text": "absent from gnomAD"}}
    assert unit_problems({"units": [{"id": "E1", "types": ["population"], "polarity": "pathogenic"}]},
                         evidence) == []
    problems = unit_problems({"units": [{"id": "E9", "types": ["astrology"], "polarity": "sure"}]},
                             evidence)
    assert len(problems) == 3


def test_a_teacher_setting_the_local_client_would_ignore_is_refused():
    with pytest.raises(ValueError, match="temperature"):
        run_teacher([], TeacherConfig(temperature=0.7))


# -- CLI -------------------------------------------------------------------------------------------

def test_a_pretraining_corpus_without_exclusions_needs_an_explicit_opt_out(tmp_path):
    from vpdl.slm.cli import main
    assert main(["pretrain-corpus", "--out", str(tmp_path / "corpus")]) == 2
    assert not (tmp_path / "corpus").exists()


def test_vus_outcomes_are_joined_per_variant(tmp_path):
    from vpdl.slm.cli import main
    from vpdl.slm.schema import CLASSES
    predictions = pd.DataFrame({"example_id": ["d1", "d2", "d3"], "split": ["test"] * 3,
                                **{f"p_{c}": [0.2] * 3 for c in CLASSES}})
    examples = pd.DataFrame({"example_id": ["d1", "d2", "d3"],
                             "variant_id": ["clinvar:1", "clinvar:1", "clinvar:2"],
                             "target_label": ["vus"] * 3, "target_binary": [np.nan] * 3,
                             "gene": ["MLH1"] * 3})
    old = pd.DataFrame({"variant_id": ["clinvar:1", "clinvar:2"],
                        "clinvar_classification": ["Uncertain significance"] * 2})
    new = pd.DataFrame({"variant_id": ["clinvar:1", "clinvar:2"],
                        "clinvar_classification": ["Pathogenic", "Benign"]})
    for name, frame in (("p", predictions), ("e", examples), ("old", old), ("new", new)):
        frame.to_parquet(tmp_path / f"{name}.parquet", index=False)
    out = tmp_path / "evaluation.json"
    assert main(["evaluate", "--predictions", str(tmp_path / "p.parquet"),
                 "--examples", str(tmp_path / "e.parquet"), "--vus-old", str(tmp_path / "old.parquet"),
                 "--vus-new", str(tmp_path / "new.parquet"), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["vus_reclassification"]["variants_joined"] == 2


# -- shipped configurations ------------------------------------------------------------------------

@pytest.mark.parametrize("path", sorted((ROOT / "configs" / "slm").glob("exp*.toml")),
                         ids=lambda p: p.name)
def test_every_shipped_finetune_config_loads(path):
    from vpdl.slm.config import load_finetune_config
    config = load_finetune_config(path)          # unknown keys raise
    assert config.examples and config.tasks


def test_the_shipped_pretraining_config_loads():
    from vpdl.slm.config import load_pretrain_config
    config = load_pretrain_config(ROOT / "configs" / "slm" / "pretrain_dgx.toml")
    assert config.backbone
