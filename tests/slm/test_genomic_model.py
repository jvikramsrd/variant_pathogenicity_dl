"""Model, training and pretraining — tiny random-weight models, on the CPU, in seconds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from vpdl.slm.modeling.backbone import (BackboneSpec, NAMED_BACKBONES, load_backbone, resolve_spec,
                                        tiny_tokenizer, tokenizer_fingerprint)
from vpdl.slm.modeling.continued import PretrainConfig, mask_tokens, pack_for_tokenizer, run_pretrain
from vpdl.slm.modeling.data import assemble_documents, to_batch, tokenize_cached
from vpdl.slm.modeling.finetune import FinetuneConfig, run_finetune
from vpdl.slm.modeling.losses import LossWeights, multitask_loss, ordinal_emd, soft_cross_entropy
from vpdl.slm.modeling.multitask import GenomicSLM, HeadConfig, applicability_mask
from vpdl.slm.modeling.peft import apply_peft
from vpdl.slm.modeling.sizing import estimate_memory_gib, size_report
from vpdl.slm.schema import CLASSES
from vpdl.slm.text.acmg import CODES


@pytest.fixture(scope="module")
def backbone():
    return load_backbone(BackboneSpec("tiny-bert", max_length=32), seed=0)


def _examples(n: int = 12) -> pd.DataFrame:
    labels = ["pathogenic", "benign", "vus"] * (n // 3)
    return pd.DataFrame({
        "example_id": [f"e{i}" for i in range(n)],
        "document_id": [f"d{i}" for i in range(n)],
        "variant_id": [f"clinvar:{i}" for i in range(n)],
        "gene": ["MLH1", "BRCA1", "TP53"] * (n // 3),
        "split": (["train"] * (n - 4)) + ["val"] * 2 + ["test"] * 2,
        "input_text": [f"the variant is absent from gnomAD and was seen in {i} families" for i in range(n)],
        "target_label": labels, "target_soft": [None] * n,
        "target_binary": [1.0, 0.0, np.nan] * (n // 3),
        "source_id": ["clinvar_submission_summary"] * n, "submitter": ["lab"] * n,
        "date": ["2024-01-01"] * n, "tier": [2] * n,
        "conclusions_removed": [1] * n, "external_removed": [0] * n, "code_lists_removed": [0] * n,
        "codes_masked": [0] * n, "holdout_removed": [0] * n, "sentences": [3] * n,
        "target_category": ["five_class"] * n, "task": ["classify"] * n,
    })


# -- backbones ---------------------------------------------------------------------

def test_the_tiny_tokenizer_is_identical_every_time():
    assert tokenizer_fingerprint(tiny_tokenizer()) == tokenizer_fingerprint(tiny_tokenizer())


def test_named_baselines_resolve_to_hub_ids_with_their_licence_recorded():
    assert resolve_spec("pubmedbert").startswith("hf:microsoft/")
    assert all(entry["licence"] for entry in NAMED_BACKBONES.values())


def test_a_backbone_pools_a_batch_to_one_vector_per_document(backbone):
    tokens = backbone.tokenize(["the variant is absent from gnomAD", "short"], 16)
    pooled = backbone.encode(torch.tensor(tokens["input_ids"]), torch.tensor(tokens["attention_mask"]))
    assert pooled.shape == (2, backbone.hidden_size)


# -- parameter-efficient tuning -------------------------------------------------------

def test_lora_trains_a_small_fraction_of_an_encoder(backbone):
    summary = apply_peft(backbone.model, {"kind": "lora", "lora_rank": 2})
    assert 0 < summary["trainable_fraction"] < 0.5
    apply_peft(backbone.model, {"kind": "full", "allow_full": True})     # restore


def test_lora_works_on_a_decoder_layout():
    decoder = load_backbone(BackboneSpec("tiny-llama"), seed=0)
    summary = apply_peft(decoder.model, {"kind": "lora", "lora_rank": 2})
    assert summary["trainable_params"] > 0
    assert summary["trainable_fraction"] < 0.5


def test_full_fine_tuning_of_a_large_backbone_is_refused_without_evidence():
    class _Fake:
        def __init__(self):
            self.layers = [torch.nn.Linear(4, 4) for _ in range(2)]
            self._parameters = [torch.nn.Parameter(torch.zeros(200_000_000))]

        def parameters(self):
            return iter(self._parameters)

    with pytest.raises(ValueError, match="refused"):
        apply_peft(_Fake(), {"kind": "full"})


# -- the model ------------------------------------------------------------------------

def test_every_head_produces_the_shape_its_task_needs(backbone):
    config = HeadConfig(tasks=("classify", "acmg", "evidence"), embedding_dim=16,
                        cat_cardinalities=(3, 4), n_numeric=2, dl_dim=8)
    model = GenomicSLM(backbone, config)
    tokens = backbone.tokenize(["absent from gnomAD"] * 3, 16)
    out = model(input_ids=torch.tensor(tokens["input_ids"]),
                attention_mask=torch.tensor(tokens["attention_mask"]),
                categorical=torch.zeros(3, 2, dtype=torch.long), numeric=torch.zeros(3, 2),
                numeric_mask=torch.ones(3, 2), dl_embedding=torch.zeros(3, 8),
                dl_mask=torch.tensor([1.0, 0.0, 1.0]))
    assert out["class_logits"].shape == (3, len(CLASSES))
    assert out["acmg_logits"].shape == (3, len(CODES))
    assert out["embedding"].shape == (3, 16)
    units = model.forward_units(torch.tensor(tokens["input_ids"]),
                                torch.tensor(tokens["attention_mask"]))
    assert units["type_logits"].shape[0] == 3 and units["polarity_logits"].shape == (3, 4)


def test_a_variant_without_a_dl_record_is_masked_not_imputed(backbone):
    model = GenomicSLM(backbone, HeadConfig(embedding_dim=8, dl_dim=4))
    tokens = backbone.tokenize(["absent from gnomAD"] * 2, 16)
    ids, mask = torch.tensor(tokens["input_ids"]), torch.tensor(tokens["attention_mask"])
    model.eval()
    with torch.no_grad():
        absent = model(input_ids=ids, attention_mask=mask, dl_embedding=torch.zeros(2, 4),
                       dl_mask=torch.zeros(2))
        noise = model(input_ids=ids, attention_mask=mask, dl_embedding=torch.randn(2, 4),
                      dl_mask=torch.zeros(2))
    assert torch.allclose(absent["class_logits"], noise["class_logits"], atol=1e-5)


def test_gene_specifications_mask_the_acmg_head():
    mask = applicability_mask(["MLH1", "BRCA1"])
    assert mask[0, CODES.index("PM1")] == 0.0
    assert mask[1, CODES.index("PM1")] == 1.0
    assert mask[:, CODES.index("PP5")].sum() == 0.0


# -- losses ---------------------------------------------------------------------------

def test_a_soft_target_is_best_matched_by_splitting_the_probability():
    """A "Pathogenic/Likely pathogenic" row is fitted by 0.5/0.5, not by picking one side."""
    target = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0]])
    split_evenly = torch.tensor([[5.0, 5.0, 0.0, 0.0, 0.0]])
    one_side = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])
    assert float(soft_cross_entropy(split_evenly, target)) < float(soft_cross_entropy(one_side, target))


def test_the_ordinal_term_punishes_a_far_mistake_more_than_a_near_one():
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])          # pathogenic
    near = torch.tensor([[0.0, 5.0, 0.0, 0.0, 0.0]])            # likely pathogenic
    far = torch.tensor([[0.0, 0.0, 0.0, 0.0, 5.0]])             # benign
    assert float(ordinal_emd(near, target)) < float(ordinal_emd(far, target))


def test_unsupervised_rows_and_inapplicable_codes_do_not_contribute():
    outputs = {"class_logits": torch.zeros(2, 5, requires_grad=True),
               "acmg_logits": torch.zeros(2, len(CODES), requires_grad=True)}
    batch = {"class_target": torch.zeros(2, 5),                  # no class supervision at all
             "acmg_target": torch.ones(2, len(CODES)),
             "acmg_supervised": torch.zeros(2),                  # ...and no code supervision
             "acmg_applicable": torch.ones(2, len(CODES))}
    total, parts = multitask_loss(outputs, batch, LossWeights())
    assert float(total.detach()) == 0.0 and "classify" not in parts


# -- tokenisation cache -------------------------------------------------------------------

def test_tokenising_twice_reuses_the_cache_and_a_new_tokenizer_does_not(tmp_path, backbone):
    texts = ["the variant is absent from gnomAD", "it segregated with disease"]
    before = tokenizer_fingerprint(backbone.tokenizer)
    first = tokenize_cached(texts, backbone, 16, tmp_path, "docs")
    # Using the tokenizer must not change its identity, or nothing would ever hit the cache.
    assert tokenizer_fingerprint(backbone.tokenizer) == before
    second = tokenize_cached(texts, backbone, 16, tmp_path, "docs")
    assert np.array_equal(first["input_ids"], second["input_ids"])
    assert len(list(tmp_path.iterdir())) == 1
    tokenize_cached(texts, backbone, 24, tmp_path, "docs")
    assert len(list(tmp_path.iterdir())) == 2        # a different maximum length, a different entry
    assert first["meta"]["tokenizer"] == tokenizer_fingerprint(backbone.tokenizer)
    assert "truncated_fraction" in first["meta"]


def test_batches_carry_only_what_the_model_needs(backbone):
    examples = _examples()
    tensors = assemble_documents(examples, backbone, 16)
    batch = to_batch(tensors, np.array([0, 1]), torch.device("cpu"))
    assert set(batch) >= {"input_ids", "attention_mask", "class_target"}
    assert batch["class_target"].shape == (2, 5)
    assert tensors.binary.shape == (len(examples),)


# -- fine-tuning ---------------------------------------------------------------------------

def _finetune_config(tmp_path, **kwargs) -> FinetuneConfig:
    examples = _examples()
    path = tmp_path / "examples.parquet"
    examples.to_parquet(path, index=False)
    defaults = dict(run_name="test", out_dir=str(tmp_path / "run"), examples=str(path),
                    records=str(tmp_path / "missing"), backbone="tiny-bert", max_length=16,
                    peft={"kind": "full", "allow_full": True}, embedding_dim=8,
                    use_structured=False, train={"epochs": 1, "batch_size": 4, "lr": 1e-3},
                    seed=3, cache_dir=str(tmp_path / "cache"))
    return FinetuneConfig(**(defaults | kwargs))


def test_a_dry_run_builds_everything_and_trains_nothing(tmp_path):
    report = run_finetune(_finetune_config(tmp_path), dry_run=True)
    assert report["trained"] is False
    assert report["finite_loss"] and report["checkpoint_roundtrip_identical"]
    assert report["probs_shape"][1] == 5
    assert not (tmp_path / "run" / "predictions.parquet").exists()


def test_training_writes_predictions_metrics_and_a_registry_line(tmp_path, monkeypatch):
    registry = tmp_path / "registry.jsonl"
    monkeypatch.setattr("vpdl.slm.modeling.finetune.REGISTRY", registry)
    result = run_finetune(_finetune_config(tmp_path))
    assert result["run_id"] and Path(result["out"]).exists()
    predictions = pd.read_parquet(tmp_path / "run" / "predictions.parquet")
    assert set(f"p_{c}" for c in CLASSES) <= set(predictions.columns)
    assert np.allclose(predictions[[f"p_{c}" for c in CLASSES]].sum(1), 1.0, atol=1e-5)
    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert "test" in metrics and "five_class" in metrics["test"]
    record = json.loads(registry.read_text().splitlines()[0])
    assert record["kind"] == "slm_finetune" and record["experiment_id"]
    assert "calibration_fitted" in metrics["test"]


def test_a_resumed_run_refuses_a_different_dataset(tmp_path):
    config = _finetune_config(tmp_path, train={"epochs": 1, "batch_size": 4, "lr": 1e-3,
                                               "resume": True})
    run_finetune(config)
    other = _examples(6)
    other_path = tmp_path / "other.parquet"
    other.to_parquet(other_path, index=False)
    changed = _finetune_config(tmp_path, examples=str(other_path),
                               train={"epochs": 1, "batch_size": 4, "lr": 1e-3, "resume": True})
    with pytest.raises(ValueError, match="different configuration"):
        run_finetune(changed)


# -- continued pretraining -------------------------------------------------------------------

def _corpus(tmp_path) -> str:
    folder = tmp_path / "corpus"
    folder.mkdir(exist_ok=True)
    documents = [{"id": f"doc{i}", "source": "test",
                  "text": "the variant is absent from population databases and segregated with disease "
                          f"in {i} families with colorectal cancer"} for i in range(60)]
    with (folder / "train-00000.jsonl").open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document) + "\n")
    with (folder / "val.jsonl").open("w", encoding="utf-8") as handle:
        for document in documents[:10]:
            handle.write(json.dumps(document) + "\n")
    return str(folder)


def _pretrain_config(tmp_path, **kwargs) -> PretrainConfig:
    corpus = _corpus(tmp_path)
    backbone = load_backbone(BackboneSpec("tiny-bert"))
    pack_for_tokenizer(corpus, backbone.tokenizer, tmp_path / "tokens")
    defaults = dict(backbone="tiny-bert", corpus_dir=corpus, token_dir=str(tmp_path / "tokens"),
                    out_dir=str(tmp_path / "pretrain"), context=32, micro_batch=2, total_steps=2,
                    warmup_steps=1, eval_every=2, checkpoint_every=1, log_every=1, seed=5)
    return PretrainConfig(**(defaults | kwargs))


def test_masking_hides_tokens_and_supervises_only_those():
    tokenizer = tiny_tokenizer()
    inputs = torch.randint(5, 50, (4, 16))
    masked, labels = mask_tokens(inputs, tokenizer, 0.5,
                                 torch.Generator().manual_seed(0))
    supervised = labels != -100
    assert supervised.any()
    assert (labels[~supervised] == -100).all()
    assert (masked[~supervised] == inputs[~supervised]).all()


def test_token_files_record_the_tokenizer_that_made_them(tmp_path):
    _pretrain_config(tmp_path)          # packs the corpus with the tiny tokenizer
    meta = json.loads((tmp_path / "tokens" / "data_meta.json").read_text())
    assert meta["tokenizer_fingerprint"] == tokenizer_fingerprint(tiny_tokenizer())
    assert meta["splits"]["train"]["tokens"] > 0


def test_pretraining_with_a_foreign_tokenizers_tokens_is_refused(tmp_path):
    config = _pretrain_config(tmp_path)
    meta_path = tmp_path / "tokens" / "data_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["tokenizer_fingerprint"] = "0000000000000000"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="different tokenizer"):
        run_pretrain(config)


def test_a_pretraining_dry_run_reports_the_objective_and_trains_nothing(tmp_path):
    report = run_pretrain(_pretrain_config(tmp_path), dry_run=True)
    assert report["objective"] == "mlm" and report["trained"] is False
    assert report["finite_loss"] and report["supervised_positions"] > 0
    assert not (tmp_path / "pretrain" / "checkpoints").exists()


def test_pretraining_saves_a_model_a_log_and_resumes_from_its_checkpoint(tmp_path):
    config = _pretrain_config(tmp_path, total_steps=2)
    result = run_pretrain(config)
    assert result["step"] == 2 and (tmp_path / "pretrain" / "final").exists()
    assert (tmp_path / "pretrain" / "log.jsonl").read_text().strip()
    longer = _pretrain_config(tmp_path, total_steps=2)
    again = run_pretrain(longer)                       # already finished: resumes, nothing to do
    assert again["step"] == 2
    run_json = json.loads((tmp_path / "pretrain" / "run.json").read_text())
    assert run_json["objective"] == "mlm" and run_json["git"] is not None


def test_a_decoder_backbone_pretrains_causally(tmp_path):
    corpus = _corpus(tmp_path)
    backbone = load_backbone(BackboneSpec("tiny-llama"))
    pack_for_tokenizer(corpus, backbone.tokenizer, tmp_path / "decoder_tokens")
    config = PretrainConfig(backbone="tiny-llama", corpus_dir=corpus,
                            token_dir=str(tmp_path / "decoder_tokens"),
                            out_dir=str(tmp_path / "decoder"), context=32, micro_batch=2,
                            total_steps=1, warmup_steps=1, eval_every=1, checkpoint_every=1)
    assert run_pretrain(config, dry_run=True)["objective"] == "clm"


# -- sizing -------------------------------------------------------------------------------

def test_size_candidates_are_estimates_with_their_assumptions_on_the_record():
    report = size_report(corpus_tokens=2e9, supervised_examples=50_000, memory_gib=120)
    assert report["assumptions"]["tflops"] and "ESTIMATES" in report["assumptions"]["note"]
    names = {c["name"] for c in report["candidates"]}
    assert {"biomedical-encoder-base", "from-scratch-small"} <= names
    for candidate in report["candidates"]:
        assert candidate["memory_lora_gib"] <= candidate["memory_full_finetune_gib"]
    assert "overfits" in report["supervision_note"]


def test_memory_grows_with_the_model_and_the_batch():
    small = estimate_memory_gib(110, 12, 768, 512, 8)
    large = estimate_memory_gib(335, 24, 1024, 512, 8)
    bigger_batch = estimate_memory_gib(110, 12, 768, 512, 32)
    assert small["total_gib"] < large["total_gib"]
    assert small["activations_gib"] < bigger_batch["activations_gib"]
