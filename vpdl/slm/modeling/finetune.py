"""Supervised (multi-task) training of the genomic SLM.

Training itself is ``vpdl.dl.trainer.Trainer`` — the DL branch's loop, reused
unchanged: bf16 where the device supports it, gradient accumulation with a
correct short last window, warm-up + cosine, early stopping on a validation
score, and resumable checkpoints (model, optimiser, scheduler, scaler, RNG,
best weights). The dataset version, the split hash and the head configuration
go into the checkpoint's identity, so a resume against different data or a
different model is refused rather than silently mixing two experiments.

What this module adds: assembling batches from the example tables, the
multi-task loss, the validation score (pathogenic-vs-benign ROC-AUC when the
split has both classes, else macro F1), predictions and embeddings for every
evaluated row, calibration fitted on validation only, and one registry record
per run.

Evidence units, when the evidence heads are on, are sampled per step from a
generator keyed by the step's first index, so a resumed run sees the same
unit batches as an uninterrupted one.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vpdl.slm.build.features import DEFAULT_FEATURES, StructuredEncoder, feature_frame
from vpdl.slm.modeling.backbone import BackboneSpec, load_backbone
from vpdl.slm.modeling.data import (TaskTensors, assemble_documents, assemble_units, to_batch,
                                    unit_batch)
from vpdl.slm.modeling.losses import LossWeights, multitask_loss
from vpdl.slm.modeling.multitask import GenomicSLM, HeadConfig, applicability_mask
from vpdl.slm.modeling.peft import apply_peft
from vpdl.slm.modeling.predict import mc_dropout_samples, predict_documents, predict_units
from vpdl.slm.schema import CLASSES

__all__ = ["FinetuneConfig", "run_finetune", "REGISTRY"]

REGISTRY = Path("runs/slm_genomic/registry.jsonl")
EVAL_SPLITS = ("val", "mmr_val", "test", "broad_test")


@dataclass
class FinetuneConfig:
    run_name: str = "slm-finetune"
    out_dir: str = "runs/slm_genomic/finetune"
    examples: str = ""                       # parquet written by `vpdl-slm examples`
    unit_examples: str | None = None
    records: str = "data/slm_genomic"        # for variants / documents (structured features)
    backbone: str = "tiny-bert"
    pooling: str = "auto"
    max_length: int = 384
    peft: dict[str, Any] = field(default_factory=lambda: {"kind": "lora"})
    tasks: tuple[str, ...] = ("classify",)
    embedding_dim: int = 256
    dropout: float = 0.1
    features: tuple[str, ...] = DEFAULT_FEATURES
    use_structured: bool = True
    dl_outputs: str | None = None
    dl_dropout: float = 0.1
    loss: dict[str, float] = field(default_factory=dict)
    class_weights: str = "none"              # none | balanced
    train: dict[str, Any] = field(default_factory=lambda: {"epochs": 3, "batch_size": 8, "lr": 2e-5})
    unit_batch_size: int = 16
    seed: int = 42
    select_metric: str = "auto"              # auto | binary_auc | macro_f1 | neg_loss
    calibration: str = "temperature"
    mc_samples: int = 0
    cache_dir: str = "data/slm_genomic/token_cache"
    limit_rows: int | None = None
    gene_specific_acmg: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_examples(config: FinetuneConfig) -> pd.DataFrame:
    examples = pd.read_parquet(config.examples)
    if config.limit_rows:
        examples = examples.groupby("split", group_keys=False).head(config.limit_rows)
    return examples.reset_index(drop=True)


def _dl_inputs(config: FinetuneConfig, examples: pd.DataFrame, variants: pd.DataFrame):
    if not config.dl_outputs:
        return None, None
    from vpdl.slm.interface import DLInput
    dl = DLInput.from_outputs(config.dl_outputs)
    protein = examples["variant_id"].map(variants.set_index("variant_id")["protein_variant_id"])
    return dl.matrix(protein.tolist()), dl


def _class_weights(tensors: TaskTensors, rows: np.ndarray, mode: str):
    if mode != "balanced":
        return None
    import torch
    counts = tensors.class_target[rows].sum(0)
    weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1) / len(CLASSES), 0.0)
    return torch.tensor(weights, dtype=torch.float32)


def run_finetune(config: FinetuneConfig, dry_run: bool = False) -> dict[str, Any]:

    from vpdl.dl.trainer import Trainer, TrainConfig, seed_everything
    from vpdl.slm.build.records import load_tables

    started = time.time()
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)

    examples = _load_examples(config)
    tables = load_tables(config.records, ["variants", "documents"]) if Path(config.records).exists() else {}
    variants = tables.get("variants", pd.DataFrame(columns=["variant_id", "consequence", "variant_type",
                                                            "origin", "chromosome", "hgvs_p",
                                                            "protein_variant_id"]))
    backbone = load_backbone(BackboneSpec(config.backbone, config.pooling, config.max_length),
                             seed=config.seed)
    peft_summary = apply_peft(backbone.model, config.peft)

    structured, encoder = None, None
    if config.use_structured and config.features:
        raw = feature_frame(examples, variants, tables.get("documents"))
        encoder = StructuredEncoder(tuple(config.features)).fit(raw, examples["split"])
        structured = encoder.transform(raw)
    dl_arrays, dl = _dl_inputs(config, examples, variants)
    applicable = (applicability_mask(examples["gene"].fillna("").tolist()).numpy()
                  if config.gene_specific_acmg else None)
    tensors = assemble_documents(examples, backbone, config.max_length, config.cache_dir,
                                 structured, dl_arrays, applicable)
    units = None
    if config.unit_examples and "evidence" in config.tasks:
        unit_frame = pd.read_parquet(config.unit_examples)
        if config.limit_rows:
            unit_frame = unit_frame.groupby("split", group_keys=False).head(config.limit_rows)
        units = assemble_units(unit_frame.reset_index(drop=True), backbone, 128, config.cache_dir)

    heads = HeadConfig(tasks=tuple(config.tasks), embedding_dim=config.embedding_dim,
                       dropout=config.dropout,
                       cat_cardinalities=tuple(encoder.cardinalities()) if encoder else (),
                       n_numeric=len(encoder.numeric) if encoder else 0,
                       dl_dim=int(dl_arrays[0].shape[1]) if dl_arrays is not None else None,
                       dl_dropout=config.dl_dropout)
    model = GenomicSLM(backbone, heads)
    weights = LossWeights(**config.loss)

    train_rows = tensors.rows(["train", "mmr_train"])
    val_rows = tensors.rows(["val", "mmr_val"])
    if len(train_rows) == 0:
        raise ValueError(f"{config.examples}: no training rows")
    train_config = TrainConfig(checkpoint_dir=str(out / "checkpoints"), **config.train)
    train_config.extra.update({"examples": str(config.examples), "backbone": config.backbone,
                               "tasks": list(config.tasks), "features": list(config.features),
                               "dl_outputs": config.dl_outputs,
                               "split_ids": _split_digest(examples), "heads": heads.as_dict()})
    trainer = Trainer(model, train_config, seed=config.seed, run_name=config.run_name)
    device = trainer.device
    # Tensor cores for fp32 matmuls, and one-off kernel selection for these fixed
    # shapes. Off by default in PyTorch; the same call the pretraining loop makes.
    from vpdl.slm.modeling.continued import prepare_device
    accelerator = prepare_device(device)
    class_weight = _class_weights(tensors, train_rows, config.class_weights)
    if class_weight is not None:
        class_weight = class_weight.to(device)
    unit_train = units.rows(["train", "mmr_train"]) if units is not None else np.zeros(0, dtype=int)

    def batch_fn(indices: np.ndarray) -> dict[str, Any]:
        rows = train_rows[indices]
        batch = to_batch(tensors, rows, device)
        if len(unit_train):
            rng = np.random.default_rng([config.seed, int(rows[0]), len(rows)])
            chosen = unit_train[rng.integers(0, len(unit_train), size=min(config.unit_batch_size,
                                                                          len(unit_train)))]
            batch |= unit_batch(units, chosen, device)
        return batch

    def loss_fn(module, batch):
        outputs = module(**{k: v for k, v in batch.items() if not k.startswith("unit_")})
        unit_outputs = (module.forward_units(batch["unit_input_ids"], batch["unit_attention_mask"])
                        if "unit_input_ids" in batch and "evidence" in config.tasks else None)
        total, parts = multitask_loss(outputs, batch, weights, class_weight, unit_outputs)
        return total

    def score_fn(module) -> float:
        if len(val_rows) == 0:
            return float("nan")
        predictions = predict_documents(module, tensors, val_rows, device, autocast=trainer.autocast)
        return _validation_score(config.select_metric, tensors, val_rows, predictions["probs"])

    if dry_run:
        report = _dry_run(model, tensors, units, train_rows, device, batch_fn, loss_fn, trainer,
                          heads, peft_summary, config)
        report["accelerator"] = accelerator
        report["seconds"] = round(time.time() - started, 2)
        return report

    result = trainer.fit(len(train_rows), batch_fn, loss_fn, score_fn if len(val_rows) else None)
    artefacts = _evaluate_and_save(model, tensors, units, trainer, config, examples, out, heads,
                                   peft_summary | {"accelerator": accelerator}, result, started)
    return artefacts


def _split_digest(examples: pd.DataFrame) -> str:
    import hashlib
    payload = "|".join(f"{e}:{s}" for e, s in zip(examples["example_id"].astype(str),
                                                   examples["split"].astype(str)))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _validation_score(metric: str, tensors: TaskTensors, rows: np.ndarray, probs: np.ndarray) -> float:
    from vpdl.evaluate import roc_auc
    from vpdl.slm.evaluation.metrics import five_class_metrics, pathogenic_score
    binary = tensors.binary[rows]
    usable = np.isfinite(binary)
    if metric in ("auto", "binary_auc") and usable.sum() and len(np.unique(binary[usable])) == 2:
        value = roc_auc(binary[usable].astype(int), pathogenic_score(probs)[usable])
        if np.isfinite(value):
            return float(value)
    if metric in ("auto", "macro_f1"):
        value = five_class_metrics(tensors.class_index[rows], probs).get("macro_f1", float("nan"))
        if np.isfinite(value):
            return float(value)
    return float(-((probs - tensors.class_target[rows]) ** 2).sum(1).mean())


def _dry_run(model, tensors, units, train_rows, device, batch_fn, loss_fn, trainer, heads,
             peft_summary, config) -> dict[str, Any]:
    """Build everything, run ONE step on a few rows, save and reload a checkpoint. No training."""
    import torch
    rows = np.arange(min(4, len(train_rows)))
    batch = batch_fn(rows)
    model.train()
    loss = loss_fn(model, batch)
    loss.backward()
    shapes = {key: list(value.shape) for key, value in batch.items() if hasattr(value, "shape")}
    predictions = predict_documents(model, tensors, train_rows[rows], device)
    unit_check = None
    if units is not None and "evidence" in config.tasks:
        unit_check = {k: list(v.shape) for k, v in
                      predict_units(model, units, np.arange(min(4, len(units.example_ids))), device).items()}
    checkpoint = Path(trainer.config.checkpoint_dir or ".") / "dry-run"
    checkpoint.mkdir(parents=True, exist_ok=True)
    path = checkpoint / "state.pt"
    torch.save({"model": model.state_dict(), "config": config.as_dict()}, path)
    reloaded = torch.load(path, map_location="cpu", weights_only=False)
    same = all(torch.equal(v.cpu(), reloaded["model"][k].cpu()) for k, v in model.state_dict().items())
    path.unlink()
    return {"dry_run": True, "device": str(device), "batch_shapes": shapes,
            "loss": float(loss.detach()), "finite_loss": bool(np.isfinite(float(loss.detach()))),
            "probs_shape": list(predictions["probs"].shape),
            "embedding_dim": int(predictions["embedding"].shape[1]),
            "unit_outputs": unit_check, "heads": heads.as_dict(), "peft": peft_summary,
            "checkpoint_roundtrip_identical": bool(same),
            "tokenizer": tensors.token_meta,
            "rows": {"train": int(len(train_rows)), "examples": int(len(tensors.example_ids))},
            "precision": str(trainer.autocast_dtype), "trained": False}


def _evaluate_and_save(model, tensors, units, trainer, config, examples, out: Path, heads,
                       peft_summary, result, started) -> dict[str, Any]:
    from vpdl.dl.tracking import append_registry, run_record
    from vpdl.slm.evaluation.calibration import calibration_summary, fit_calibrator
    from vpdl.slm.evaluation.metrics import (binary_metrics, five_class_metrics, pathogenic_score,
                                             validation_threshold)
    from vpdl.slm.evaluation.uncertainty import evaluate_uncertainty, mutual_information

    device = trainer.device
    metrics: dict[str, Any] = {"fit": asdict(result)}
    rows_by_split = {name: tensors.rows(name) for name in EVAL_SPLITS}
    rows_by_split = {k: v for k, v in rows_by_split.items() if len(v)}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for name, rows in rows_by_split.items():
        predictions[name] = predict_documents(model, tensors, rows, device, autocast=trainer.autocast)

    calibrator = None
    validation = next((n for n in ("val", "mmr_val") if n in predictions), None)
    if validation and config.calibration:
        calibrator = fit_calibrator(config.calibration, predictions[validation]["logits"],
                                    tensors.class_index[rows_by_split[validation]],
                                    np.full(len(rows_by_split[validation]), validation))
    threshold = None
    if validation:
        rows = rows_by_split[validation]
        binary = tensors.binary[rows]
        usable = np.isfinite(binary)
        if usable.sum() and len(np.unique(binary[usable])) == 2:
            threshold = validation_threshold(binary[usable], pathogenic_score(predictions[validation]["probs"])[usable])

    frames = []
    for name, rows in rows_by_split.items():
        probs = predictions[name]["probs"]
        calibrated = calibrator.transform(predictions[name]["logits"]) if calibrator else probs
        y = tensors.class_index[rows]
        binary = tensors.binary[rows]
        usable = np.isfinite(binary)
        entry: dict[str, Any] = {
            "n": int(len(rows)),
            "five_class": five_class_metrics(y, probs),
            "five_class_calibrated": five_class_metrics(y, calibrated),
            "calibration_raw": calibration_summary(probs, y),
            "calibration_fitted": calibration_summary(calibrated, y),
            "uncertainty": evaluate_uncertainty(calibrated, y),
        }
        if usable.sum() and len(np.unique(binary[usable])) == 2:
            entry["binary"] = binary_metrics(binary[usable], pathogenic_score(calibrated)[usable],
                                             threshold, is_validation=name.endswith("val"))
        samples = None
        if config.mc_samples:
            samples = mc_dropout_samples(model, tensors, rows, device, config.mc_samples,
                                         seed=config.seed)
            entry["uncertainty"]["mutual_information_mean"] = float(mutual_information(samples).mean())
        metrics[name] = entry
        frame = pd.DataFrame(calibrated, columns=[f"p_{c}" for c in CLASSES])
        frame.insert(0, "example_id", tensors.example_ids[rows])
        frame.insert(1, "split", name)
        frame["pathogenic_score"] = pathogenic_score(calibrated)
        frame["entropy"] = -(np.clip(calibrated, 1e-12, 1) * np.log(np.clip(calibrated, 1e-12, 1))).sum(1)
        if samples is not None:
            frame["mutual_information"] = mutual_information(samples)
        frames.append(frame)
        np.save(out / f"embeddings_{name}.npy", predictions[name]["embedding"])
    if frames:
        pd.concat(frames).to_parquet(out / "predictions.parquet", index=False)

    if units is not None and "evidence" in config.tasks:
        from vpdl.slm.evaluation.metrics import multilabel_metrics, single_label_f1
        from vpdl.slm.text.evidence import EVIDENCE_TYPES, POLARITIES
        unit_rows = units.rows(["test", "broad_test"])
        if not len(unit_rows):
            unit_rows = units.rows(["val", "mmr_val"])
        if len(unit_rows):
            unit_predictions = predict_units(model, units, unit_rows, device, autocast=trainer.autocast)
            metrics["evidence_units"] = {
                "types": multilabel_metrics(units.types[unit_rows], unit_predictions["type_probs"],
                                            EVIDENCE_TYPES),
                "polarity": single_label_f1(units.polarity[unit_rows],
                                            unit_predictions["polarity_probs"].argmax(1), POLARITIES),
                "note": "targets are rule-derived weak labels (vpdl.slm.text.evidence), not gold",
            }

    model_dir = out / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save({"format": "vpdl-slm-model/1", "config": config.as_dict(), "heads": heads.as_dict(),
                "state_dict": model.state_dict(), "peft": peft_summary,
                "calibrator": getattr(calibrator, "__dict__", None),
                "threshold": threshold}, model_dir / "model.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    record = run_record("slm_finetune", config.as_dict(), seed=config.seed,
                        dataset_path=config.examples,
                        model_version={"backbone": config.backbone, "peft": peft_summary},
                        metrics={k: v.get("five_class", {}).get("macro_f1") for k, v in metrics.items()
                                 if isinstance(v, dict) and "five_class" in v},
                        checkpoint=str(model_dir / "model.pt"),
                        artefacts={"metrics": str(out / "metrics.json"),
                                   "predictions": str(out / "predictions.parquet")},
                        started=started)
    append_registry(record, REGISTRY)
    return {"metrics": metrics, "out": str(out), "experiment_id": record["experiment_id"],
            "run_id": record["run_id"]}
