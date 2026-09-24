"""Broad genomic continued pretraining: a biomedical model, further trained on this corpus.

    general pretrained model -> biomedical pretrained model -> BROAD GENOMIC CONTINUED
    PRETRAINING (here) -> custom genomic SLM -> task training (finetune.py)

Objective follows the backbone: masked language modelling for an encoder
(BERT family), causal language modelling for a decoder (our from-scratch
Llama, or any causal model). ``auto`` picks by backbone kind.

The loop is step-based, not epoch-based, because a pretraining run lasts days:
checkpoints every ``checkpoint_every`` steps AND at least every
``checkpoint_minutes``; Ctrl-C / SIGTERM finish the step, save and exit;
resume continues from (seed, step) so the batches are exactly those an
uninterrupted run would have seen. That discipline, and several of its
helpers, come from ``vpdl.slm.train`` — the from-scratch pretraining loop this
project already runs on the DGX; it is imported, not copied, and unchanged.

Packing: the corpus (``{"id", "source", "text"}`` JSONL shards, as
``vpdl slm-corpus`` writes) is tokenised once into a flat uint16 array per
split with the backbone's own tokenizer, recorded with that tokenizer's
fingerprint so token files can never be paired with a different vocabulary.
"""

from __future__ import annotations

import json
import logging
import math
import os
import signal
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from vpdl.slm.corpus import iter_texts
from vpdl.slm.modeling.backbone import resolve_spec, tokenizer_fingerprint
from vpdl.slm.train import (TokenWindows, latest_checkpoint, learning_rate, list_checkpoints,
                            missing_python_headers, same_run)

logger = logging.getLogger(__name__)

__all__ = ["PretrainConfig", "pack_for_tokenizer", "run_pretrain", "mask_tokens", "autotune",
           "prepare_device", "model_flops_per_token", "MEASURED_PEAK_TFLOPS"]

# Measured on this project's DGX Spark on 2026-09-22 (docs/RUNLOG.md): 90.1 TFLOPS
# bf16 peak matrix multiply. Throughput is reported as a share of it so that
# "the GPU is busy" is a number, not an impression.
MEASURED_PEAK_TFLOPS = 90.1


@dataclass
class PretrainConfig:
    backbone: str = "tiny-bert"
    corpus_dir: str = "data/slm_genomic/pretrain_corpus"
    token_dir: str = "data/slm_genomic/pretrain_tokens"
    out_dir: str = "runs/slm_genomic/pretrain"
    objective: str = "auto"                  # auto | mlm | clm
    context: int = 512
    micro_batch: int = 16
    grad_accum: int = 1
    total_steps: int = 10_000
    lr: float = 5e-5
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    weight_decay: float = 0.01
    beta2: float = 0.98
    grad_clip: float = 1.0
    mlm_probability: float = 0.15
    eval_every: int = 250
    eval_batches: int = 20
    checkpoint_every: int = 250
    checkpoint_minutes: float = 30.0
    keep_last: int = 3
    log_every: int = 10
    seed: int = 0
    compile: bool = False
    peft: dict[str, Any] = field(default_factory=lambda: {"kind": "full", "allow_full": True})

    @property
    def steps(self) -> int:                  # named as vpdl.slm.train's config, so learning_rate() fits
        return max(1, self.total_steps)

    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch * self.context * self.grad_accum

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def pack_for_tokenizer(corpus_dir: Path | str, tokenizer, out_dir: Path | str,
                       batch_documents: int = 2000, limit_documents: int | None = None) -> dict[str, Any]:
    """Corpus JSONL shards -> ``train.bin`` / ``val.bin`` (uint16) + ``data_meta.json``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if tokenizer.vocab_size >= 2 ** 16:
        raise ValueError(f"vocabulary of {tokenizer.vocab_size} does not fit uint16 token files")
    separator = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    counts = {}
    for split in ("train", "val"):
        tokens = documents = 0
        with (out / f"{split}.bin").open("wb") as handle:
            batch: list[str] = []

            def flush() -> None:
                nonlocal tokens, documents
                if not batch:
                    return
                for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]:
                    array = np.asarray(ids + ([separator] if separator is not None else []),
                                       dtype=np.uint16)
                    handle.write(array.tobytes())
                    tokens += len(array)
                documents += len(batch)
                batch.clear()

            for text in iter_texts(corpus_dir, split):
                batch.append(text)
                if len(batch) >= batch_documents:
                    flush()
                if limit_documents and documents >= limit_documents:
                    break
            flush()
        counts[split] = {"documents": documents, "tokens": tokens}
        logger.info("%s: %d documents -> %d tokens", split, documents, tokens)
    meta = {"splits": counts, "dtype": "uint16", "separator_id": separator,
            "vocab_size": int(tokenizer.vocab_size),
            "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "data_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def mask_tokens(inputs, tokenizer, probability: float, generator=None):
    """BERT-style masking: 80% [MASK], 10% random, 10% unchanged; labels -100 elsewhere."""
    import torch
    labels = inputs.clone()
    special = torch.zeros_like(inputs, dtype=torch.bool)
    for token_id in filter(lambda t: t is not None, [tokenizer.cls_token_id, tokenizer.sep_token_id,
                                                     tokenizer.pad_token_id]):
        special |= inputs == token_id
    probability_matrix = torch.full(labels.shape, probability, device=inputs.device)
    probability_matrix.masked_fill_(special, 0.0)
    masked = torch.bernoulli(probability_matrix, generator=generator).bool()
    labels[~masked] = -100
    replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=inputs.device),
                               generator=generator).bool() & masked
    inputs = inputs.clone()
    inputs[replaced] = tokenizer.mask_token_id
    randomised = (torch.bernoulli(torch.full(labels.shape, 0.5, device=inputs.device),
                                  generator=generator).bool() & masked & ~replaced)
    inputs[randomised] = torch.randint(len(tokenizer), labels.shape, dtype=inputs.dtype,
                                       device=inputs.device, generator=generator)[randomised]
    return inputs, labels


def prepare_device(device) -> dict[str, Any]:
    """Switch on what this accelerator can do, and report what was switched on.

    On the GB10 (and any recent NVIDIA part) three settings decide whether the
    matrix units are used at all, and all three are off by default in PyTorch:

    * ``float32_matmul_precision("high")`` — lets fp32 matmuls run on the tensor
      cores instead of the much slower fp32 path (bf16 autocast covers most of
      the model, but not everything);
    * ``cudnn.benchmark`` — picks the fastest kernel for this fixed shape once,
      which is right when every step has the same shape, as here;
    * TF32 for convolutions and matmuls.

    Nothing here changes what is computed beyond fp32 matmul rounding; the
    settings that WOULD change results (precision, seeds, batch order) are
    config, not defaults.
    """
    import torch
    applied: dict[str, Any] = {"device": str(device)}
    if device.type != "cuda":
        return applied | {"note": "CPU: no accelerator settings apply"}
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    free, total = torch.cuda.mem_get_info()
    applied |= {"gpu": torch.cuda.get_device_name(0),
                "float32_matmul_precision": "high", "tf32": True, "cudnn_benchmark": True,
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "memory_free_gib": round(free / 2 ** 30, 1),
                "memory_total_gib": round(total / 2 ** 30, 1),
                "sdpa_flash": torch.backends.cuda.flash_sdp_enabled(),
                "sdpa_mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled()}
    logger.info("accelerator: %s, %.0f GiB free, bf16=%s, TF32 on, cuDNN autotune on",
                applied["gpu"], applied["memory_free_gib"], applied["bf16_supported"])
    return applied


def model_flops_per_token(model, context: int) -> float:
    """6 x parameters + attention, the arithmetic ``vpdl.slm.model`` already uses.

    Lets a continued-pretraining run report achieved TFLOPS on the same scale as
    the from-scratch runs (docs/RUNLOG.md, 2026-09-22: 30.7 TFLOPS with compile,
    90.1 TFLOPS peak measured on this machine), so underuse is visible as a
    number rather than a feeling.
    """
    config = model.config
    layers = getattr(config, "num_hidden_layers", 0) or 0
    width = getattr(config, "hidden_size", 0) or 0
    parameters = sum(p.numel() for p in model.parameters())
    return 6 * parameters + 12 * layers * width * context


def _load_lm(spec: str, objective: str, seed: int):
    """(model, tokenizer, objective) for continued pretraining."""
    import torch
    name = resolve_spec(spec)
    if name in ("tiny-bert", "tiny-llama"):
        from transformers import BertConfig, BertForMaskedLM, LlamaConfig, LlamaForCausalLM
        from vpdl.slm.modeling.backbone import tiny_tokenizer
        tokenizer = tiny_tokenizer()
        torch.manual_seed(seed)
        if name == "tiny-bert":
            config = BertConfig(vocab_size=tokenizer.vocab_size, hidden_size=32, num_hidden_layers=2,
                                num_attention_heads=2, intermediate_size=64,
                                max_position_embeddings=256, pad_token_id=tokenizer.pad_token_id)
            return BertForMaskedLM(config), tokenizer, "mlm"
        config = LlamaConfig(vocab_size=tokenizer.vocab_size, hidden_size=32, num_hidden_layers=2,
                             num_attention_heads=2, intermediate_size=64, max_position_embeddings=256,
                             pad_token_id=tokenizer.pad_token_id)
        return LlamaForCausalLM(config), tokenizer, "clm"
    scheme, _, location = name.partition(":")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
    config = AutoConfig.from_pretrained(location)
    kind = "clm" if config.model_type in ("llama", "gpt2", "mistral", "qwen2", "gemma", "gemma2") else "mlm"
    objective = kind if objective == "auto" else objective
    tokenizer = AutoTokenizer.from_pretrained(location)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    loader = AutoModelForMaskedLM if objective == "mlm" else AutoModelForCausalLM
    try:
        # PyTorch's fused attention. Without it a 512-token BERT step runs the
        # slow eager path and the GPU idles on memory traffic.
        model = loader.from_pretrained(location, attn_implementation="sdpa")
    except (ValueError, ImportError, TypeError) as error:
        logger.warning("SDPA attention unavailable for %s (%s); falling back to eager.",
                       location, error)
        model = loader.from_pretrained(location)
    return model, tokenizer, objective


def autotune(config: PretrainConfig, micro_batches: Sequence[int] = (),
             try_compile: bool = True, steps: int = 6,
             target_tokens_per_step: int | None = None) -> dict[str, Any]:
    """Measure this machine, then say what to put in the config.

    A default batch size chosen on a laptop wastes a 128 GiB unified-memory
    machine, and a batch size chosen by guessing wastes a day finding out. This
    runs a few short benchmarks — increasing micro-batch until it stops paying
    or runs out of memory, with and without ``torch.compile`` — and returns the
    table it measured plus the setting with the best tokens/s. It trains
    nothing and writes no checkpoint.

    ``target_tokens_per_step`` (optional) keeps the optimiser step size fixed
    while the micro-batch grows, by adjusting gradient accumulation — so the
    tuning changes speed, not what is computed.
    """
    import torch
    from dataclasses import replace

    if not micro_batches:
        micro_batches = (8, 16, 32, 64, 128, 256)
    rows: list[dict[str, Any]] = []
    for compiled in ([False, True] if try_compile else [False]):
        for micro in micro_batches:
            accum = max(1, round((target_tokens_per_step or config.tokens_per_step)
                                 / (micro * config.context))) if target_tokens_per_step else config.grad_accum
            candidate = replace(config, micro_batch=micro, grad_accum=accum, compile=compiled)
            try:
                result = run_pretrain(candidate, benchmark_steps=steps)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as error:   # noqa: PERF203
                message = str(error)
                rows.append({"micro_batch": micro, "compile": compiled, "grad_accum": accum,
                             "failed": message.split("\n")[0][:120]})
                if "out of memory" in message.lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    break                       # bigger batches will not fit either
                continue
            rows.append({k: result[k] for k in ("micro_batch", "grad_accum", "compile",
                                                "tokens_per_s", "tflops",
                                                "share_of_measured_peak", "tokens_per_step")})
            logger.info("micro_batch %d compile=%s: %d tok/s, %.1f TFLOPS (%.0f%% of peak)",
                        micro, compiled, result["tokens_per_s"], result["tflops"],
                        100 * result["share_of_measured_peak"])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    usable = [r for r in rows if "tokens_per_s" in r]
    best = max(usable, key=lambda r: r["tokens_per_s"]) if usable else None
    report = {"measured": rows, "best": best, "peak_tflops_reference": MEASURED_PEAK_TFLOPS,
              "backbone": config.backbone, "context": config.context, "steps_per_measurement": steps}
    if best:
        report["put_in_config"] = {"micro_batch": best["micro_batch"],
                                   "grad_accum": best["grad_accum"], "compile": best["compile"]}
        report["note"] = (f"{best['tokens_per_s']:,} tokens/s = {best['tflops']} TFLOPS, "
                          f"{100 * best['share_of_measured_peak']:.0f}% of this machine's measured "
                          f"bf16 peak. Under ~20% means something else is the bottleneck — check "
                          f"the accelerator block for bf16 and SDPA.")
    return report


def run_pretrain(config: PretrainConfig, dry_run: bool = False,
                 benchmark_steps: int = 0) -> dict[str, Any]:
    import torch

    from vpdl.dl.trainer import resolve_precision
    from vpdl.slm.modeling.peft import apply_peft

    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(config.token_dir)
    model, tokenizer, objective = _load_lm(config.backbone, config.objective, config.seed)
    # The encoder/decoder inside the task wrapper: HF names its attribute in base_model_prefix
    # ("bert", "roberta", "electra", "model", ...), so no family is hard-coded here.
    prefix = getattr(model, "base_model_prefix", "") or ""
    base = getattr(model, prefix, None) if prefix else None
    peft_summary = apply_peft(base if base is not None else
                              getattr(model, "bert", getattr(model, "model", model)), config.peft)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and (problem := missing_python_headers()):
        raise RuntimeError(problem)
    accelerator = prepare_device(device)
    model.to(device)
    autocast_dtype, needs_scaler = resolve_precision("auto", device)
    scaler = torch.amp.GradScaler(device.type) if needs_scaler else None
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config.lr,
        betas=(0.9, config.beta2), weight_decay=config.weight_decay,
        fused=(device.type == "cuda"))

    meta_path = data / "data_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta and meta.get("tokenizer_fingerprint") != tokenizer_fingerprint(tokenizer):
        raise ValueError(f"{data}: token files were packed with a different tokenizer "
                         f"({meta.get('tokenizer_fingerprint')}); repack with `vpdl-slm pretrain-pack`.")
    train_windows = TokenWindows(data / "train.bin", config.context, config.seed, stream=0)
    validation = data / "val.bin"
    val_windows = TokenWindows(validation, config.context, config.seed, stream=1) if validation.exists() else None

    checkpoints = out / "checkpoints"
    start = 0
    if not dry_run and not benchmark_steps:
        newest = latest_checkpoint(out)
        if newest is not None:
            state = torch.load(newest, map_location=device, weights_only=True)
            if not same_run(state["config"], config.as_dict()):
                raise ValueError(f"{newest} was written with a different configuration; use a new "
                                 "--out directory or the same settings.")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start = state["step"]
            logger.info("resumed from %s at step %d of %d", newest.name, start, config.steps)

    per_token = model_flops_per_token(model, config.context)
    stepper = torch.compile(model) if config.compile else model

    def seed_step(step: int) -> None:
        # Dropout draws from torch's global RNG. Keyed on (seed, step) like the windows and the
        # MLM masks, a resumed run makes exactly the dropout draws an uninterrupted one would —
        # without it, resume matched only until the first step after the checkpoint.
        torch.manual_seed(int(np.random.default_rng([config.seed, step, 104729]).integers(0, 2 ** 31)))

    def windows_to_batch(step: int, micro: int):
        x, _ = train_windows.batch(step, micro, config.micro_batch, device)
        generator = torch.Generator(device=device).manual_seed(
            int(np.random.default_rng([config.seed, step, micro]).integers(0, 2 ** 31)))
        if objective == "mlm":
            return mask_tokens(x, tokenizer, config.mlm_probability, generator)
        return x, x.clone()

    def step_loss(inputs, labels):
        output = stepper(input_ids=inputs, labels=labels)
        return output.loss

    def validate() -> float:
        if val_windows is None:
            return float("nan")
        model.eval()
        losses = []
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=autocast_dtype or torch.float32,
                                             enabled=autocast_dtype is not None):
            for index in range(config.eval_batches):
                x, _ = val_windows.batch(0, index, config.micro_batch, device)
                if objective == "mlm":
                    generator = torch.Generator(device=device).manual_seed(index)
                    x, labels = mask_tokens(x, tokenizer, config.mlm_probability, generator)
                else:
                    labels = x.clone()
                losses.append(float(step_loss(x, labels)))
        model.train()
        return sum(losses) / max(1, len(losses))

    if dry_run:
        model.train()
        seed_step(0)
        inputs, labels = windows_to_batch(0, 0)
        loss = step_loss(inputs, labels)
        loss.backward()
        supervised = int((labels != -100).sum()) if objective == "mlm" else int(labels.numel())
        return {"dry_run": True, "objective": objective, "backbone": config.backbone,
                "device": str(device), "precision": str(autocast_dtype), "loss": float(loss.detach()),
                "finite_loss": bool(math.isfinite(float(loss.detach()))),
                "input_shape": list(inputs.shape), "supervised_positions": supervised,
                "peft": peft_summary, "data_meta": meta, "accelerator": accelerator,
                "tokens_per_step": config.tokens_per_step, "trained": False}

    stop = {"requested": False}

    def request_stop(signum, _frame):
        logger.warning("signal %s: finishing this step, saving, then exiting.", signum)
        stop["requested"] = True

    previous = {}
    for name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, name):
            try:
                previous[name] = signal.signal(getattr(signal, name), request_stop)
            except (ValueError, OSError):
                pass

    def save(step: int) -> Path:
        checkpoints.mkdir(parents=True, exist_ok=True)
        target = checkpoints / f"step-{step:07d}.pt"
        temporary = target.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                    "config": config.as_dict()}, temporary)
        os.replace(temporary, target)
        for old in list_checkpoints(out)[:-max(1, config.keep_last)]:
            old.unlink()
        return target

    end = start + benchmark_steps if benchmark_steps else config.steps
    if not benchmark_steps and start >= end:
        # Resumed onto a finished run: say so and leave the checkpoint alone.
        logger.info("%s: already at step %d of %d; nothing to do.", out, start, config.steps)
        return {"step": start, "already_finished": True, "objective": objective,
                "final": str(out / "final"), "backbone_for_finetune": f"hf:{out / 'final'}"}
    if not benchmark_steps:
        from vpdl.provenance import _git_state
        (out / "run.json").write_text(json.dumps({
            "config": config.as_dict(), "objective": objective, "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "precision": str(autocast_dtype), "peft": peft_summary, "data_meta": meta,
            "accelerator": accelerator, "flops_per_token": per_token,
            "tokens_per_step": config.tokens_per_step,
            "torch": torch.__version__, "git": _git_state(),
            "parameters": sum(p.numel() for p in model.parameters()),
        }, indent=2, default=str))
    model.train()
    started = last_saved = time.time()
    last: dict[str, Any] = {}
    timings: list[float] = []
    try:
        with (out / ("benchmark.jsonl" if benchmark_steps else "log.jsonl")).open("a") as log:
            for step in range(start, end):
                began = time.time()
                seed_step(step)
                rate = learning_rate(step, config)
                for group in optimizer.param_groups:
                    group["lr"] = rate
                optimizer.zero_grad(set_to_none=True)
                total = 0.0
                for micro in range(config.grad_accum):
                    inputs, labels = windows_to_batch(step, micro)
                    with torch.autocast(device_type=device.type,
                                        dtype=autocast_dtype or torch.float32,
                                        enabled=autocast_dtype is not None):
                        loss = step_loss(inputs, labels) / config.grad_accum
                    (scaler.scale(loss) if scaler else loss).backward()
                    total += float(loss.detach())
                if scaler:
                    scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                if not math.isfinite(total):
                    raise RuntimeError(f"loss became {total} at step {step}; stopped with the last "
                                       f"checkpoint ({latest_checkpoint(out)}) intact — lower --lr "
                                       "and resume.")
                seconds = time.time() - began
                timings.append(seconds)
                done = step + 1
                tokens_per_s = config.tokens_per_step / seconds
                tflops = per_token * tokens_per_s / 1e12
                last = {"step": done, "loss": round(total, 4), "lr": rate,
                        "grad_norm": round(float(norm), 3),
                        "tokens_per_s": round(tokens_per_s),
                        "tflops": round(tflops, 1),
                        "share_of_measured_peak": round(tflops / MEASURED_PEAK_TFLOPS, 3),
                        "hours_left": round((end - done) * seconds / 3600, 2)}
                if done % config.eval_every == 0 or done == end:
                    last["val_loss"] = round(validate(), 4)
                if done % config.log_every == 0 or "val_loss" in last or done == end:
                    log.write(json.dumps(last) + "\n")
                    log.flush()
                    logger.info("step %d/%d loss %.4f%s  %.0f tok/s  %.1f TFLOPS (%.0f%% of peak)"
                                "  %.1f h left", done, end, total,
                                f" val {last['val_loss']:.4f}" if "val_loss" in last else "",
                                last["tokens_per_s"], last["tflops"],
                                100 * last["share_of_measured_peak"], last["hours_left"])
                if benchmark_steps:
                    continue
                if (done % config.checkpoint_every == 0 or done == end or stop["requested"]
                        or time.time() - last_saved >= config.checkpoint_minutes * 60):
                    saved = save(done)
                    last_saved = time.time()
                    if stop["requested"]:
                        logger.warning("stopped at step %d; checkpoint %s", done, saved)
                        return {"stopped": True, "step": done, "checkpoint": str(saved)}
    finally:
        for name, handler in previous.items():
            signal.signal(getattr(signal, name), handler)

    if benchmark_steps:
        steady = timings[2:] or timings
        seconds = sum(steady) / len(steady)
        tokens_per_s = config.tokens_per_step / seconds
        return {"tokens_per_s": round(tokens_per_s),
                "tflops": round(per_token * tokens_per_s / 1e12, 1),
                "share_of_measured_peak": round(per_token * tokens_per_s / 1e12
                                                / MEASURED_PEAK_TFLOPS, 3),
                "projected_hours": round(config.steps * seconds / 3600, 1),
                "micro_batch": config.micro_batch, "grad_accum": config.grad_accum,
                "context": config.context, "compile": config.compile,
                "tokens_per_step": config.tokens_per_step,
                "objective": objective, "steps_measured": len(timings),
                "accelerator": accelerator}
    final = out / "final"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    return {**last, "final": str(final), "objective": objective,
            "hours": round((time.time() - started) / 3600, 2),
            "backbone_for_finetune": f"hf:{final}"}
