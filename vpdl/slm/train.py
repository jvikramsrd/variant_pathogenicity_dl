"""Pretraining loop: resumable, logged, measured.

* **Resumable.** A checkpoint (weights, optimiser, step) is written atomically
  every ``checkpoint_every`` steps. Training data is drawn deterministically
  from (seed, step), so a run stopped on day 3 and resumed continues exactly
  as if it had never stopped; ``tests/slm`` checks the weights match exactly.
* **Logged.** ``log.jsonl``: loss, learning rate, gradient norm, tokens/s,
  achieved TFLOPS and time remaining; validation loss on the same fixed
  held-out batches every time, so the numbers are comparable across the run.
* **Guarded.** A non-finite loss stops the run with the last good checkpoint
  intact, instead of spending days training a broken model.
* **Measured before committing.** ``benchmark_steps`` runs a few steps, reports
  the real speed and projects the full run's time, and saves nothing.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from vpdl.slm.model import build, count_parameters, flops_per_token
from vpdl.slm.tokenizer import EOS, SPECIAL_TOKENS, load_tokenizer

logger = logging.getLogger(__name__)

__all__ = ["TrainConfig", "learning_rate", "TokenWindows", "train"]


@dataclass(frozen=True)
class TrainConfig:
    size: str = "small"
    context: int = 2048
    micro_batch: int = 16
    tokens_per_step: int = 524_288            # 2^19 tokens per optimiser step, as GPT-3 small
    total_tokens: int = 2_500_000_000         # ~20 tokens per parameter for "small"
    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 20
    checkpoint_every: int = 250
    log_every: int = 10
    seed: int = 0
    compile: bool = False

    @property
    def accumulation(self) -> int:
        return max(1, self.tokens_per_step // (self.micro_batch * self.context))

    @property
    def step_tokens(self) -> int:
        return self.micro_batch * self.context * self.accumulation

    @property
    def steps(self) -> int:
        return max(1, self.total_tokens // self.step_tokens)


def learning_rate(step: int, config: TrainConfig) -> float:
    """Linear warm-up, then cosine decay to ``min_lr_ratio`` of the peak."""
    if step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps
    span = max(1, config.steps - config.warmup_steps)
    progress = min(1.0, (step - config.warmup_steps) / span)
    floor = config.min_lr_ratio
    return config.lr * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


class TokenWindows:
    """Random ``context + 1``-token windows from a packed token file.

    The windows for (step, micro-batch) depend only on the seed, so resuming at
    step N reproduces exactly the batches an uninterrupted run would have seen.
    """

    def __init__(self, path: Path | str, context: int, seed: int, stream: int):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        if len(self.data) <= context + 1:
            raise ValueError(f"{path}: {len(self.data)} tokens is shorter than one window.")
        self.context, self.seed, self.stream = context, seed, stream

    def batch(self, step: int, micro: int, size: int, device):
        import torch
        rng = np.random.default_rng([self.seed, self.stream, step, micro])
        starts = rng.integers(0, len(self.data) - self.context - 1, size=size)
        windows = np.stack([self.data[s:s + self.context + 1] for s in starts]).astype(np.int64)
        x = torch.from_numpy(windows[:, :-1]).to(device, non_blocking=True)
        y = torch.from_numpy(windows[:, 1:]).to(device, non_blocking=True)
        return x, y


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def missing_python_headers() -> str | None:
    """The fix to print if Triton cannot build its GPU launcher, else None.

    Recent PyTorch routes some ordinary operations (the rotary position
    embedding's outer product among them) through Triton kernels even without
    ``torch.compile``, and Triton compiles a small C helper on first use that
    needs ``Python.h``. Without it the first training step dies deep inside
    Triton (first DGX run, 2026-09-22); checking up front turns that into one line.
    """
    import sys
    import sysconfig
    header = Path(sysconfig.get_paths()["include"]) / "Python.h"
    if header.exists():
        return None
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    return (f"PyTorch's GPU kernels (Triton) need the Python development headers, and "
            f"{header} is missing. Install them once, then re-run: "
            f"sudo apt install python{version}-dev")


def _loss(model, x, y):
    import torch.nn.functional as F
    logits = model(input_ids=x).logits
    return F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))


def train(data_dir: Path | str, out_dir: Path | str, config: TrainConfig,
          device: str | None = None, benchmark_steps: int = 0) -> dict:
    import torch

    data, out = Path(data_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(data / "tokenizer.json")
    vocab = tokenizer.get_vocab_size()
    special = {t: tokenizer.token_to_id(t) for t in SPECIAL_TOKENS}
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    on_gpu = device.type == "cuda"
    if on_gpu and (problem := missing_python_headers()):
        raise RuntimeError(problem)

    model = build(config.size, vocab, config.context, special, config.seed).to(device)
    model.config.use_cache = False
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": config.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=config.lr, betas=(0.9, config.beta2), eps=1e-8, fused=on_gpu)

    checkpoint = out / "checkpoint.pt"
    start = 0
    if checkpoint.exists() and not benchmark_steps:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state["config"] != asdict(config):
            raise ValueError(f"{checkpoint} was written with a different configuration; "
                             "use a new --out directory or the same settings.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start = state["step"]
        logger.info("resumed at step %d of %d", start, config.steps)

    stepper = torch.compile(model) if config.compile else model
    train_windows = TokenWindows(data / "train.bin", config.context, config.seed, stream=0)
    val_windows = TokenWindows(data / "val.bin", config.context, config.seed, stream=1)
    parameters = count_parameters(config.size, vocab, config.context)
    per_token = flops_per_token(config.size, vocab, config.context)
    end = start + benchmark_steps if benchmark_steps else config.steps

    if not benchmark_steps:
        (out / "run.json").write_text(json.dumps({
            "config": asdict(config), "parameters": parameters, "steps": config.steps,
            "tokens_per_step": config.step_tokens, "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if on_gpu else None,
            "torch": torch.__version__, "git_commit": _git_commit(),
            "data_meta": json.loads((data / "data_meta.json").read_text())
            if (data / "data_meta.json").exists() else None,
        }, indent=2))
    logger.info("%s: %.1fM parameters, %d steps of %d tokens, accumulation %d, on %s",
                config.size, parameters / 1e6, config.steps, config.step_tokens,
                config.accumulation, device)

    def autocast():
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=on_gpu)

    def validate() -> float:
        model.eval()
        losses = []
        with torch.no_grad(), autocast():
            for index in range(config.eval_batches):
                x, y = val_windows.batch(0, index, config.micro_batch, device)
                losses.append(float(_loss(stepper, x, y)))
        model.train()
        return sum(losses) / len(losses)

    def save(step: int) -> None:
        temporary = checkpoint.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": step, "config": asdict(config)}, temporary)
        os.replace(temporary, checkpoint)              # never a half-written checkpoint

    model.train()
    timings: list[float] = []
    last: dict = {}
    started = time.time()
    with (out / ("benchmark.jsonl" if benchmark_steps else "log.jsonl")).open("a") as log:
        for step in range(start, end):
            began = time.time()
            rate = learning_rate(step, config)
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            total = torch.zeros((), device=device)
            for micro in range(config.accumulation):
                x, y = train_windows.batch(step, micro, config.micro_batch, device)
                with autocast():
                    loss = _loss(stepper, x, y) / config.accumulation
                loss.backward()
                total += loss.detach()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            if on_gpu:
                torch.cuda.synchronize()
            value = float(total)
            if not math.isfinite(value):
                raise RuntimeError(f"Loss became {value} at step {step}; stopped. The last "
                                   f"checkpoint ({checkpoint}) is intact — lower --lr and resume.")
            seconds = time.time() - began
            timings.append(seconds)
            done = step + 1
            last = {"step": done, "loss": round(value, 4), "lr": rate,
                    "grad_norm": round(float(norm), 3),
                    "tokens_per_s": round(config.step_tokens / seconds),
                    "tflops": round(per_token * config.step_tokens / seconds / 1e12, 1),
                    "tokens_seen": done * config.step_tokens,
                    "hours_left": round((end - done) * seconds / 3600, 2)}
            if done % config.eval_every == 0 or done == end:
                last["val_loss"] = round(validate(), 4)
            if done % config.log_every == 0 or "val_loss" in last or done == end:
                log.write(json.dumps(last) + "\n")
                log.flush()
                logger.info("step %d/%d loss %.3f%s  %.0f tok/s  %.1f TFLOPS  %.1f h left",
                            done, end, value,
                            f" val {last['val_loss']:.3f}" if "val_loss" in last else "",
                            last["tokens_per_s"], last["tflops"], last["hours_left"])
            if not benchmark_steps and (done % config.checkpoint_every == 0 or done == end):
                save(done)

    if benchmark_steps:
        steady = timings[2:] or timings            # skip start-up / compilation steps
        seconds = sum(steady) / len(steady)
        return {"size": config.size, "parameters": parameters, "context": config.context,
                "micro_batch": config.micro_batch, "tokens_per_s": round(config.step_tokens / seconds),
                "tflops": round(per_token * config.step_tokens / seconds / 1e12, 1),
                "projected_hours": round(config.steps * seconds / 3600, 1),
                "for_tokens": config.total_tokens}

    final = out / "final"
    model.config.use_cache = True            # off for training, on for generating answers
    model.save_pretrained(final)
    from transformers import PreTrainedTokenizerFast
    PreTrainedTokenizerFast(tokenizer_file=str(data / "tokenizer.json"), eos_token=EOS,
                            bos_token=EOS, pad_token="<|pad|>").save_pretrained(final)
    return {**last, "parameters": parameters, "final": str(final),
            "hours": round((time.time() - started) / 3600, 2)}
