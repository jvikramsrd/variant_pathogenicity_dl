"""Pretraining loop: resumable, logged, measured.

* **Checkpointed.** A full checkpoint (weights, optimiser, step) is written
  atomically every ``checkpoint_every`` steps **and** at least every
  ``checkpoint_minutes``, whichever comes first, and the last ``keep_last`` are
  kept in ``checkpoints/`` so there is always an earlier one to roll back to.
* **Stops cleanly.** Ctrl-C, ``kill`` or a system shutdown (SIGINT/SIGTERM)
  finishes the current step, saves a checkpoint and exits — nothing is lost.
  A second Ctrl-C aborts at once, without saving.
* **Resumable, exactly.** Training data is drawn deterministically from
  (seed, step), so a run resumed from a checkpoint continues exactly as if it
  had never stopped; ``tests/slm`` checks the final weights match bit for bit.
  ``resume_from`` rolls back to an earlier checkpoint; the newer ones are moved
  aside, not deleted.
* **Milestones.** Every ``milestone_every`` steps the weights alone are saved in
  Hugging Face format (``milestones/step-NNNNNNN``), loadable for evaluation
  while training continues.
* **Logged.** ``log.jsonl``: loss, learning rate, gradient norm, tokens/s,
  achieved TFLOPS, time remaining, and validation loss on the same fixed
  held-out batches every time.
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
import re
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from vpdl.slm.model import build, count_parameters, flops_per_token
from vpdl.slm.tokenizer import EOS, SPECIAL_TOKENS, load_tokenizer

logger = logging.getLogger(__name__)

__all__ = ["TrainConfig", "learning_rate", "TokenWindows", "same_run",
           "list_checkpoints", "latest_checkpoint", "train"]


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
    checkpoint_minutes: float = 30.0          # ...or sooner: never more than this much work at risk
    keep_last: int = 3                        # full checkpoints kept (~1.3 GB each for "small")
    milestone_every: int = 1000               # weights-only snapshots kept for evaluation; 0 = off
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


# Settings that change how a run is run — speed, bookkeeping — but not what it
# computes. A run may be resumed with these changed (e.g. dropping --compile
# after a compiler problem). Measured on the DGX 2026-09-22: compiled and eager
# runs gave identical losses.
BOOKKEEPING = frozenset({"compile", "checkpoint_every", "checkpoint_minutes", "keep_last",
                         "milestone_every", "eval_every", "eval_batches", "log_every"})


def same_run(saved: dict, current: dict) -> bool:
    """Whether a checkpoint's settings match this run's, ignoring bookkeeping ones."""
    keys = (set(saved) | set(current)) - BOOKKEEPING
    return all(saved.get(key) == current.get(key) for key in keys)


_CHECKPOINT = re.compile(r"step-(\d+)\.pt$")


def list_checkpoints(out_dir: Path | str) -> list[Path]:
    """Full checkpoints of a run, oldest first."""
    folder = Path(out_dir) / "checkpoints"
    found = [(int(m.group(1)), p) for p in folder.glob("step-*.pt")
             if (m := _CHECKPOINT.search(p.name))]
    return [path for _, path in sorted(found)]


def latest_checkpoint(out_dir: Path | str) -> Path | None:
    found = list_checkpoints(out_dir)
    return found[-1] if found else None


# Set by Ctrl-C / SIGTERM; checked once per step.
_STOP = threading.Event()


def _request_stop(signum, frame) -> None:
    if _STOP.is_set():
        raise KeyboardInterrupt("second interrupt: stopping without saving")
    _STOP.set()
    logger.warning("Stop requested: finishing this step, saving a checkpoint, then exiting. "
                   "Press Ctrl-C again to abort without saving.")


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


def _load_state(torch, out: Path, resume_from: Path | str | None, device):
    """The checkpoint to resume from, or None for a fresh start.

    Explicit ``resume_from``: that file, or an error. Otherwise the newest
    readable checkpoint — an unreadable newest one (should never happen: saves
    are atomic) falls back to the one before it, loudly.
    """
    if resume_from is not None:
        path = Path(resume_from)
        if not path.exists():
            raise FileNotFoundError(f"--resume-from {path} does not exist.")
        return path, torch.load(path, map_location=device, weights_only=True)
    # Before 2026-09-22 a run kept one `checkpoint.pt` beside the log. A run started
    # then must continue, not silently restart from step 0.
    legacy = [out / "checkpoint.pt"] if (out / "checkpoint.pt").exists() else []
    for path in [*reversed(list_checkpoints(out)), *legacy]:
        try:
            return path, torch.load(path, map_location=device, weights_only=True)
        except Exception as error:                      # noqa: BLE001 — reported, then skipped
            logger.error("%s is unreadable (%s); trying the previous checkpoint.", path, error)
    return None


def train(data_dir: Path | str, out_dir: Path | str, config: TrainConfig,
          device: str | None = None, benchmark_steps: int = 0,
          resume_from: Path | str | None = None) -> dict:
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

    checkpoints = out / "checkpoints"
    start = 0
    loaded = None if benchmark_steps else _load_state(torch, out, resume_from, device)
    if loaded is not None:
        path, state = loaded
        if not same_run(state["config"], asdict(config)):
            raise ValueError(f"{path} was written with a different configuration; "
                             "use a new --out directory or the same settings.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start = state["step"]
        if resume_from is not None:
            # Rolling back: newer checkpoints belong to the abandoned path. Move
            # them aside (never delete) so "newest" means this path from now on.
            newer = [p for p in list_checkpoints(out)
                     if int(_CHECKPOINT.search(p.name).group(1)) > start]
            if newer:
                aside = checkpoints / f"abandoned-{time.strftime('%Y%m%d-%H%M%S')}"
                aside.mkdir(parents=True)
                for p in newer:
                    shutil.move(str(p), aside / p.name)
                logger.warning("rolled back to step %d; %d newer checkpoints moved to %s",
                               start, len(newer), aside)
        logger.info("resumed from %s at step %d of %d", path.name, start, config.steps)

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

    def save(step: int) -> Path:
        checkpoints.mkdir(parents=True, exist_ok=True)
        target = checkpoints / f"step-{step:07d}.pt"
        temporary = target.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": step, "config": asdict(config)}, temporary)
        os.replace(temporary, target)                  # never a half-written checkpoint
        for old in list_checkpoints(out)[:-max(1, config.keep_last)]:
            old.unlink()
        return target

    def milestone(step: int) -> None:
        folder = out / "milestones" / f"step-{step:07d}"
        model.config.use_cache = True
        model.save_pretrained(folder)                  # weights only, loadable for evaluation
        model.config.use_cache = False

    # Ctrl-C / kill / shutdown: finish the step, save, exit. Only in real runs,
    # and only where signals can be installed (the main thread).
    previous_handlers = {}
    if not benchmark_steps:
        _STOP.clear()
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                previous_handlers[signum] = signal.signal(signum, _request_stop)
            except (ValueError, OSError):
                pass

    model.train()
    timings: list[float] = []
    last: dict = {}
    started = last_saved = time.time()
    try:
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
                    newest = latest_checkpoint(out)
                    raise RuntimeError(
                        f"Loss became {value} at step {step}; stopped. The last checkpoint "
                        f"({newest}) is intact — lower --lr and resume, or roll back with "
                        "--resume-from an earlier one.")
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
                if benchmark_steps:
                    continue
                if config.milestone_every and done % config.milestone_every == 0:
                    milestone(done)
                stopping = _STOP.is_set()
                if (done % config.checkpoint_every == 0 or done == end or stopping
                        or time.time() - last_saved >= config.checkpoint_minutes * 60):
                    saved = save(done)
                    last_saved = time.time()
                    if stopping:
                        log.write(json.dumps({"event": "stopped", "step": done}) + "\n")
                        logger.warning("Stopped at step %d; checkpoint %s. Run the same "
                                       "command to continue.", done, saved)
                        return {"stopped": True, "step": done, "checkpoint": str(saved),
                                "resume": "run the same command again"}
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

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
