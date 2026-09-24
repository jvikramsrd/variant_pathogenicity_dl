"""One torch training loop for every new DL model: supervised heads, PLM
fine-tuning and continued pretraining.

What it owns, so no model re-implements it:

* **precision** — bf16 autocast where the device supports it (DGX Spark GB10
  does), fp16 + GradScaler otherwise on CUDA, fp32 on CPU. bf16 keeps fp32's
  exponent range, so no loss scaling is needed and none is used;
* **gradient accumulation** with a correctly-sized last window (a short tail
  window is divided by its own length, not the nominal one — v1's fix);
* **warmup + cosine** learning-rate schedule stepped per optimizer step;
* **fused AdamW** on CUDA when available, with a silent-free fallback (logged);
* optional **torch.compile**; checkpoints always hold the uncompiled module's
  weights, so compiled and eager runs resume each other;
* **early stopping** on a validation score, best weights restored;
* **resumable checkpoints**: model, optimizer, scheduler, AMP scaler, epoch,
  step, best score/weights, patience counter, every RNG, and the config. A
  resume with a different config is refused. Batch order comes from an
  epoch-keyed seed, so an interrupted run resumes onto the same batches.
  SIGTERM / Ctrl-C save before exiting.

It does not own data loading. Callers pass ``batch_fn(indices) -> batch`` so
data can be pre-tensorised, memory-mapped, or tokenised on the fly. **The batch
must already be on ``trainer.device``** — the model is moved there at
construction, and moving batches automatically would hide the transfer cost of
a data pipeline that should be fixed instead.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import random
import signal
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from vpdl.models.base import derive_seed

logger = logging.getLogger(__name__)

__all__ = ["TrainConfig", "FitResult", "Trainer", "TRAIN_CHECKPOINT_FORMAT",
           "resolve_precision", "seed_everything"]

TRAIN_CHECKPOINT_FORMAT = "vpdl-dl-train-v1"

# Config fields that may change between an interrupted run and its resume.
_RESUME_MUTABLE = {"resume", "checkpoint_dir", "keep_last", "log_every"}


@dataclass
class TrainConfig:
    epochs: int = 50
    patience: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    grad_accum: int = 1
    max_grad_norm: float | None = 1.0
    warmup_frac: float = 0.0
    schedule: str = "cosine"          # cosine | constant
    precision: str = "auto"           # auto | bf16 | fp16 | fp32
    compile: bool = False
    fused_optimizer: bool = True
    checkpoint_dir: str | None = None
    resume: bool = False
    keep_last: int = 2
    log_every: int = 0                # optimizer steps; 0 = per epoch only
    # Save only parameters that train (LoRA / adapters / unfrozen layers) and
    # buffers. The frozen base is reproducible from the hub checkpoint, and a
    # 650M model's frozen weights would otherwise be 2.6 GB per saved epoch.
    checkpoint_trainable_only: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # A typo such as batch_size = 0 used to be clamped to 1 silently — a different run
        # from the one configured. Invalid values are refused instead.
        import numbers

        for name in ("epochs", "patience", "batch_size", "grad_accum", "keep_last"):
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f"TrainConfig.{name} must be a positive integer, got {value!r}")
        if not 0.0 <= float(self.warmup_frac) < 1.0:
            raise ValueError(f"TrainConfig.warmup_frac must be in [0, 1), got {self.warmup_frac!r}")
        if self.schedule not in ("cosine", "constant"):
            raise ValueError(f"TrainConfig.schedule must be cosine or constant, got {self.schedule!r}")

    def identity(self) -> str:
        """Hash of every field a resume must agree on."""
        payload = {k: v for k, v in asdict(self).items() if k not in _RESUME_MUTABLE}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str)
                              .encode()).hexdigest()[:16]


@dataclass
class FitResult:
    best_score: float
    best_epoch: int
    epochs_run: int
    steps: int
    resumed_from: str | None
    history: list[dict[str, float]]
    stopped_early: bool
    interrupted: bool = False


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_precision(requested: str, device) -> tuple[Any, bool]:
    """``(autocast_dtype or None, needs_grad_scaler)`` for `device`."""
    import torch

    if device.type != "cuda":
        if requested in ("bf16", "fp16"):
            logger.info("precision=%s requested on %s: running fp32 (autocast buys "
                        "nothing on this device).", requested, device.type)
        return None, False
    if requested == "fp32":
        return None, False
    bf16_ok = torch.cuda.is_bf16_supported()
    if requested == "bf16" and not bf16_ok:
        raise RuntimeError("bf16 requested but this CUDA device does not support it")
    if requested in ("auto", "bf16") and bf16_ok:
        return torch.bfloat16, False
    return torch.float16, True


def _rng_state() -> dict[str, Any]:
    import torch
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict[str, Any]) -> None:
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class Trainer:
    """Fit `model` with `loss_fn(model, batch)`; score with `score_fn(model)`.

    ``score_fn`` returns a validation score where HIGHER is better (ROC-AUC,
    or negative loss for pretraining). Without it there is no early stopping
    and the final weights are kept.
    """

    def __init__(self, model, config: TrainConfig, seed: int, device=None,
                 param_groups: list[dict] | None = None, run_name: str = "run") -> None:
        import torch

        self.torch = torch
        self.config = config
        self.seed = int(seed)
        self.run_name = run_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.module = model.to(self.device)
        self.model = self.module
        if config.compile:
            try:
                self.model = torch.compile(self.module)
            except Exception as error:                     # noqa: BLE001
                logger.warning("torch.compile unavailable (%s); running eager.", error)
        self.autocast_dtype, needs_scaler = resolve_precision(config.precision, self.device)
        self.scaler = torch.amp.GradScaler(self.device.type) if needs_scaler else None
        groups = param_groups or [{"params": [p for p in self.module.parameters()
                                              if p.requires_grad]}]
        for group in groups:
            group.setdefault("lr", config.lr)
        self.optimizer = self._optimizer(groups)
        self.scheduler = None
        self._stop_requested = False

    # -- construction helpers --------------------------------------------
    def _optimizer(self, groups):
        torch = self.torch
        kwargs = dict(lr=self.config.lr, weight_decay=self.config.weight_decay)
        if self.config.fused_optimizer and self.device.type == "cuda":
            try:
                return torch.optim.AdamW(groups, fused=True, **kwargs)
            except (RuntimeError, TypeError) as error:
                logger.info("fused AdamW unavailable (%s); using the default.", error)
        return torch.optim.AdamW(groups, **kwargs)

    def _schedule(self, total_steps: int):
        torch = self.torch
        total = max(1, int(total_steps))
        warmup = int(round(total * self.config.warmup_frac))
        cosine = self.config.schedule == "cosine"

        def factor(step: int) -> float:
            if warmup and step < warmup:
                return (step + 1) / warmup
            if not cosine:
                return 1.0
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, factor)

    def autocast(self):
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return self.torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)

    # -- checkpoints -------------------------------------------------------
    @property
    def _dir(self) -> Path | None:
        return Path(self.config.checkpoint_dir) if self.config.checkpoint_dir else None

    def _save(self, epoch: int, step: int, best_score: float, best_epoch: int,
              stalled: int, best_state, history, finished: bool) -> Path | None:
        if self._dir is None:
            return None
        torch = self.torch
        self._dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": TRAIN_CHECKPOINT_FORMAT, "config": asdict(self.config),
            "config_identity": self.config.identity(), "seed": self.seed,
            "run_name": self.run_name, "epoch": epoch, "step": step,
            "best_score": best_score, "best_epoch": best_epoch, "stalled": stalled,
            "finished": finished, "history": history,
            "model": self._state(), "best_state": self._state(best_state),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "rng": _rng_state(), "saved_at": time.time(),
        }
        path = self._dir / f"epoch-{epoch:04d}.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)                   # atomic: never a half-written file
        for old in sorted(self._dir.glob("epoch-*.pt"))[:-max(1, self.config.keep_last)]:
            old.unlink(missing_ok=True)
        return path

    def _state(self, state: dict | None = ...) -> dict | None:
        """The (possibly trainable-only) weights a checkpoint stores."""
        if state is None:
            return None
        state = self.module.state_dict() if state is ... else state
        if not self.config.checkpoint_trainable_only:
            return state
        keep = {name for name, p in self.module.named_parameters() if p.requires_grad}
        keep |= {name for name, _ in self.module.named_buffers()}
        return {k: v for k, v in state.items() if k in keep}

    def _load_weights(self, state: dict) -> None:
        if self.config.checkpoint_trainable_only:
            missing, unexpected = self.module.load_state_dict(state, strict=False)
            if unexpected:
                raise ValueError(f"checkpoint has parameters the model lacks: {unexpected[:5]}")
        else:
            self.module.load_state_dict(state)

    def latest_checkpoint(self) -> Path | None:
        if self._dir is None or not self._dir.exists():
            return None
        found = sorted(self._dir.glob("epoch-*.pt"))
        return found[-1] if found else None

    def _load(self, path: Path) -> dict[str, Any]:
        payload = self.torch.load(path, map_location="cpu", weights_only=False)
        recorded = payload.get("format")
        if recorded is not None and recorded != TRAIN_CHECKPOINT_FORMAT:
            raise ValueError(f"{path}: checkpoint format {recorded!r}, expected "
                             f"{TRAIN_CHECKPOINT_FORMAT!r}. Refusing a foreign format.")
        if payload.get("config_identity") != self.config.identity():
            raise ValueError(
                f"{path} was written by a different configuration "
                f"({payload.get('config_identity')} vs {self.config.identity()}). "
                "Resuming would silently mix two experiments; start a new run directory.")
        if payload.get("seed") != self.seed:
            raise ValueError(f"{path}: seed {payload.get('seed')} != {self.seed}")
        self._load_weights(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if self.scheduler is not None and payload.get("scheduler"):
            self.scheduler.load_state_dict(payload["scheduler"])
        if self.scaler is not None and payload.get("scaler"):
            self.scaler.load_state_dict(payload["scaler"])
        _set_rng_state(payload["rng"])
        return payload

    # -- the loop ----------------------------------------------------------
    def _install_signal_handlers(self):
        previous = {}

        def request_stop(signum, _frame):
            logger.warning("signal %s: saving a checkpoint at the next step boundary.", signum)
            self._stop_requested = True

        for name in ("SIGTERM", "SIGINT"):
            if hasattr(signal, name):
                try:
                    previous[name] = signal.signal(getattr(signal, name), request_stop)
                except ValueError:                     # not the main thread
                    pass
        return previous

    @staticmethod
    def _restore_signal_handlers(previous):
        for name, handler in previous.items():
            signal.signal(getattr(signal, name), handler)

    def fit(self, n_train: int, batch_fn: Callable[[np.ndarray], Any],
            loss_fn: Callable[[Any, Any], Any],
            score_fn: Callable[[Any], float] | None = None) -> FitResult:
        """Train for `config.epochs`. `batch_fn(indices)` returns a batch on
        ``self.device``; `loss_fn(model, batch)` returns a scalar loss."""
        torch = self.torch
        config = self.config
        micro = max(1, config.batch_size)
        batches_per_epoch = max(1, math.ceil(n_train / micro))
        steps_per_epoch = max(1, math.ceil(batches_per_epoch / config.grad_accum))
        self.scheduler = self._schedule(config.epochs * steps_per_epoch)

        start_epoch, step = 0, 0
        best_score, best_epoch, stalled = -math.inf, -1, 0
        best_state, history, resumed_from = None, [], None
        if config.resume and (latest := self.latest_checkpoint()) is not None:
            payload = self._load(latest)
            start_epoch, step = payload["epoch"] + 1, payload["step"]
            best_score, best_epoch = payload["best_score"], payload["best_epoch"]
            stalled, best_state = payload["stalled"], payload["best_state"]
            history, resumed_from = payload["history"], str(latest)
            logger.info("%s: resumed from %s (epoch %d, step %d)", self.run_name,
                        latest.name, payload["epoch"], step)
            if payload.get("finished"):
                logger.info("%s: checkpoint marks the run finished; nothing to do.",
                            self.run_name)
                if best_state is not None:
                    self._load_weights(best_state)
                return FitResult(best_score, best_epoch, len(history), step, resumed_from,
                                 history, stopped_early=False)

        previous = self._install_signal_handlers()
        interrupted = stopped = False
        epoch = start_epoch - 1
        try:
            for epoch in range(start_epoch, config.epochs):
                started = time.time()
                self.model.train()
                order = np.random.default_rng(
                    derive_seed(self.seed, f"{self.run_name}:epoch{epoch}")).permutation(n_train)
                running, seen = 0.0, 0
                self.optimizer.zero_grad(set_to_none=True)
                for index in range(batches_per_epoch):
                    batch = batch_fn(order[index * micro:(index + 1) * micro])
                    with self.autocast():
                        loss = loss_fn(self.model, batch)
                    window_start = (index // config.grad_accum) * config.grad_accum
                    window = min(config.grad_accum, batches_per_epoch - window_start)
                    scaled = loss.float() / window
                    (self.scaler.scale(scaled) if self.scaler else scaled).backward()
                    running += float(loss.detach().float().item())
                    seen += 1
                    last_in_window = (index + 1) % config.grad_accum == 0 or \
                        index + 1 == batches_per_epoch
                    if last_in_window:
                        if self.scaler:
                            self.scaler.unscale_(self.optimizer)
                        if config.max_grad_norm:
                            torch.nn.utils.clip_grad_norm_(
                                [p for g in self.optimizer.param_groups for p in g["params"]],
                                config.max_grad_norm)
                        if self.scaler:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.scheduler.step()
                        step += 1
                        if config.log_every and step % config.log_every == 0:
                            logger.info("%s step %d loss %.4f", self.run_name, step,
                                        running / seen)
                    if self._stop_requested:
                        break
                if self._stop_requested:
                    interrupted = True
                    # A partial epoch is not resumable exactly; save the last
                    # complete one instead of pretending this one finished.
                    logger.warning("%s: interrupted mid-epoch %d; the last complete "
                                   "epoch's checkpoint is the resume point.",
                                   self.run_name, epoch)
                    break

                score = float(score_fn(self.model)) if score_fn else -running / max(1, seen)
                history.append({"epoch": epoch, "train_loss": running / max(1, seen),
                                "score": score, "seconds": round(time.time() - started, 2)})
                if np.isfinite(score) and score > best_score:
                    best_score, best_epoch, stalled = score, epoch, 0
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self._state().items()}
                else:
                    stalled += 1
                logger.info("%s epoch %d/%d loss %.4f score %.4f (best %.4f @ %d) %.1fs",
                            self.run_name, epoch + 1, config.epochs, history[-1]["train_loss"],
                            score, best_score, best_epoch + 1, history[-1]["seconds"])
                done = score_fn is not None and stalled >= config.patience
                self._save(epoch, step, best_score, best_epoch, stalled, best_state,
                           history, finished=done or epoch + 1 == config.epochs)
                if done:
                    stopped = True
                    break
        except KeyboardInterrupt:
            interrupted = True
            logger.warning("%s: KeyboardInterrupt — the last complete epoch's checkpoint "
                           "is the resume point.", self.run_name)
            raise
        finally:
            self._restore_signal_handlers(previous)

        if best_state is not None and score_fn is not None:
            self._load_weights(best_state)
        return FitResult(best_score, best_epoch, len(history), step, resumed_from,
                         history, stopped_early=stopped, interrupted=interrupted)
