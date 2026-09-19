"""Hardware detection that reports what it found instead of assuming.

Targets are a discrete x86-64 CUDA card and a DGX Spark (GB10 Grace-Blackwell,
aarch64 Ubuntu, unified CPU/GPU memory). Those two want opposite batching
strategies, so nothing here hardcodes one: it measures, logs, and recommends.

The v1 grid ran `--batch_size 1 --grad_accum 8 --gradient_checkpointing` because
it was squeezed onto a ~15 GiB card, and those flags then travelled to every
machine the code touched, including ones with eight times the memory. Settings
that exist to survive one box should not become defaults everywhere.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["DeviceInfo", "detect", "log_summary"]

_GIB = 1 << 30


@dataclass(frozen=True)
class DeviceInfo:
    kind: str                      # "cuda" | "mps" | "cpu"
    name: str
    machine: str                   # platform.machine(), e.g. aarch64 / x86_64
    total_memory_gib: float | None
    compute_capability: tuple[int, int] | None
    supports_bf16: bool
    unified_memory: bool
    torch_version: str | None

    @property
    def recommends_micro_batching(self) -> bool:
        """True only when memory is genuinely tight.

        Unified-memory parts get the large pool, so the checkpointing and
        accumulation contortions are unnecessary there.
        """
        if self.total_memory_gib is None:
            return self.kind != "cuda"
        return self.total_memory_gib < 24.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "recommends_micro_batching": self.recommends_micro_batching
        }


def detect() -> DeviceInfo:
    """Inspect the machine. Never raises: an absent torch yields a CPU record."""
    machine = platform.machine()

    try:
        import torch
    except ImportError:
        return DeviceInfo(
            kind="cpu", name="cpu (torch not installed)", machine=machine,
            total_memory_gib=None, compute_capability=None,
            supports_bf16=False, unified_memory=False, torch_version=None,
        )

    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        capability = torch.cuda.get_device_capability(index)

        # Heuristic, and labelled as one: an aarch64 host with an integrated
        # CUDA device is a Grace-Blackwell or Jetson-class part sharing one
        # LPDDR pool with the CPU. Confirmed properties belong in
        # docs/v2/HARDWARE.md, recorded from a real run on the box.
        unified = machine == "aarch64" and getattr(props, "is_integrated", 0) == 1

        return DeviceInfo(
            kind="cuda",
            name=props.name,
            machine=machine,
            total_memory_gib=round(props.total_memory / _GIB, 1),
            compute_capability=capability,
            supports_bf16=bool(torch.cuda.is_bf16_supported()),
            unified_memory=unified,
            torch_version=torch.__version__,
        )

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return DeviceInfo(
            kind="mps", name="Apple MPS", machine=machine,
            total_memory_gib=None, compute_capability=None,
            supports_bf16=False, unified_memory=True,
            torch_version=torch.__version__,
        )

    return DeviceInfo(
        kind="cpu", name=platform.processor() or "cpu", machine=machine,
        total_memory_gib=None, compute_capability=None, supports_bf16=False,
        unified_memory=False, torch_version=torch.__version__,
    )


def autocast_dtype(info: DeviceInfo | None = None):
    """Preferred autocast dtype, or None where autocast buys nothing.

    bf16 over fp16 wherever supported: it keeps fp32's exponent range, so the
    loss-scaling failure modes of fp16 finetuning do not arise.
    """
    info = info or detect()
    if info.kind != "cuda":
        return None
    import torch
    return torch.bfloat16 if info.supports_bf16 else torch.float16


def log_summary(info: DeviceInfo | None = None) -> DeviceInfo:
    """Emit one line naming what was found. Call once at every entry point.

    A wrong hardware assumption should surface here, in the first second of a
    run, rather than as an OOM three hours into a grid.
    """
    info = info or detect()
    memory = f"{info.total_memory_gib} GiB" if info.total_memory_gib else "unknown"
    logger.info(
        "device=%s (%s) arch=%s memory=%s bf16=%s unified=%s torch=%s "
        "micro_batching_recommended=%s",
        info.kind, info.name, info.machine, memory, info.supports_bf16,
        info.unified_memory, info.torch_version, info.recommends_micro_batching,
    )
    return info
