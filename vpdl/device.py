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
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["DeviceInfo", "detect", "log_summary"]

_GIB = 1 << 30


@dataclass(frozen=True)
class DeviceInfo:
    kind: str                      # what torch can USE: "cuda" | "mps" | "cpu"
    name: str
    machine: str                   # platform.machine(), e.g. aarch64 / x86_64
    total_memory_gib: float | None
    compute_capability: tuple[int, int] | None
    supports_bf16: bool
    unified_memory: bool
    torch_version: str | None
    # What the driver can SEE, independent of torch. On a machine without
    # torch these two answers differ, and reporting only the first told a DGX
    # Spark owner "no accelerator detected" while its GB10 sat idle.
    gpu_name: str | None = None

    @property
    def gpu_present(self) -> bool:
        return self.gpu_name is not None

    @property
    def recommends_micro_batching(self) -> bool | None:
        """True only when memory is genuinely tight; None when it is unknown.

        Unified-memory parts get the large pool, so the checkpointing and
        accumulation contortions are unnecessary there. Without a memory figure
        there is nothing to recommend from, and saying so beats a guess.
        """
        if self.total_memory_gib is None:
            return None
        return self.total_memory_gib < 24.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "gpu_present": self.gpu_present,
            "recommends_micro_batching": self.recommends_micro_batching,
        }


def _probe_nvidia_smi() -> tuple[str, float | None] | None:
    """Ask the driver directly, for machines where torch cannot answer yet."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        lines = subprocess.run(
            [executable, "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip().splitlines()
    except (subprocess.SubprocessError, OSError):
        return None
    if not lines:
        return None
    name, _, memory = lines[0].partition(",")
    try:
        memory_gib = round(float(memory.strip()) / 1024, 1)
    except ValueError:
        memory_gib = None          # unified-memory parts may report [N/A]
    return name.strip(), memory_gib


def detect() -> DeviceInfo:
    """Inspect the machine. Never raises: an absent torch yields a CPU record."""
    machine = platform.machine()

    try:
        import torch
    except ImportError:
        # Memory is deliberately NOT taken from the probe: this record's `kind`
        # is cpu, and a GPU memory figure beside it reads as the CPU's.
        probe = _probe_nvidia_smi()
        return DeviceInfo(
            kind="cpu", name="cpu (torch not installed)", machine=machine,
            total_memory_gib=None, compute_capability=None,
            supports_bf16=False, unified_memory=False, torch_version=None,
            gpu_name=probe[0] if probe else None,
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
            gpu_name=props.name,
        )

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return DeviceInfo(
            kind="mps", name="Apple MPS", machine=machine,
            total_memory_gib=None, compute_capability=None,
            supports_bf16=False, unified_memory=True,
            torch_version=torch.__version__,
        )

    # torch is installed but cannot reach a GPU the driver can see. On aarch64
    # the default PyPI torch wheel is CPU-only, so this is the expected failure
    # after a plain `pip install torch` on a DGX Spark — worth naming, not
    # reporting as "no accelerator".
    probe = _probe_nvidia_smi()
    return DeviceInfo(
        kind="cpu", name=platform.processor() or "cpu", machine=machine,
        total_memory_gib=None, compute_capability=None, supports_bf16=False,
        unified_memory=False, torch_version=torch.__version__,
        gpu_name=probe[0] if probe else None,
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
        "device=%s (%s) gpu_seen_by_driver=%s arch=%s memory=%s bf16=%s "
        "unified=%s torch=%s micro_batching_recommended=%s",
        info.kind, info.name, info.gpu_name or "none", info.machine, memory,
        info.supports_bf16, info.unified_memory, info.torch_version,
        "unknown" if info.recommends_micro_batching is None
        else info.recommends_micro_batching,
    )
    return info
