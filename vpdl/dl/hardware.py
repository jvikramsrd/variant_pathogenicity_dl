"""What the machine can do for the DL branch — detected, never assumed.

Extends :mod:`vpdl.device` (which detects) with the capabilities the DL code
switches on: bf16, SDPA/FlashAttention back ends, torch.compile (needs Triton
and the Python headers — the DGX needed ``python3.12-dev``), fused AdamW, and
the transformers ESM attention implementation. ``smoke=True`` runs a tiny bf16
matmul and a tiny random-weight ESM forward on the device: it proves the stack
works; it is not a benchmark and reports no throughput.
"""

from __future__ import annotations

import os
import platform
import shutil
import sysconfig
from pathlib import Path
from typing import Any

__all__ = ["hardware_report", "recommendations"]


def _torch_checks(smoke: bool) -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"torch": None}
    checks: dict[str, Any] = {"torch": torch.__version__, "cuda_build": torch.version.cuda,
                              "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        checks |= {"device_name": torch.cuda.get_device_name(0),
                   "capability": list(torch.cuda.get_device_capability(0)),
                   "memory_free_gib": round(free / 2 ** 30, 1),
                   "memory_total_gib": round(total / 2 ** 30, 1),
                   "bf16": torch.cuda.is_bf16_supported(),
                   "sdpa_flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
                   "sdpa_mem_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
                   "tf32_matmul": torch.backends.cuda.matmul.allow_tf32}
        try:
            parameter = torch.nn.Parameter(torch.zeros(4, device="cuda"))
            torch.optim.AdamW([parameter], fused=True)
            checks["fused_adamw"] = True
        except (RuntimeError, TypeError) as error:
            checks["fused_adamw"] = f"no ({error})"
    try:
        import triton  # noqa: F401
        checks["triton"] = True
    except ImportError:
        checks["triton"] = False
    header = Path(sysconfig.get_paths()["include"]) / "Python.h"
    checks["python_headers"] = header.exists()
    checks["torch_compile_ready"] = bool(checks["triton"] and checks["python_headers"])
    try:
        import transformers
        from transformers import EsmConfig, EsmForMaskedLM
        checks["transformers"] = transformers.__version__
        config = EsmConfig(vocab_size=33, hidden_size=16, num_hidden_layers=1,
                           num_attention_heads=2, intermediate_size=32,
                           max_position_embeddings=64, pad_token_id=1, mask_token_id=32,
                           position_embedding_type="rotary")
        try:
            EsmForMaskedLM._from_config(config, attn_implementation="sdpa")
            checks["esm_sdpa"] = True
        except (ValueError, ImportError, TypeError) as error:
            checks["esm_sdpa"] = f"no ({error})"
    except ImportError:
        checks["transformers"] = None
    if smoke and torch.cuda.is_available():
        a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        checks["smoke_bf16_matmul"] = bool(torch.isfinite((a @ a).float()).all())
        if checks.get("transformers"):
            from vpdl.dl.plm.backbones import tiny_backbone
            backbone = tiny_backbone()
            backbone.model.to("cuda")
            from vpdl.dl.plm.forward import hidden_states
            out = hidden_states(backbone, ["MKTAYIAKQR"])
            checks["smoke_esm_forward"] = bool(out[0].shape == (10, 32))
    return checks


def recommendations(report: dict[str, Any]) -> list[str]:
    device, checks = report["device"], report["checks"]
    notes = []
    if device["machine"] == "aarch64" and checks.get("torch") and not checks.get("cuda_available"):
        notes.append("torch is installed but cannot reach the GPU: the PyPI aarch64 wheel is "
                     "CPU-only. Install NVIDIA's CUDA build or use the NGC PyTorch container.")
    if checks.get("cuda_available") and not checks.get("bf16"):
        notes.append("no bf16: training falls back to fp16 + GradScaler.")
    if checks.get("cuda_available") and not checks.get("torch_compile_ready"):
        notes.append("torch.compile unavailable (needs Triton and Python.h; on DGX OS: "
                     "sudo apt install python3.12-dev). Runs stay eager, correctly.")
    if device.get("unified_memory"):
        notes.append("unified memory: batch sizes are configuration, not survival settings — "
                     "no batch-1 + accumulation + checkpointing needed for 650M probes; keep "
                     "gradient checkpointing for 650M fine-tuning of long MSH6 windows.")
    if checks.get("memory_total_gib") and checks["memory_total_gib"] < 24:
        notes.append("< 24 GiB accelerator memory: use micro-batching and gradient "
                     "checkpointing for ESM-2 650M fine-tuning.")
    return notes


def hardware_report(smoke: bool = False) -> dict[str, Any]:
    from vpdl.device import detect

    usage = shutil.disk_usage(".")
    report = {"device": detect().as_dict(),
              "host": {"machine": platform.machine(), "system": platform.system(),
                       "python": platform.python_version(), "cpus": os.cpu_count(),
                       "disk_free_gib": round(usage.free / 2 ** 30, 1)},
              "checks": _torch_checks(smoke)}
    report["recommendations"] = recommendations(report)
    return report
