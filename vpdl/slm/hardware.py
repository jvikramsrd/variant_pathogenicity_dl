"""What this machine is, and what the intended training machine is — never confused.

``vpdl.dl.hardware`` already detects the stack (torch, CUDA, bf16, SDPA,
Triton, fused AdamW, memory) and recommends settings; this adds the SLM
branch's needs — a tokenizers/transformers check, whether an encoder and a
decoder can be built and run — and, above all, states plainly whether the
DGX Spark is the machine in front of us. Every report carries both:

    detected  what was measured here, now
    intended  NVIDIA DGX Spark (GB10, aarch64, unified memory) — the training target

A run that is not on the intended target is not a failure; it is a fact the
report states, so no number is ever attributed to hardware it did not run on.
"""

from __future__ import annotations

from typing import Any

__all__ = ["INTENDED_TARGET", "hardware_report", "is_intended_target"]

INTENDED_TARGET = {
    "name": "NVIDIA DGX Spark (GB10 Grace-Blackwell)", "machine": "aarch64",
    "memory": "unified CPU/GPU", "precision": "bf16",
    "measured_here_before": "90.1 TFLOPS bf16 peak matmul; 30.7 TFLOPS sustained while training the "
                            "from-scratch small model with torch.compile (docs/RUNLOG.md, 2026-09-22)",
}


def is_intended_target(report: dict[str, Any]) -> bool:
    device = report.get("device", {})
    name = str(device.get("gpu_name") or device.get("name") or "")
    return device.get("machine") == "aarch64" and ("GB10" in name or "Spark" in name)


def _slm_checks(smoke: bool) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    try:
        import tokenizers
        checks["tokenizers"] = tokenizers.__version__
    except ImportError:
        checks["tokenizers"] = None
    try:
        import pyarrow
        checks["pyarrow"] = pyarrow.__version__
    except ImportError:
        checks["pyarrow"] = None
    try:
        import sklearn
        checks["scikit_learn"] = sklearn.__version__
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
        checks["sklearn_tfidf"] = True
    except Exception as error:                                        # noqa: BLE001
        checks["scikit_learn"] = checks.get("scikit_learn")
        checks["sklearn_tfidf"] = f"no ({type(error).__name__}: {str(error)[:80]})"
    if smoke:
        try:
            import torch

            from vpdl.slm.modeling.backbone import BackboneSpec, load_backbone
            from vpdl.slm.modeling.multitask import GenomicSLM, HeadConfig
            backbone = load_backbone(BackboneSpec("tiny-bert"))
            model = GenomicSLM(backbone, HeadConfig())
            tokens = backbone.tokenize(["the variant was absent from gnomAD"], 32)
            output = model(input_ids=torch.tensor(tokens["input_ids"]),
                           attention_mask=torch.tensor(tokens["attention_mask"]))
            checks["smoke_forward"] = list(output["class_logits"].shape) == [1, 5]
        except Exception as error:                                    # noqa: BLE001
            checks["smoke_forward"] = f"no ({type(error).__name__}: {str(error)[:120]})"
    return checks


def hardware_report(smoke: bool = False) -> dict[str, Any]:
    from vpdl.dl.hardware import hardware_report as dl_report
    report = dl_report(smoke=smoke)
    report["checks"].update(_slm_checks(smoke))
    report["intended_target"] = INTENDED_TARGET
    report["running_on_intended_target"] = is_intended_target(report)
    if not report["running_on_intended_target"]:
        report.setdefault("recommendations", []).append(
            "NOT the intended target: this is a development machine. Pretraining, fine-tuning, "
            "ablations and any timing figure belong on the DGX Spark "
            "(docs/slm/GENOMIC_SLM_DGX_RUNBOOK.md).")
    return report
