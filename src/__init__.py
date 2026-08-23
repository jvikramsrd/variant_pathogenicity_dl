"""variant_pathogenicity_dl: gene-specific variant pathogenicity pipeline.

Modules:
    data_loader   -- UniProt / ClinVar acquisition and cleaning.
    esm_extractor -- ESM-2 embeddings and zero-shot PLLR scores (cached).
    dataset       -- torch Dataset + position-grouped CV splitters.
    model         -- residual MLP classification head.
    loss          -- Focal loss and class-balanced weighted BCE.
    calibration   -- ECE / MCE / Brier, temperature scaling, isotonic regression.
    train         -- 5-fold CV training loop, benchmarking and VUS inference.

Heavy dependencies are imported lazily inside functions so that `import src`
remains cheap (e.g. for `--help`).
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = [
    "data_loader",
    "esm_extractor",
    "dataset",
    "model",
    "loss",
    "calibration",
    "train",
]
