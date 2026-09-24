# Dependency, environment and Windows-compatibility audit — local code audit, 2026-09-24

Declarations: `pyproject.toml` (core numpy/pandas/scikit-learn/scipy; extras `gbm`, `slm`,
`genomic-slm`, `plm`, `dl`, `dev`; torch deliberately undeclared), plus the v1-era
`requirements.txt` / `requirements-cuda.txt`. **No package was upgraded or added to the
project.** Tests ran in throw-away `uv` environments.

## Environments used

| Name | Contents | Used for |
|---|---|---|
| A | Python 3.14.7, torch 2.14.0+cpu, transformers 5.17.0, tokenizers 0.23.2, pyarrow 25.0.1, numpy 2.5.3, **pandas 3.0.6, scikit-learn 1.9.1**, scipy 1.18.1, xgboost, lightgbm, matplotlib, tabulate, requests, seaborn, biopython, tqdm | the original baseline run |
| B | identical except **pandas 2.3.3, scikit-learn 1.8.0** | everything after the session restart |

Why B: after the session restarted, Windows Application Control began refusing to load
pandas 3.0.6's and scikit-learn 1.9.1's compiled modules ("An Application Control policy has
blocked this file") — the same wheels that had loaded an hour earlier, from both cache
locations tried (the default uv cache and one outside `AppData`). numpy, torch, scipy,
pyarrow and tokenizers were unaffected. Security settings
were not touched. To keep the regression comparison fair, the unmodified baseline commit
was re-run in environment B as well (REGRESSION_AUDIT.md).

## Import check

All 126 non-`__main__` modules under `vpdl/` import cleanly (environment A, subagent L).

## Findings

| ID | Severity | Finding | Status |
|---|---|---|---|
| L-1 | HIGH | OS-dependent path separators in manifests (see REPRODUCIBILITY_AUDIT.md). | **FIXED** |
| L-2 | MEDIUM | `fair-esm` (extra `plm`) is never imported; ESM loads through `transformers`. | OPEN — remove or annotate |
| L-3 | MEDIUM | `requirements*.txt` pin floors below pyproject's (pandas ≥2.0 vs ≥2.1, numpy ≥1.24 vs ≥1.26, scikit-learn ≥1.3 vs ≥1.4), lack the v2 extras, and list unused `torchvision`. They serve only the v1 archive tests (`requests`, `seaborn`, `biopython` are v1-only). | OPEN — mark them v1-only |
| L-4 | LOW | `build-records` needs pyarrow (`genomic-slm`) but is reachable from the `slm` extra; failure is a bare `ModuleNotFoundError`. | OPEN |
| — | INFO | No upper bounds, and pandas 3 / transformers 5 are already in use; the code handles the transformers `torch_dtype` → `dtype` rename. | Documented |

## Windows compatibility (portability kept; nothing made Windows-only)

Fine: `os.replace` for atomic writes; `spawn`-safe pools (module-level functions); explicit
`encoding="utf-8"` on text writes; signal handlers guarded; no hard-coded `/home`, `/mnt`,
`/tmp`; subprocess calls use argument lists; CUDA calls guarded.

Fixed here: path separators in manifests (L-1); memory-mapped `.npy` handles that blocked
rewriting exports on Windows (I-5).

Open: registry appends are locked with `fcntl` on POSIX only; `--jobs N` on Windows would
append unlocked (N-2/O-13; training never runs on Windows).

## Linux / DGX-specific by design (for the DGX audit)

`fcntl` locking; `python3.12-dev` headers for Triton/`torch.compile`; `nvidia-smi`; NVIDIA's
aarch64 CUDA torch build (PyPI's aarch64 wheel is CPU-only); unified-memory detection; TF32 /
cuDNN autotune; Ollama as a system service.
