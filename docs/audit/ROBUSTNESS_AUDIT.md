# Robustness, error-handling and security audit — local code audit, 2026-09-24

Scope: `vpdl/` (not the v1 archive). 81 `except` sites in 30 files reviewed; patterns
searched: `except`, `suppress`, `errors="coerce"`, `dropna`, `fillna`, `filterwarnings`,
`os.replace`/`tempfile`, `rmtree`/`unlink`, `torch.load`/`pickle`/`allow_pickle`/`trust_remote_code`,
`subprocess`, `extractall`.

**No CRITICAL issue.** No bare `except: pass`, no `shell=True`, no `extractall`, no
`trust_remote_code`. The Ollama client's loopback check rejects `127.0.0.1@evil`,
`localhost.evil.com` and non-canonical loopback spellings.

## Findings

| ID | Severity | Where | Finding | Status |
|---|---|---|---|---|
| N-1 | MEDIUM | `vpdl/dl/trainer.py` | Invalid batch size / epochs silently clamped. | **FIXED** (refused) |
| MASTER-4 | LOW | `vpdl/slm/smoke.py` | If the fine-tune dry run raised, the module-level registry path stayed pointed at a deleted temp dir. | **FIXED** |
| — | LOW | `vpdl/slm/evaluation/calibration.py` | Calibration on zero labelled rows returned T = 20 silently. | **FIXED** |
| — | LOW | `vpdl/slm/interface.py` | NaN uncertainty written as invalid JSON. | **FIXED** |
| N-2 | LOW | `vpdl/dl/tracking.py` | Registry appends unlocked on Windows. | OPEN (training is DGX-only) |
| N-3 | LOW | `vpdl/dl/report.py:59-67` | A malformed CI string becomes (nan, nan) without a count. | OPEN |
| N-4 | INFO | `vpdl/experiment.py` | Per-cell result files are written directly (not temp+rename); mitigated by archiving the previous run first and writing the summary last. | Documented |
| — | INFO | several | `torch.load(weights_only=False)` and `np.load(allow_pickle=True)` read only files this project wrote, behind format-tag checks. | Acceptable for a single-machine pipeline |

## Broad excepts judged acceptable

Per-worker failure in `vpdl-dl train --jobs` (counted, exit 1); `torch.compile` and fused
AdamW fallbacks (logged); signal registration off the main thread; GPU probe in
`proteingym/combined.py` (raised or logged); hardware probes (documented best-effort);
unreadable checkpoint during resume search in `slm/train.py` (logged, previous tried);
`vpdl/cli.py` source capability probe (printed); one malformed GeneReviews chapter
(counted); backend selection in `models/gbm.py`, `models/mlp.py` (logged); cp1252 fallback for
NCBI's mixed-encoding file; malformed ClinVar/ERepo fields (counted in build stats).

## Checkpoints and data

No step of this audit wrote to, overwrote or deleted any checkpoint, dataset or result file.
Checkpoint rotation (`Trainer`, `slm/train.py`, `continued.py`) deletes only older files of its
own run directory; `vpdl-dl train` archives a cell's previous results before a re-run.
