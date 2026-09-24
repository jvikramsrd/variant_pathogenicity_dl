# Reproducibility audit — local code audit, 2026-09-24

Scope: seeding, hidden nondeterminism, run records and checkpoints in both branches
(`vpdl/provenance.py`, `vpdl/dl/tracking.py`, `vpdl/dl/trainer.py`, `vpdl/experiment.py`,
`vpdl/models/**`, `vpdl/slm/{experiments,train,config}.py`, `vpdl/slm/modeling/**`,
`vpdl/slm/build/splits.py`), checked against the real `runs/dl/registry.jsonl` and
`runs/dl/main/summary_*.json` written on the DGX.

## What a run records

| Field | DL | SLM fine-tune |
|---|---|---|
| git commit + dirty paths | registry + summary | registry; now also inside `model.pt` |
| dataset version | table sha256 | examples sha256 |
| config | registry (`CellConfig`, incl. `model_kwargs`) | registry (`FinetuneConfig`) |
| seed | yes | yes |
| model / tokenizer version | PLM backbone id, revision, commit | **was dropped** → now `backbone_version` (weights sha, commit, revision, vocab) |
| split | `split_hash` in summary | `split_ids` digest → now in the registry and `model.pt` |
| command line | **not recorded** | **not recorded** → now `command` in every registry record (both branches) |
| hardware / software versions | `vpdl.device` + library versions | same |
| metrics | registry + per-cell files | registry + `metrics.json` |

## Findings

| ID | Severity | Where | Finding | Status |
|---|---|---|---|---|
| F-1 | HIGH | `vpdl/slm/modeling/continued.py` | Resume was not exact (dropout RNG). | **FIXED** (see SLM_TRAINING_AUDIT.md) |
| J-1 | HIGH | `vpdl/slm/modeling/finetune.py` | Backbone identity not recorded. | **FIXED** |
| — | MEDIUM | `vpdl/dl/tracking.py` | No command line in any run record. | **FIXED** — `command` (not part of `experiment_id`, so ids are unchanged) |
| L-1 | HIGH (portability) | `vpdl/provenance.py:180`, `vpdl/slm/build/{records,inventory}.py`, `vpdl/dl/tracking.py` | Paths were stored with the OS separator; a manifest written on Windows could not be verified on Linux (`data\raw\x` is one file name there). | **FIXED** — stored with `/` (no change on Linux; the committed manifests contained no backslashes) |
| J-2 | MEDIUM | `finetune.py` | Packaged `model.pt` had no provenance and no format-checked loader. | **PARTLY FIXED** — provenance block (git, examples sha, split digest, backbone version, DL inputs) added; loader OPEN |
| J-3 | LOW | `vpdl/experiment.py` | `summary_*.json` omits `model_kwargs` (the registry has them). | OPEN |
| J-5 | INFO | `vpdl/experiment.py` | Bootstrap CIs reuse the cell seed for every gene. | Documented |
| — | INFO | DGX | `prepare_device` enables TF32 and cuDNN autotuning; no `use_deterministic_algorithms`. Same seed ≠ bit-identical on CUDA; the project handles noise with three seeds. | Documented for the DGX audit |

## Verified correct

`derive_seed` uses SHA-256 (no salted `hash()` anywhere); every model seeds before
construction, per fold, keyed on the gene (L7); no `DataLoader` workers (batch order from a
seeded generator per epoch, so resume replays batches); `Trainer` refuses a resume with a
different config identity or seed; SLM splits use salted SHA-1 of ids (identical on every
machine); `assert_comparable` separates dataset/split keys from replicate keys; the dirty-tree
flag works (seen in the real registry).
