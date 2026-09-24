# SLM distillation / teacher audit — local code audit, 2026-09-24

Scope: `vpdl/slm/teacher.py`, `vpdl/kb/ollama.py` (transport), `vpdl/slm/build/leakage.py`
(`_check_teacher`), `configs/slm/exp017_teacher.toml`, `vpdl/slm/experiments.py`.
No teacher was contacted; no teacher output exists.

## What exists

Prompt building for TRAINING examples only (`prompts_for`), response parsing, a grounding
check on the summary, a confidence filter, provenance per record (model, version/digest,
prompt hash, time, split at generation), and a licence gate (MedGemma needs an explicit
`accept_terms`; qwen3:32b is Apache-2.0). The leakage audit re-derives each teacher
record's split from the examples table rather than trusting the record.

## NOT IMPLEMENTED

- Soft-label / KL distillation: there is no temperature-scaled KL loss. "Distillation" here
  means training the evidence heads on filtered teacher labels (hard labels).
- A `vpdl-slm teacher` command, and anything that writes `teacher_units.parquet` (the file
  `exp017_teacher.toml` points at). EXP-017 is therefore not runnable; its entry in the
  experiment matrix now says so.

## Findings

| ID | Severity | Where | Finding | Status |
|---|---|---|---|---|
| G-1 | HIGH | `vpdl/slm/cli.py`, `experiments.py:127` | EXP-017's command names a `vpdl-slm teacher` subcommand that does not exist. | OPEN (feature) — matrix entry annotated "NOT RUNNABLE YET" |
| G-2 | HIGH | `configs/slm/exp017_teacher.toml:10` | No exporter from teacher records to the unit-examples table training reads. | OPEN (feature). Fails closed: `config_problems` reports the missing file. |
| G-3 | HIGH | `teacher.py` `run_teacher` | Only the free-text summary was grounding-checked; the `units` list (what would train the heads) was never checked — a unit id never given, or an invented evidence type/polarity, passed. | **FIXED** — `unit_problems` rejects unknown ids and labels outside the evidence vocabulary |
| G-4 | MEDIUM | `teacher.py:162` | An answer with no `confidence` skipped the confidence gate and was kept (confirmed by running). | **FIXED** — no stated confidence cannot pass; non-numeric or out-of-range confidence is treated as none |
| G-5 | MEDIUM | `teacher.py:65-66`, `kb/ollama.py:119-127` | `TeacherConfig.temperature` / `seed` were never sent (the client always uses 0/0) but were stored as provenance. | **FIXED** — non-default values are refused with the local client, so the record cannot misstate what was sent |

## Verified correct

Only training examples are prompted; the prompt never contains the label; teacher output is
never merged into a gold table or used as an evaluation truth (repo-wide search); the
licence gate works.

## Data-phase risk

The teacher's own pretraining data may contain ClinVar/PubMed text about evaluation
variants (benchmark contamination this project cannot control). Record it alongside any
EXP-017 result.
