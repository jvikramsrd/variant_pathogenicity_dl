# Local code audit — DL branch and broad-genomic SLM branch (master report)

Date: 2026-09-24. Phase: `LOCAL_CODE_AUDIT`. Machine: Windows 11 development PC, CPU only.
Baseline: commit `faf8de2`, branch `v2/rebuild`, clean tree. Data acquisition: **DEFERRED**
(nothing was downloaded; no clinical text or knowledge base was acquired). Nothing was
trained beyond tiny random-weight models in tests; no DGX performance is claimed.

Detailed reports sit beside this one in `docs/audit/`; the machine-readable summary is
`GENOMIC_SLM_AUDIT_STATUS.json`.

> The LLM/SLM branch was audited and hardened as a broad genomic model. MMR/Lynch Syndrome
> remains a downstream specialization and evaluation domain. The existing DL branch was
> audited and regression-protected but not redesigned.

## 1. Executive summary

- **One critical defect, fixed.** The SLM's conclusion masker (and the leakage audit's
  tripwire) missed verdicts written without a verb — "Pathogenic (PVS1, PM2_Supporting,
  PP3)", "Assertion: Likely Pathogenic", "… -> LP". On such a narrative the classification
  task's model input ended with the word it is asked to predict, and the audit would have
  reported it clean. No SLM result existed yet, so nothing published is affected.
- **Twenty high-severity findings: 17 fixed (one partly), 1 mitigated, 2 open.** Fixed: a
  decision threshold chosen on raw probabilities but applied to calibrated ones;
  non-reproducible resume in continued pretraining; ACMG gene specifications ignored at
  inference; leakage gates that were missing or silently passed (both branches); an
  evaluation join that always matched zero VUS; interface validation gaps; unrecorded
  backbone identity; OS-specific manifest paths; a test-suite collision that switched off
  10 DL regression tests whenever the full suite ran. Mitigated: the SLM export's empty
  explanation/ACMG/evidence fields are now flagged per record. Open: two teacher-pipeline
  gaps (feature work, EXP-017 marked not runnable).
- **The DL branch is sound and was not redesigned.** One latent checkpoint-size bug and a
  gate gap were fixed. One data-correctness issue needs a decision because fixing it
  changes results already produced on the DGX (gnomAD allele collapsing: 7 of 449 labelled
  rows in the DGX table).
- **Tests**: baseline 686 passed / 1 skipped / 10 failing (all the test collision) →
  after 767 passed / 1 skipped / 0 failing, including 71 new regression tests.
  **No regression detected in the tests executed.**

## 2. Repository architecture

| Area | Path | Status |
|---|---|---|
| DL branch | `vpdl/dl/**`, CLI `vpdl-dl`, `configs/dl/`, `docs/dl/` | live; runs on the DGX exist (`runs/dl/main`, `results/dl`) |
| Shared v2 core | `vpdl/{experiment,assemble,features,splits,evaluate,analysis,provenance,device}.py`, `vpdl/models/**`, `vpdl/sources/**` | live |
| Genomic SLM | `vpdl/slm/{build,text,modeling,evaluation}/**`, `vpdl/slm/{cli,interface,schema,labels,variants,teacher,retrieval,catalog,…}.py`, CLI `vpdl-slm`, `configs/slm/`, `docs/slm/` | built, not run on real data |
| From-scratch LM | `vpdl/slm/{pubmed,corpus,tokenizer,pack,model,train}.py`, via `vpdl slm-*` | trial run on the DGX |
| Knowledge base | `vpdl/kb/**`, `vpdl kb-*` | live |
| ProteinGym study | `vpdl/proteingym/**`, `vpdl pg-*` | live |
| v1 archive | `src/`, `scripts/`, `main.py`, `run_*.py`, `tests/test_*.py` | archive (tests still run) |

Import direction: `vpdl/dl` never imports `vpdl/slm`, `vpdl/kb` or `vpdl/proteingym`; no
`vpdl` module imports v1. The SLM uses DL public names (interface, trainer, calibration,
PEFT, leakage report types, tracking, hardware). Core ↔ DL imports are in functions only (no
import-time cycle). Duplications (two calibration modules, two leakage modules, three
splitters, several trainers) have different jobs and were judged justified; details in the
subagent reports summarised in section 23.

The only DL ↔ SLM interface is `vpdl/dl/interface.py` (DL export) read by
`vpdl/slm/interface.py` (`DLInput`), with `SLMRepresentation` subclassing the DL branch's
`ReasoningRepresentation`. **Final fusion is not implemented** (by instruction).

## 3. DL branch findings — DL_ARCHITECTURE_AUDIT.md, DL_DATA_PIPELINE_AUDIT.md

Fixed: PLM fine-tune checkpoints held the full frozen backbone (A-1, latent); `TrainConfig`
clamped invalid values silently (N-1); feature-store identity ignored the stored dtype
(O-4); the failure report's overfitting check was NaN under the `family` split (H-5); the
per-cell leakage gate skipped the label-proxy check (M-1). Added `vpdl-dl train --dry-run`.
Open: gnomAD allele collapsing (B-1, decision); weight decay on biases/LayerNorm (F-2,
scientific default).

## 4. SLM branch findings — SLM_ARCHITECTURE_AUDIT.md

Covers missense, nonsense, synonymous, frameshift, in-frame indels, start/stop loss, splice
site/region, intronic, UTRs, non-coding, CNV/structural; `m.`/`g.` fall to `other`;
regulatory/promoter is folded into upstream (documented). Five-class objective with
probabilities, uncertainty, evidence type/polarity heads, gene-masked ACMG head, grounded
deterministic explainer. The evidence heads run in parallel with the classifier and are not
consumed at inference — the flow is not a chained reasoning model (documented, not changed).
Fixed: ACMG mask at inference (C-2); export records now flag what they do not carry (C-1).

## 5. Clinical NLP findings — SLM_CLINICAL_NLP_AUDIT.md

The critical masker fix above, with a 17-case table of what is masked and what is kept.
Deferred: TSV robustness to embedded tabs/`\r` (real files needed).

## 6. VUS findings — SLM_VUS_AUDIT.md

VUS is an explicit class with its own probability, separate from abstention and uncertainty
(verified). Label mapping ran on all 108 `ClinicalSignificance` strings in the local release.
Fixed: the VUS-reclassification evaluation always joined zero rows (O-2); mutual information
on uncalibrated draws (H-2).

## 7. Training / PEFT findings — SLM_TRAINING_AUDIT.md

Fixed: exact resume of continued pretraining (F-1, reproduced then verified); non-BERT
encoder lookup (F-3); backbone identity in the record (J-1); `--lr 0` dropped (O-14). PEFT
freezing, LoRA init, gradient flow with checkpointing, and adapter-only checkpoints verified
by running. Open: weight-decay grouping (F-2); SLM configs inheriting DL defaults (O-6).

## 8. Distillation findings — SLM_DISTILLATION_AUDIT.md

No KL / soft-label distillation exists; no `vpdl-slm teacher` command and no exporter to the
table EXP-017 needs (EXP-017 is marked not runnable). Fixed: teacher units validated against
the ids and vocabulary given (G-3); missing confidence rejected (G-4); ignored
temperature/seed refused (G-5). Teacher output is never treated as ground truth (verified).

## 9. Calibration findings — CALIBRATION_AUDIT.md

No calibrator is fitted on test data in either branch. Fixed: threshold scale (HIGH);
calibrator guard fed a restated split (H-1); fitting on zero rows.

## 10. DL ↔ SLM interface findings — DL_SLM_INTERFACE_AUDIT.md

Fixed: per-class probability bounds; finiteness of `.npy`-referenced embeddings (both
readers); memory-mapped file handles (Windows); family-fold recognition in `fold_check`; the
fold report is now computed, logged and recorded in training; duplicate DL records refused;
NaN uncertainty. Open: no `quality_flags` in the frozen DL contract (use `metadata`).

## 11. Leakage findings — CURRENT_CODE_LEAKAGE_AUDIT.md

Fixed: conclusion leakage (critical); DL per-cell label-proxy check; SLM input gate inside
`finetune`; missing clusters under isolating schemes now critical; pretraining corpus
refuses to build without exclusions. The DGX's existing DL results are not affected by the
DL gate gap (its full report had run the check). Open: literature strictness default (M-6),
`dl_score` fold guard (M-7), retrieval exclusion by variant group (M-8).

## 12. Reproducibility findings — REPRODUCIBILITY_AUDIT.md

Every run record now carries the command line; SLM runs record the backbone version, split
digest and DL-input versions; manifests store `/` paths. Seeding, hash-based splits and
resume discipline verified.

## 13. Dependency findings — DEPENDENCY_AUDIT.md

All 126 modules import. `fair-esm` is declared but unused; `requirements*.txt` are v1-only
with lower floors. No package changed.

## 14. Windows compatibility findings — DEPENDENCY_AUDIT.md

Fixed: manifest path separators; memory-mapped handles blocking rewrites. Environment
note: Windows Application Control blocked pandas 3.0.6 / scikit-learn 1.9.1 DLLs mid-session;
the comparison was re-run in a second environment on both baseline and fixed code.

## 15. Bugs fixed

CRITICAL 1 · HIGH 17 (M-3 partly) · MEDIUM 10 (J-2 partly) · LOW 7, plus one HIGH mitigated
(C-1). Itemised with file, change, reason, risk and test in REGRESSION_AUDIT.md; the full
open/closed list is in `GENOMIC_SLM_AUDIT_STATUS.json`.

## 16. Files modified (25)

`vpdl/dl/{cli,failure,interface,leakage,tracking,trainer}.py`, `vpdl/dl/plm/finetune.py`,
`vpdl/provenance.py`, `vpdl/slm/{cli,experiments,interface,smoke,teacher}.py`,
`vpdl/slm/build/{inventory,leakage,records}.py`, `vpdl/slm/evaluation/calibration.py`,
`vpdl/slm/modeling/{continued,finetune,predict}.py`, `vpdl/slm/text/conclusion.py`,
`tests/dl/{conftest,test_data_qc,test_integration,test_splits_leakage}.py`.

## 17. Files created

`tests/dl/dl_helpers.py`, `tests/dl/test_dl_audit.py`, `tests/slm/test_genomic_audit.py`, and
`docs/audit/`: this report, `GENOMIC_SLM_AUDIT_STATUS.json`, `DL_ARCHITECTURE_AUDIT.md`,
`DL_DATA_PIPELINE_AUDIT.md`, `SLM_ARCHITECTURE_AUDIT.md`, `SLM_CLINICAL_NLP_AUDIT.md`,
`SLM_VUS_AUDIT.md`, `SLM_TRAINING_AUDIT.md`, `SLM_DISTILLATION_AUDIT.md`,
`CALIBRATION_AUDIT.md`, `DL_SLM_INTERFACE_AUDIT.md`, `REPRODUCIBILITY_AUDIT.md`,
`REGRESSION_AUDIT.md`, `ROBUSTNESS_AUDIT.md`, `DEPENDENCY_AUDIT.md`,
`CURRENT_CODE_LEAKAGE_AUDIT.md`.

## 18. Tests run

Full suite four times (baseline ×2 in two environments, fixed tree, final re-run — see
REGRESSION_AUDIT.md); `tests/dl` alone at baseline; new test files alone and against the
unfixed baseline; import of every module; `vpdl-slm smoke`; `vpdl-dl train --dry-run` on the
real DGX table and on synthetic tables; `vpdl-dl hardware --smoke`; `vpdl-slm hardware`;
subagent spot checks (masker table, label mapping on the real ClinVar strings, resume
equivalence, LoRA gradient flow, calibration maths, interface round trips).

## 19. Tests passed

After fixes: 767 of 768 (one CUDA-only skip). New: 71 of 71. Smoke: `ok: true` (19.9 s).
DL dry run on the real table: exit 0, 0 critical leakage findings, 6 models constructed.

## 20. Tests failed

After fixes: none. Before fixes: 1 failure + 9 errors, all the `conftest` collision.

## 21. Tests not run

The CUDA-only landmine test (`test_landmines.py:1221`); any test needing the DGX; no static
type checker or linter (ruff/mypy are not installed here).

## 22. Experiments not run

Every experiment in the SLM matrix (EXP-001 … EXP-025); any DL training cell; continued
pretraining beyond 2–4 tiny steps; the teacher; baselines on real data; any download.

## 23. Remaining risks

1. Real ClinVar narratives will contain templates the masker has not seen: run
   `vpdl-slm leakage` and read the tripwire rate before training.
2. gnomAD allele collapsing in the DL features (B-1): decide, rebuild, re-run.
3. EXP-017 (teacher) cannot run until the command and exporter exist.
4. The SLM export does not carry explanations/ACMG/evidence (flagged per record).
5. The SLM's full 15-check audit is still a separate command; only the input checks run
   inside `finetune`.
6. SLM configs inherit DL trainer/PEFT defaults they do not state.
7. No format-checked loader for the SLM's packaged `model.pt`.

## 24. DGX-specific items deferred

`pip install -e ".[genomic-slm,dl,gbm]"` and `pytest tests` on the DGX (aarch64 CUDA torch,
Triton headers); CUDA determinism (TF32, cuDNN autotune, bf16 reductions); `torch.compile`
on GB10; OOM back-off of the automatic batch size; real memory for 650M/3B backbones with
trainable-only checkpoints; registry locking with `--jobs N`; the first real DL export and
the SLM fold report on it; ClinVar TSV parsing on the real `submission_summary.txt.gz`.

## 25. Recommended DGX audit

1. Pull this branch; confirm `git status` is clean and the commit matches.
2. `pytest tests -q` on the DGX — expect 767 passed and the CUDA test now running.
3. `vpdl-dl hardware --smoke` and `vpdl-slm hardware --smoke`; record in `docs/v2/HARDWARE.md`.
4. `vpdl-dl train --dry-run` with the `runs/dl/main` configuration on the DGX table.
5. `vpdl-slm smoke` on the GPU.
6. Decide B-1 (gnomAD allele collapsing); if fixed, rebuild the canonical table and re-run
   the DL cells with three seeds.
7. Only then start the data phase (ClinVar `submission_summary`, `var_citations`, archived
   release), followed by `vpdl-slm leakage` on real examples before any training.
