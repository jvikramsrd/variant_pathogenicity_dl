# SLM training / PEFT / multi-task audit — local code audit, 2026-09-24

Scope: `vpdl/slm/modeling/{finetune,continued,peft,backbone,losses,multitask,data,predict}.py`,
`vpdl/slm/{train,model,config}.py`, `vpdl/dl/plm/{peft,finetune}.py`, `vpdl/dl/trainer.py`,
`configs/slm/*.toml`. Tiny models on the CPU only.

## Findings

| ID | Severity | Where | Finding | Status |
|---|---|---|---|---|
| F-1 | HIGH | `vpdl/slm/modeling/continued.py` | Continued pretraining did not resume exactly. Data windows and MLM masks were keyed on (seed, step), but dropout used torch's global RNG, which was neither seeded per step nor checkpointed. Reproduced: 4 uninterrupted steps vs. resume from step 2 diverged at steps 3–4 (5.324 vs 5.316). The guarantee held in `vpdl/slm/train.py` only because that model has no dropout. | **FIXED** — the global RNG is re-seeded from (seed, step) at every step; the same script now matches exactly; test `test_a_resumed_pretraining_run_makes_the_same_dropout_draws` |
| F-3 | MEDIUM | `continued.py:322` | The encoder to adapt was found by `getattr(model, "bert", …)`, so any non-BERT MLM family (RoBERTa, ELECTRA…) failed at PEFT. | **FIXED** — uses HF's `base_model_prefix` |
| MASTER-1 | HIGH | `finetune.py` `_evaluate_and_save` | Threshold chosen on raw validation probabilities, then applied to calibrated scores (see CALIBRATION_AUDIT.md). | **FIXED** |
| J-1 | HIGH | `finetune.py` | The run record dropped the backbone's version (weights sha, HF commit, revision); an unpinned hub model could change between two runs that look identical. | **FIXED** — recorded in `model_version.backbone_version` and in `model.pt` |
| — | HIGH | `finetune.py` | Training did not run any leakage check itself (see CURRENT_CODE_LEAKAGE_AUDIT.md, M-3). | **PARTLY FIXED** — `input_gate` refuses conclusion sentences and forbidden features before anything is built |
| O-14 | LOW | `vpdl/slm/cli.py` `cmd_finetune` | CLI overrides applied only when truthy, so `--lr 0` was silently dropped. | **FIXED** — `is not None` |
| F-2 | MEDIUM | `continued.py:330`, shared `Trainer` | Weight decay also applies to biases/LayerNorm/embeddings (the from-scratch loop excludes them). | OPEN — optimisation default, needs a decision |
| O-6 | MEDIUM | `finetune.py:157` | SLM runs inherit DL trainer/PEFT defaults no SLM config sets (`max_grad_norm=1.0`, cosine schedule, `lora_alpha=16`, `lora_dropout=0.05`). | OPEN — set explicitly in `configs/slm/*.toml` |
| F-4 | LOW | `predict.py`, `multitask.py` | MC-dropout also re-arms DL-modality dropout, inflating uncertainty once EXP-018 runs. | OPEN |
| F-5 | LOW | `modeling/peft.py:74` | BERT → decoder LoRA target fallback is silent. | OPEN |

## Multi-task

Heads: five-class (soft targets for pairs), ACMG (28 codes, gene-masked), evidence type
(multi-label) and polarity per sentence. `masked_bce` and `multitask_loss` guard all-missing
batches and tasks (no NaN). Rows without a class target (missing/conflicting/out-of-scope)
are excluded from the class loss; VUS rows are not. Loss weights come from `LossWeights`.
Task-dominance cannot be judged without real training.

## PEFT (verified by running)

Frozen base weights receive no gradient; LoRA `B` is zero-initialised (so `A.grad` is zero
at step 0, as expected); trainable fractions are reported; `full` fine-tuning above the size
limit is refused without `allow_full`; unmatched LoRA targets raise; LoRA + gradient
checkpointing passes gradients. Adapter-only checkpoints load onto the base with
`strict=False` and reject unexpected keys.

## Checkpoints

No SLM checkpoint exists locally (`runs/slm/` is absent; the DGX holds a trial run). The two
`.pt` files in the repository (`data/processed/transfer/*.pt`) belong to the v1 archive and
were not opened or modified. The finetune dry run's checkpoint round-trip is identical
(smoke run). The packaged `model.pt` now carries a provenance block; there is still no
format-checked loader for it (J-2, OPEN).

## Deferred to the DGX

Loss curves at scale; whether F-1's divergence mattered at thousands of steps (moot after
the fix); LoRA + compile; `pytest tests/slm` on the DGX.
