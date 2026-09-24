# DL architecture audit — local code audit, 2026-09-24

Scope: `vpdl/dl/{fusion,trainer,runner,context,calibration,feature_store,failure}.py`,
`vpdl/dl/plm/**`, `vpdl/dl/pretrain/**`, `vpdl/models/**`, and how `vpdl/experiment.py`
calls them. Baseline commit `faf8de2` (branch `v2/rebuild`). Windows PC, CPU only.
The architecture was **not redesigned**. Nothing was trained.

## Findings

| ID | Severity | Where | Finding | Status |
|---|---|---|---|---|
| A-1 | HIGH (latent) | `vpdl/dl/plm/finetune.py:221` | The PLM fine-tune's `TrainConfig` left `checkpoint_trainable_only` at `False`, so every resumable checkpoint would hold the full frozen backbone twice (`model` + `best_state`) per epoch — tens of GB for ESM-2 3B. `vpdl/dl/pretrain/run.py` already sets it. Latent: no CLI path sets `checkpoint_dir` for `plm_finetune` today. | **FIXED** — set to `True`; test `test_plm_finetune_checkpoints_hold_only_what_trains` |
| N-1 | MEDIUM | `vpdl/dl/trainer.py:60,320` | `TrainConfig` had no validation; `batch_size = 0` was silently clamped to 1 (a different run from the one configured). | **FIXED** — `__post_init__` refuses non-positive epochs/patience/batch_size/grad_accum/keep_last, warmup outside [0,1), unknown schedule |
| F-2 | MEDIUM | `vpdl/dl/trainer.py:183` | The shared `Trainer` applies weight decay to every parameter, including biases and LayerNorm (the from-scratch loop `vpdl/slm/train.py` already excludes them). | OPEN — changing it changes every DL model's optimisation; a scientific default, left for a decision |
| O-6 | MEDIUM | `vpdl/dl/trainer.py:60-81` | `TrainConfig.identity()` hashes every field, so adding a field on the DL side blocks resuming older SLM checkpoints that reuse this trainer. | OPEN — noted for anyone adding a field |

## Verified correct (by running or by reading)

- True masked marginal: the `<mask>` token is substituted at the scored site before the
  forward pass (ran on the tiny backbone). v1's "masked marginal never masked" bug is not
  present in v2.
- The siamese PLM fine-tune reproduces the wild-type-marginal zero-shot score exactly at
  initialisation (ran).
- No double sigmoid / logits-probability confusion: every model trains on raw logits with
  `binary_cross_entropy_with_logits` and applies `sigmoid` only at prediction.
- Gradient accumulation (short last window divisor), AMP ordering
  (`unscale_` → clip → `step` → `update` → `zero_grad` → `scheduler.step`), and checkpoint
  resume (format tag, config identity, seed, optimizer, scheduler, scaler, full RNG) are
  correct. Checkpoint writes are atomic (`os.replace`).
- LoRA + gradient checkpointing on a frozen backbone passes gradients to LoRA parameters
  under transformers 5.17 (ran).
- No BatchNorm in the torch models, so MC-dropout's `train()` toggle cannot corrupt
  running statistics; RNG is saved and restored around MC sampling.
- Embedding-block and tabular standardisation are fitted on the inner-training fold only.

## Deferred to the DGX

OOM back-off of `resolve_batch_size` under real memory pressure; `torch.compile` on
aarch64/GB10; ESM-2 beyond 1,022 residues (MSH6); bf16 numerics on Blackwell.
