# DGX Spark optimisation (DL branch)

Target: GB10 Grace-Blackwell, aarch64, 128 GB unified CPU/GPU memory, bf16
native. Measured on this box so far (docs/RUNLOG.md, 2026-09-22): 90.1 TFLOPS
bf16 matmul; the SLM trained at 30.7 TFLOPS with `torch.compile` (1.5× eager,
identical losses) and needed `sudo apt install python3.12-dev` for Triton.
Everything below is either implemented and switchable, or a recommendation
labelled as one. Check what the box reports first:

```bash
python scripts/check_hardware.py --smoke --out runs/dl/hardware.json
```

## What is implemented, and where

| technique | where | default |
|---|---|---|
| BF16 autocast (no loss scaler) | `dl/trainer.py::resolve_precision`, `dl/plm/forward.py::inference_autocast` | on when the GPU supports bf16; fp16 + GradScaler otherwise; fp32 on CPU |
| mixed precision for the loss | losses computed in fp32 inside autocast | always |
| gradient checkpointing | `PLMFinetuneClassifier(gradient_checkpointing=True)`, `PretrainConfig.gradient_checkpointing` | on for 650M fine-tuning / pretraining |
| gradient accumulation | `TrainConfig.grad_accum` (short tail windows divided by their own length) | 1 (fine-tune), 2 (pretraining) |
| efficient attention | ESM loaded with `attn_implementation="sdpa"` → PyTorch SDPA picks FlashAttention / memory-efficient kernels where shape and GPU allow; falls back to eager, logged | sdpa |
| torch.compile | `TrainConfig.compile`, `--compile`; checkpoints store the uncompiled module, so eager and compiled runs resume each other | on in `configs/dl/pretrain_dgx.toml` |
| fused AdamW | `Trainer._optimizer` (`fused=True` on CUDA, logged fallback) | on |
| LoRA / adapters / last-N | `dl/plm/peft.py` | LoRA |
| cached embeddings | `dl/feature_store.py` — extract once, every probe/fusion cell reads the cache | always |
| memory mapping | feature arrays are `.npy`, opened `mmap_mode="r"`; a fold reads only its rows | always |
| efficient batching | length-sorted batches for inference (`hidden_states`, `site_log_probs`); chunked VT reduction (no `[N, L, d]` tensor ever) | always |
| checkpoint resumption | `--resume`: model, optimizer, scheduler, scaler, epoch, step, best weights, patience, all RNGs, config hash | per run dir |
| SIGTERM / Ctrl-C | handler requests a stop at the next step boundary; the last complete epoch is the resume point | always |

Data loading is in-process (`batch_fn(indices)`), not a multi-worker
DataLoader: the inputs are small (≤ 17k variants, a few thousand corpus
sequences) and tokenisation is microseconds. Worker processes would add
startup cost and cross-process RNG hazards for no measurable gain here.

## Per workload

### ESM / ESM-2 embedding extraction
* `torch.inference_mode()` + bf16 autocast, only the last layer kept (a
  forward hook when `--layer` selects an earlier one; never
  `output_hidden_states=True`, which retains all 34 layers).
* One WT pass per window per protein; one VT pass per unique `(position, mut)`;
  under `sliding`, only windows containing the mutation are recomputed.
* `--batch-size 8`–`32` is safe for 650M on this box; `--chunk 64` bounds peak
  memory at chunk × L × d floats regardless of the variant count.

### MSH6 (1,360 residues)
* `full` on ESM-2: one 1,362-token pass. Attention cost ~1.8× a 1,022-token
  pass; memory is not an issue at inference.
* ESM-1b cannot see > 1,022 residues: use `hierarchical`, `centered` or
  `asymmetric` (refused otherwise, `ContextError`).
* Fine-tuning uses one window per variant (`centered` default, ≤ 1,022 tokens),
  with gradient checkpointing on.

### Pretraining
* LoRA rank 16 on all projections; the frozen base means no optimizer state
  for 650M weights; checkpoints hold trainable weights only
  (`checkpoint_trainable_only=True`).
* 512-residue crops; micro-batch 16 × accumulation 2; bf16; compile on.
* Early stopping on held-out MLM loss; ≤ 20 epochs.

### Fusion and probes
* Inputs are cached vectors; each fold standardises embeddings on its
  inner-training rows only. CPU-speed work; the GPU is not the bottleneck.

## Recommendations (not measured yet)

* Do **not** increase model size because the memory allows it. ESM-2 3B is
  registered but should be run only with evidence the 650M model underfits.
* Prefer three seeds of a small configuration over one seed of a large one:
  on this panel the seed spread (MCC SD up to 0.056) dominates small gains.
* Run CPU-bound jobs (canonical, structure SASA, leakage reports) while a GPU
  job trains; they do not compete for the GPU.
* Do not run DL jobs while `slm-train` is using the GPU unless both fit — the
  unified pool is shared with the CPU, and an OOM there kills both.
