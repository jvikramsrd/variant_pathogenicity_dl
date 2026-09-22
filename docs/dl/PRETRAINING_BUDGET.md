# Pretraining and DL compute budget (estimates)

**Every number here is an estimate from stated assumptions, not a measurement.**
Replace each with the measured value from the first DGX run and record the
measurement in `docs/RUNLOG.md`.

## Assumptions

| quantity | value | source |
|---|---|---|
| bf16 matmul peak on this DGX Spark | 90.1 TFLOPS | measured 2026-09-22 (SLM benchmark) |
| sustained training throughput | ~30 TFLOPS | the SLM reached 30.7 TFLOPS with `torch.compile`; ESM shapes may differ |
| sustained inference throughput | ~40 TFLOPS | assumption |
| ESM-2 650M parameters N | 6.5e8 | model card |
| FLOPs per token, inference | 2N ≈ 1.3e9 | standard |
| FLOPs per token, LoRA training | ≈ 4N ≈ 2.6e9 (forward + activation gradients; frozen weight gradients skipped) | standard approximation |
| attention overhead at 512–1,022 tokens | < 10 % of the above | 4·T·d·L vs 2N |
| MMR corpus size | ~5,000 sequences (pinned queries; unknown until fetched) | assumption — the manifest records the real count |
| crop | 512 residues | `configs/dl/pretrain_dgx.toml` |

## Pretraining (per arm, per fold)

Tokens per epoch ≈ 5,000 × 512 = 2.6e6.

| arm | passes / sample | FLOPs / epoch | time / epoch | ≤ 20 epochs |
|---|---|---|---|---|
| P1 | 1 | 6.7e15 | ~4 min | ~1.2 h |
| P2 | 2 | 1.3e16 | ~8 min | ~2.5 h |
| P3 | 4 | 2.7e16 | ~15 min | ~5 h |
| P4 | 5 | 3.3e16 | ~19 min | ~6 h |

Early stopping (patience 3) will usually end well before 20 epochs.

| scope | upper bound | likely (≈ 8 epochs) |
|---|---|---|
| one fold, P1–P4 | ~15 h | ~6 h |
| strict, four folds | ~60 h | ~24 h |
| transductive, one model per arm | ~15 h | ~6 h |

Recommended order: transductive P1 first (cheapest signal), then strict P1 for
all folds; run P2–P4 only if P1 is not already a regression.

## Downstream extraction and training

| job | size | estimate |
|---|---|---|
| masked marginal, one backbone | positions with variants (MSH2 all 934 via DMS + others ≈ 1,500), one pass each | ~1–3 min |
| embeddings, `full`, labelled + functional rows | ≈ 17,000 variants (mostly the MSH2 assay) × one VT pass of ~950 tokens | ~10 min |
| embeddings, all rows (~74,000) | ~74,000 VT passes | ~40 min per backbone per policy |
| `hierarchical` policy | centred + sliding passes | ~2× `full` |
| ESM-1b vs ESM-2 × 5 context policies (labelled + functional) | 10 extractions | ~2–3 h |
| `plm_finetune` LoRA, one cell (4 folds × 1 seed) | ~330 training variants × 2 passes × 1,022 tokens × ≤ 10 epochs per fold | ~40 min; ~2 h for 3 seeds |
| probes / fusion / sequence baselines | cached inputs | seconds to minutes per cell |

## Memory

| object | size |
|---|---|
| ESM-2 650M fp32 master weights | 2.6 GB |
| LoRA rank 16 (all projections) + AdamW state | ~0.1 GB |
| activations, 16 × 512 tokens, bf16, checkpointed | a few GB |
| one embedding entry, 17k variants × 6 × 1,280 × fp16 | ~0.27 GB (~1.1 GB for all rows) |
| strict pretraining checkpoints (trainable only, keep 2) | megabytes per arm |

The 128 GB unified pool is not the constraint anywhere in this plan; wall
clock is. Do not scale to ESM-2 3B unless the 650M model demonstrably
underfits (training and validation loss both high), and record why.

## Storage

Feature store for the full plan (2 backbones × 5 policies × labelled +
functional, plus P1–P4 × 4 folds): ~15 GB, all regenerable, and ignored by git
(`features/`, `data/dl/`, DL checkpoints and exported `.npy` arrays are in
`.gitignore`).
