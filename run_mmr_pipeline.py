#!/usr/bin/env python3
"""End-to-end Lynch-syndrome MMR pipeline: download -> clean/process -> train.

Runs PROJECT_PLAN.md Phases 0-3 back to back as one command:

    1. Environment setup (reused from run_pipeline.py -- unchanged).
    2. Unit tests.
    3. DOWNLOAD + CLEAN (broad pretraining panel): resolve the DMS-wide gene
       panel, pull ClinVar/ProteinGym/AlphaMissense/zero-shot scores (+
       gnomAD, AlphaFold, InterPro, UniProt point features when requested),
       merge into one audited, leakage-checked table.
    4. DOWNLOAD + CLEAN (MMR-specific): pin MLH1/MSH2/MSH6/PMS2, reuse the
       shared multi-source builder restricted to those four genes, join
       gnomAD (AF + gene constraint), AlphaFold, InterPro, UniProt point
       features, MaveDB DMS and (optionally) CIMRA OddsPath, apply the PMS2
       pseudogene gate, emit the leave-one-gene-out split manifest.
    5. TRAIN, stage 1: pretrain the classification head on the broad panel's
       frozen ESM-2 embeddings (leave-one-MMR-gene-out by default). This is
       feature-extraction + linear-probe warm-up, not the main DL training.
    6. TRAIN, stage 2: warm-start fine-tune on the MMR-specific data with
       leave-one-gene-out evaluation (frozen-embedding branch/fusion
       ablations).
    7. TRAIN, stage 2b: true ESM-2 backbone gradient fine-tuning -- ProPath's
       Siamese recipe by default (or CSBJ's token classifier) via
       src/esm_finetune.py. This is the actual deep-learning training stage
       (gradients flow into the transformer, not just a head on frozen
       embeddings) and runs by default; pass --no-full_finetune to skip it
       and stop at the frozen-embedding stages.

Steps 2-7 are the numbered "Stage i/N" banners printed at run time; N adapts
to the flags you passed (--skip_tests, --skip_build, --no-full_finetune), so
the banner sequence always matches what is actually going to run. Step 1
(environment setup) happens before the first banner.

Stage 2b needs an accelerator in practice. If no CUDA/MPS device is visible
the pipeline stops before that stage with an explanation rather than starting
a fine-tune that would take days on CPU; pass --allow_cpu_finetune to run it
anyway, or --no-full_finetune to skip it.

Stops there by design -- no calibration, no LLM branch, no fusion research
beyond the ablations already built into stage 2. Those are separate,
standalone tools (scripts/compare_*.py, scripts/eval_leave_one_protein_out.py,
scripts/build_cluster_split.py) you can run manually; this pipeline only
covers data collection, data preprocessing, and DL through fine-tuning.

This script is a sibling to run_pipeline.py (the original single-/multi-gene
extended-dataset workflow) and reuses its environment-setup helpers rather
than duplicating them; run_pipeline.py itself is untouched.

Examples
--------
    # Full run including real backbone fine-tuning (the default). On a GPU
    # box, also raise the feature/model quality:
    python run_mmr_pipeline.py --features esm+priors \
        --esm_model facebook/esm2_t33_650M_UR50D --all_sources

    # CPU-friendly / quick smoke test: skip the expensive backbone
    # fine-tuning stage and stop at the frozen-embedding stages
    python run_mmr_pipeline.py --no-full_finetune

    # Re-run training only, reusing already-downloaded/cleaned data
    python run_mmr_pipeline.py --skip_build
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_pipeline import banner, ensure_environment, env_flag, run  # noqa: E402

DEFAULT_ESM_MODEL = "facebook/esm2_t33_650M_UR50D"
FEATURE_CHOICES = ("priors", "esm+priors")
PRETRAIN_MODE_CHOICES = ("leave_gene_out", "practical")
EVAL_CHOICES = ("lopo", "holdout")
FINETUNE_MODE_CHOICES = ("wt_site", "siamese")
MMR_GENES = ("MLH1", "MSH2", "MSH6", "PMS2")

PRETRAIN_CHECKPOINT_NAME = "mmr_pipeline_pretrain.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, clean, and train the Lynch-syndrome MMR "
                    "pathogenicity model end to end (Phases 0-3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--features", choices=FEATURE_CHOICES,
        default=os.environ.get("FEATURES", "priors"),
        help="Feature mode for both training stages (env: FEATURES). "
             "'priors' runs anywhere with no GPU; 'esm+priors' needs an "
             "ESM-2 forward pass per variant (GPU strongly advised).")
    parser.add_argument(
        "--esm_model", default=os.environ.get("ESM_MODEL", DEFAULT_ESM_MODEL),
        help="HuggingFace ESM-2 checkpoint for esm+priors mode / full "
             "fine-tuning (env: ESM_MODEL). Use facebook/esm2_t12_35M_UR50D "
             "for a fast CPU smoke test.")
    parser.add_argument(
        "--skip_build", action="store_true", default=env_flag("SKIP_BUILD"),
        help="Skip stages 3-4 (download/clean) and reuse whatever's already "
             "on disk under data/processed and data/mmr (env: SKIP_BUILD).")
    parser.add_argument(
        "--skip_tests", action="store_true",
        help="Skip stage 2 (unit tests).")

    broad = parser.add_argument_group("stage 3 -- broad pretraining panel")
    broad.add_argument(
        "--panel_genes", default=None,
        help="Comma-separated gene panel for pretraining. Default: every "
             "human protein with a ProteinGym DMS assay "
             "(scripts/make_expanded_panel.py).")
    broad.add_argument(
        "--all_sources", action="store_true",
        help="Also fetch gnomAD (AF + gene constraint), AlphaFold pLDDT, "
             "InterPro, and UniProt point features for the broad panel, not "
             "just the MMR genes (opt-in: one extra REST/GraphQL call per "
             "gene per source, so this can add real time for an 80+ gene "
             "panel). Passed through as --all_sources to "
             "scripts/build_extended_dataset.py.")

    mmr = parser.add_argument_group("stage 4 -- MMR-specific dataset")
    pms2 = mmr.add_mutually_exclusive_group()
    pms2.add_argument(
        "--exclude_pms2", action="store_true",
        help="Drop PMS2 entirely until a homology-region confirmation table "
             "exists (this is the pipeline's default behaviour when none of "
             "the three PMS2 options below is given -- standard NGS calls in "
             "the PMS2CL pseudogene-homology region, exons 11-15, are "
             "untrustworthy).")
    pms2.add_argument(
        "--pms2_homology_csv", type=Path, default=None,
        help="CSV of PMS2 exon 11-15 substitutions with an "
             "orthogonally_confirmed flag; overrides --exclude_pms2.")
    pms2.add_argument(
        "--pms2_codon_range", type=int, nargs=2, default=None,
        metavar=("START", "END"),
        help="Explicit protein-coordinate span of the homology region; "
             "overrides --exclude_pms2. Use '--pms2_codon_range 382 862' for "
             "exons 11-15 -- derived from the Ensembl exon table for MANE "
             "Select ENST00000265849 by "
             "scripts/derive_pms2_homology_range.py (which self-validates "
             "against the 862 aa pinned for P54278). This is the preferred "
             "way to include the fourth Lynch gene: it keeps PMS2 codons "
             "1-381 instead of dropping the gene, at the cost of the 56%% of "
             "residues that genuinely are unreliable on short-read calls.")
    mmr.add_argument(
        "--gene_constant_priors", choices=("auto", "drop", "keep"),
        default="auto",
        help="gnomAD gene-level constraint columns (pLI, oe_lof, oe_mis, "
             "mis_z, syn_z). Constant within a gene, so under the "
             "leave-one-gene-out evaluation they act as a gene identifier "
             "rather than variant evidence -- keeping them collapsed MLH1 to "
             "ROC-AUC 0.500 in every seed tried. 'auto' therefore passes "
             "'drop' to BOTH training stages whenever --eval lopo, so the "
             "warm-start schemas still match; 'keep' restores the old "
             "behaviour.")
    mmr.add_argument(
        "--rank_normalize", choices=("off", "add", "replace"), default="off",
        help="Give the head within-gene percentile ranks of the published "
             "scores instead of (or alongside) their raw values. Measured on "
             "this panel it did not beat --gene_constant_priors drop on its "
             "own; 'add' helped MSH6 (0.971 -> 0.985 ROC-AUC) but "
             "destabilised MLH1's threshold. Off by default.")
    mmr.add_argument(
        "--cimra_csv", type=Path, default=None,
        help="Optional CIMRA OddsPath CSV (src/cimra.py documents the "
             "schema; no bulk CIMRA API exists). Validation-only, never fed "
             "into training.")
    mmr.add_argument(
        "--min_stars", type=int, default=2, choices=[1, 2, 3, 4],
        help="Minimum ClinVar review stars for a row to carry a label. "
             "Applies to BOTH builds (broad panel and MMR); lowering it "
             "trades label confidence for label count.")

    train1 = parser.add_argument_group("stage 5 -- pretrain (broad panel)")
    train1.add_argument(
        "--pretrain_mode", choices=PRETRAIN_MODE_CHOICES, default="leave_gene_out",
        help="'leave_gene_out' excludes every MMR gene from pretraining "
             "(the honest transfer estimate); 'practical' allows them.")

    train2 = parser.add_argument_group("stage 6 -- fine-tune (MMR genes)")
    train2.add_argument(
        "--eval", choices=EVAL_CHOICES, default="lopo",
        help="'lopo' evaluates every leave-one-gene-out split; 'holdout' "
             "evaluates a single --holdout_gene.")
    train2.add_argument("--holdout_gene", choices=MMR_GENES, default=None)
    train2.add_argument(
        "--n_bootstrap", type=int, default=10_000,
        help="Bootstrap CI iterations (lower for a quick smoke run).")
    train2.add_argument(
        "--full_finetune", action=argparse.BooleanOptionalAction, default=True,
        help="Run true ESM-2 backbone gradient fine-tuning "
             "(src/esm_finetune.py, scripts/finetune_esm_mmr.py) after the "
             "frozen-embedding stage -- ProPath's Siamese recipe by default. "
             "This is the actual DL training stage, not a linear probe over "
             "frozen embeddings; it is the pipeline default. Expensive: the "
             "pipeline refuses to start it without a CUDA/MPS device unless "
             "--allow_cpu_finetune is given. Pass --no-full_finetune to skip "
             "it and stop at the frozen-embedding stages (CPU-friendly).")
    train2.add_argument(
        "--finetune_mode", choices=FINETUNE_MODE_CHOICES, default="siamese",
        help="Stage 2b recipe (ignored under --no-full_finetune): "
             "'siamese'=ProPath-style WT+VT forward passes; "
             "'wt_site'=CSBJ-style single WT forward pass.")
    train2.add_argument(
        "--n_unfrozen_layers", type=int, default=-1,
        help="Stage 2b backbone depth (ignored under --no-full_finetune): "
             "-1 = full backbone fine-tune, 0 = frozen (ablation floor), "
             "N>0 = unfreeze the last N transformer layers only. A small N "
             "is the cheapest way to get real backbone gradients.")
    train2.add_argument(
        "--allow_cpu_finetune", action="store_true",
        help="Run stage 2b even when no CUDA/MPS device is visible. Without "
             "this the pipeline stops before stage 2b on a CPU-only box, "
             "because a full backbone fine-tune there takes days rather than "
             "hours. Reasonable to pass with a tiny checkpoint "
             "(--esm_model facebook/esm2_t12_35M_UR50D), a shallow unfreeze "
             "(--n_unfrozen_layers 2) and one --holdout_gene, as a smoke "
             "test that the stage runs at all.")
    train2.add_argument(
        "--gradient_checkpointing", action="store_true",
        help="Stage 2b memory/compute trade (ignored under "
             "--no-full_finetune): recompute activations instead of storing "
             "them. Roughly 30%% slower, and it is often what makes the "
             "650M checkpoint fit on a consumer GPU.")
    train2.add_argument(
        "--batch_size", type=int, default=8,
        help="Stage 2b micro-batch -- the batch that has to fit in VRAM. "
             "A full siamese fine-tune of esm2_t33_650M does not fit at 8 on "
             "anything smaller than an 80 GB card; drop this to 1-2 and raise "
             "--grad_accum to keep the effective batch at ProPath's 8.")
    train2.add_argument(
        "--grad_accum", type=int, default=1,
        help="Stage 2b gradient accumulation. Effective batch = --batch_size "
             "* --grad_accum, so the optimizer still sees ProPath's batch of "
             "8 when the card can only hold 1 example at a time.")
    train2.add_argument(
        "--max_residues", type=int, default=1022,
        help="Stage 2b mutation-centred crop width. Activation memory is "
             "linear in this; halving it halves the activation term.")
    train2.add_argument(
        "--no_amp", dest="amp", action="store_false",
        help="Disable stage 2b mixed-precision autocast (bf16 where "
             "supported, else fp16). On by default on CUDA; it roughly "
             "halves activation memory.")
    train2.set_defaults(amp=True)
    train2.add_argument(
        "--epochs", type=int, default=20,
        help="Stage 2b epoch cap. Deliberately above src/esm_finetune.py's "
             "ProPath default of 10: --patience early-stops on validation "
             "ROC-AUC, so a higher cap costs nothing on splits that plateau "
             "early and only buys epochs on splits still improving.")
    train2.add_argument(
        "--patience", type=int, default=3,
        help="Stage 2b early-stopping patience, in epochs without a "
             "validation ROC-AUC improvement. This, not --epochs, is what "
             "actually decides how long a split runs.")
    train2.add_argument(
        "--clinical_weight", type=float, default=5.0,
        help="Stage 2b per-sample loss weight for clinical-source rows. Note "
             "that stage 2b already filters to clinvar/pg_clinical rows only, "
             "so any value here applies uniformly -- it does not reweight one "
             "class against another, it just scales the loss, which shifts "
             "how often gradient clipping binds. Use 1.0 for an unscaled run.")
    train2.add_argument(
        "--skip_vram_preflight", action="store_true",
        help="Run stage 2b even when the estimated peak VRAM exceeds the "
             "detected card. The estimate is approximate; pass this if you "
             "believe it is wrong for your setup.")

    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Re-download/re-fetch every source instead of "
                             "reusing on-disk caches.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print every stage's command instead of running "
                             "it -- verify the planned sequence before "
                             "committing to a multi-hour/GPU run.")
    return parser.parse_args()


def detect_accelerator(py: str) -> str:
    """Return 'cuda', 'mps', 'cpu' or 'unknown' for the interpreter *py*.

    Probed in the child interpreter rather than this one: after
    ensure_environment() the stages run inside a virtualenv that may have a
    different torch build (e.g. requirements-cuda.txt) from whatever is
    importable here. 'unknown' means the probe itself failed -- torch missing
    or not yet installed -- and is deliberately treated as "do not block".
    """
    import subprocess
    probe = (
        "import torch;"
        "cuda=torch.cuda.is_available();"
        "print('cuda' if cuda else "
        "('mps' if getattr(torch.backends,'mps',None) and "
        "torch.backends.mps.is_available() else 'cpu'));"
        "print(torch.cuda.get_device_properties(0).total_memory/2**30 "
        "if cuda else 0.0)"
    )
    try:
        out = subprocess.run([py, "-c", probe], capture_output=True,
                             text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return "unknown", 0.0
    if out.returncode != 0:
        return "unknown", 0.0
    lines = out.stdout.strip().splitlines()
    device = lines[0].strip() if lines else ""
    try:
        vram_gib = float(lines[1]) if len(lines) > 1 else 0.0
    except ValueError:
        vram_gib = 0.0
    if device not in ("cuda", "mps", "cpu"):
        return "unknown", 0.0
    return device, vram_gib


#: Per-layer shape of the ESM-2 checkpoints this pipeline can drive, used by
#: :func:`estimate_finetune_vram_gib`. (layers, hidden, ffn, params)
_ESM_SHAPES = {
    "facebook/esm2_t6_8M_UR50D":     (6, 320, 1280, 7_400_000),
    "facebook/esm2_t12_35M_UR50D":   (12, 480, 1920, 33_600_000),
    "facebook/esm2_t30_150M_UR50D":  (30, 640, 2560, 148_000_000),
    "facebook/esm2_t33_650M_UR50D":  (33, 1280, 5120, 652_000_000),
    "facebook/esm2_t36_3B_UR50D":    (36, 2560, 10240, 2_840_000_000),
}


def estimate_finetune_vram_gib(esm_model: str, mode: str, n_unfrozen_layers: int,
                               batch_size: int, max_residues: int,
                               grad_accum: int, amp: bool,
                               gradient_checkpointing: bool) -> float | None:
    """Rough peak VRAM for one stage-2b training step, in GiB.

    Deliberately a closed-form estimate rather than a trial allocation: the
    point is to reject an impossible configuration *before* the hour of
    downloads and the two frozen-embedding stages, on a box where the OOM
    would otherwise land at the very end of the pipeline.

    Two terms dominate:

    * **Static state** -- fp32 weights, plus fp32 gradients and two AdamW
      moments for every *trainable* parameter. A full fine-tune pays
      16 bytes/param; freezing all but the last N layers pays 4 bytes/param
      plus 12 bytes for the N layers' share.
    * **Activations** -- every saved tensor of every layer that participates
      in backward, times two for siamese's WT+VT passes. Without
      checkpointing, frozen leading layers save nothing (autograd builds no
      graph for them); autocast halves what the rest save; and gradient
      checkpointing replaces the per-layer total with one layer's worth plus
      the layer boundaries.

    Accuracy is roughly +/-25%%, which is enough to separate "fits" from
    "needs 3x the card". Returns ``None`` for an unrecognised checkpoint.
    """
    shape = _ESM_SHAPES.get(esm_model)
    if shape is None:
        return None
    n_layers, hidden, _ffn, n_params = shape
    grad_layers = n_layers if n_unfrozen_layers == -1 else max(0, min(n_unfrozen_layers, n_layers))
    trainable = n_params * (grad_layers / n_layers)

    static = (n_params * 4 + trainable * 12) / 2**30

    seq = max_residues + 2                       # <cls> + residues + <eos>
    unit = batch_size * seq * hidden * 4         # one B x L x H fp32 tensor
    per_layer = 16 * unit                        # saved tensors per encoder layer
    if amp:
        per_layer /= 2
    passes = 2 if mode == "siamese" else 1
    if gradient_checkpointing:
        # Only layer boundaries survive the forward; recompute peaks at one
        # layer. Boundaries are counted over *all* layers, not just the
        # trainable ones: gradient_checkpointing_enable() forces requires_grad
        # on the embedding output, so even a partial unfreeze backpropagates
        # (cheaply, and pointlessly) through the frozen stack below.
        activations = passes * (n_layers * unit + per_layer)
    else:
        activations = passes * grad_layers * per_layer
    # grad_accum does not change peak: it is the same micro-batch, more steps.
    return static + activations / 2**30


def _suggest_fitting_config(args, vram_gib: float) -> str:
    """First configuration that fits *vram_gib*, as a copy-pasteable flag list.

    Ordered by how little it gives up: keep the requested backbone depth and
    the effective batch, spend compute (checkpointing) and precision (AMP)
    first, shrink the crop next, and only then fall back to unfreezing fewer
    layers -- which changes what is actually being benchmarked.
    """
    # Stricter than the 0.9 rejection threshold: a suggestion the user will
    # actually run should clear the estimate's own ~25%% error bar, not sit
    # one padded batch away from the same OOM.
    budget = 0.7 * vram_gib
    effective = max(1, args.batch_size * args.grad_accum)
    for depth in (args.n_unfrozen_layers, 8, 6, 4, 2):
        for residues in (args.max_residues, 512):
            for micro in (args.batch_size, 4, 2, 1):
                if micro > effective:
                    continue
                need = estimate_finetune_vram_gib(
                    args.esm_model, args.finetune_mode, depth, micro,
                    residues, 1, True, True)
                if need is None or need > budget:
                    continue
                flags = [f"--batch_size {micro}",
                         f"--grad_accum {max(1, effective // micro)}",
                         "--gradient_checkpointing"]
                if residues != args.max_residues:
                    flags.append(f"--max_residues {residues}")
                if depth != args.n_unfrozen_layers:
                    flags.append(f"--n_unfrozen_layers {depth}")
                note = (f" (~{need:.1f} GiB; effective batch still "
                        f"{micro * max(1, effective // micro)})")
                if depth != args.n_unfrozen_layers:
                    note += (f"\n     Note: --n_unfrozen_layers {depth} is a "
                             "partial fine-tune, not the full-backbone run you "
                             "asked for -- the frozen leading layers are what "
                             "buy the memory back.")
                return "Fits here: " + " ".join(flags) + note
    return (f"Nothing in the search space fits {vram_gib:.1f} GiB with "
            f"{args.esm_model}. Use a smaller checkpoint "
            "(--esm_model facebook/esm2_t30_150M_UR50D) or a larger GPU.")


class StageCounter:
    """Numbered banners whose denominator matches the flags actually passed.

    The stage list is fixed up front from the parsed arguments, so a run with
    --skip_tests does not print "Stage 2/6" as its first line.
    """

    def __init__(self, total: int) -> None:
        self.total = total
        self.n = 0

    def __call__(self, text: str) -> None:
        self.n += 1
        banner(f"Stage {self.n}/{self.total} - {text}")

    @staticmethod
    def skip(text: str) -> None:
        """Announce a stage that will not run. Deliberately unnumbered: the
        numbering counts stages that actually execute, so "Stage 3/4" always
        means three of four real stages are done."""
        banner(f"SKIPPED - {text}")


def _make_runner(env: dict, dry_run: bool):
    def _runner(cmd: list[str], _env: dict) -> None:
        if dry_run:
            print("+ [dry-run] " + " ".join(cmd))
            return
        run(cmd, _env)
    return _runner


def main() -> None:
    args = parse_args()
    if args.eval == "holdout" and args.holdout_gene is None:
        raise SystemExit("--holdout_gene is required with --eval holdout.")
    os.chdir(ROOT)

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    # Stage 2b allocates and frees large, differently-shaped activation
    # buffers every step (padded batches vary in sequence length), which
    # fragments the default caching allocator badly enough to OOM with GiB
    # still nominally free. Expandable segments is PyTorch's own remedy --
    # but it is a POSIX-only allocator backend, and setting it on Windows
    # only earns a UserWarning on the first .to(device) of every run.
    if sys.platform != "win32":
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.dry_run:
        # Environment setup (possible venv creation + a multi-GB requirements
        # install) is a real side effect -- never trigger it just to preview
        # a command sequence. Report what would happen and use the
        # interpreter already running this script instead.
        print("+ [dry-run] would call ensure_environment() "
             "(creates/reuses a venv and installs requirements*.txt)")
        py = sys.executable
    else:
        py = str(ensure_environment(env))
    run_stage = _make_runner(env, args.dry_run)
    overwrite = ["--overwrite_cache"] if args.overwrite_cache else []

    # Stage 2b is the one stage that can silently turn a coffee break into a
    # multi-day run, so decide whether it is viable BEFORE spending an hour on
    # downloads and the frozen-embedding stages -- failing at the end of a
    # long pipeline is the worst possible time to find out.
    device, vram_gib = "unknown", 0.0
    if args.full_finetune:
        device, vram_gib = detect_accelerator(py)
        if device == "cuda":
            need = estimate_finetune_vram_gib(
                args.esm_model, args.finetune_mode, args.n_unfrozen_layers,
                args.batch_size, args.max_residues, args.grad_accum, args.amp,
                args.gradient_checkpointing)
            if need is not None:
                print(f"[preflight] stage 2b estimated peak VRAM "
                      f"~{need:.1f} GiB against {vram_gib:.1f} GiB detected "
                      f"({args.esm_model}, {args.finetune_mode}, micro-batch "
                      f"{args.batch_size} x {args.max_residues} residues, "
                      f"amp={args.amp}, checkpointing="
                      f"{args.gradient_checkpointing}).")
            # 0.9 headroom: the estimate covers the training step, not the
            # driver context, fragmentation, or the eval pass.
            if need is not None and need > 0.9 * vram_gib and not args.skip_vram_preflight:
                fits = _suggest_fitting_config(args, vram_gib)
                msg = (
                    f"Stage 2b as configured needs roughly {need:.1f} GiB of "
                    f"VRAM but this GPU has {vram_gib:.1f} GiB.\n"
                    "  This is the configuration that OOMs deep inside the "
                    "backward pass after the rest of the pipeline has already "
                    "run, so it is refused here instead.\n"
                    f"  {fits}\n"
                    "  Or pass --skip_vram_preflight to try it anyway."
                )
                if args.dry_run:
                    banner("PREFLIGHT WARNING")
                    print(msg)
                    print("\n[dry-run] continuing anyway to show the full "
                          "planned command sequence.")
                else:
                    raise SystemExit("\n" + msg)
        if device == "cpu" and not args.allow_cpu_finetune:
            msg = (
                "Stage 2b (true ESM-2 backbone gradient fine-tuning) is on by "
                "default, but no CUDA or MPS device is visible to "
                f"{py}.\n"
                f"  Fine-tuning {args.esm_model} on CPU is a multi-day job, "
                "not a slow afternoon.\n"
                "  Pick one:\n"
                "    --no-full_finetune       stop at the frozen-embedding "
                "stages (the CPU-friendly run)\n"
                "    --allow_cpu_finetune     run it on CPU anyway (pair with "
                "--esm_model facebook/esm2_t12_35M_UR50D\n"
                "                             --n_unfrozen_layers 2 --eval "
                "holdout --holdout_gene MLH1 for a smoke test)\n"
                "    install a CUDA torch build (requirements-cuda.txt) and "
                "re-run"
            )
            if args.dry_run:
                banner("PREFLIGHT WARNING")
                print(msg)
                print("\n[dry-run] continuing anyway to show the full "
                      "planned command sequence.")
            else:
                raise SystemExit("\n" + msg)

    # Fixed up front so every banner's denominator matches this run's flags.
    total_stages = (
        (0 if args.skip_tests else 1)
        + (0 if args.skip_build else 2)
        + 2
        + (1 if args.full_finetune else 0)
    )
    stage = StageCounter(total_stages)

    if args.skip_tests:
        stage.skip("unit tests (--skip_tests)")
    else:
        stage("unit tests")
        run_stage([py, "tests/test_datasets.py"], env)
        run_stage([py, "tests/test_merge.py"], env)
        run_stage([py, "tests/test_mmr_modules.py"], env)
        run_stage([py, "tests/test_new_data_sources.py"], env)

    panel_json = ROOT / "data" / "raw" / "uniprot" / "expanded_panel.json"
    extended_train_csv = ROOT / "data" / "processed" / "extended" / "extended_dataset_train.csv"
    mmr_csv = ROOT / "data" / "mmr" / "processed" / "extended" / "extended_dataset.csv"

    if args.skip_build:
        stage.skip("download + clean, both panels (--skip_build): reusing "
                   "on-disk data")
        for path in (extended_train_csv, mmr_csv):
            if not path.exists():
                raise SystemExit(f"--skip_build requires an existing {path}; "
                                 "run once without --skip_build first.")
    else:
        stage("download + clean: broad pretraining panel")
        if args.panel_genes:
            genes_args = ["--genes", args.panel_genes]
        else:
            if not panel_json.exists() or args.overwrite_cache:
                run_stage([py, "scripts/make_expanded_panel.py"], env)
            genes_args = ["--panel_file", str(panel_json)]
        build_args = [py, "scripts/build_extended_dataset.py", *genes_args,
                     "--min_stars", str(args.min_stars)] + overwrite
        if args.all_sources:
            build_args.append("--all_sources")
        run_stage(build_args, env)
        run_stage([py, "scripts/audit_extended_dataset.py"], env)

        stage("download + clean: MMR-specific dataset (MLH1/MSH2/MSH6/PMS2)")
        mmr_args = [py, "scripts/build_mmr_dataset.py",
                   "--min_stars", str(args.min_stars)] + overwrite
        if args.pms2_homology_csv is not None:
            mmr_args += ["--pms2_homology_csv", str(args.pms2_homology_csv)]
        elif args.pms2_codon_range is not None:
            mmr_args += ["--pms2_codon_range",
                        str(args.pms2_codon_range[0]), str(args.pms2_codon_range[1])]
        else:
            mmr_args.append("--exclude_pms2")
        if args.cimra_csv is not None:
            mmr_args += ["--cimra_csv", str(args.cimra_csv)]
        run_stage(mmr_args, env)

    # Both stages must agree on the prior-column schema or the stage-2
    # warm-start validation fails, so resolve 'auto' once here and pass the
    # SAME literal setting to stage 1 and stage 2 rather than letting each
    # script resolve it against its own defaults.
    gene_const = args.gene_constant_priors
    if gene_const == "auto":
        gene_const = "drop" if args.eval == "lopo" else "keep"

    stage(f"train, stage 1: pretrain the head on the broad panel "
          f"({args.features} mode, {args.pretrain_mode}) -- frozen embeddings")
    pretrain_args = [
        py, "scripts/pretrain_esm_80.py",
        "--features", args.features, "--mode", args.pretrain_mode,
        "--checkpoint_name", PRETRAIN_CHECKPOINT_NAME,
        "--gene_constant_priors", gene_const,
    ] + overwrite
    if args.features == "esm+priors":
        pretrain_args += ["--esm_model", args.esm_model]
    run_stage(pretrain_args, env)
    checkpoint_path = ROOT / "data" / "processed" / "transfer" / PRETRAIN_CHECKPOINT_NAME

    stage(f"train, stage 2: warm-start fine-tune on MMR data ({args.eval}) "
          f"-- still frozen embeddings")
    transfer_args = [
        py, "scripts/run_mmr_transfer.py",
        "--checkpoint", str(checkpoint_path),
        "--features", args.features, "--eval", args.eval,
        "--n_bootstrap", str(args.n_bootstrap),
        "--gene_constant_priors", gene_const,
        "--rank_normalize", args.rank_normalize,
    ] + overwrite
    if args.features == "esm+priors":
        transfer_args += ["--esm_model", args.esm_model]
    if args.eval == "holdout":
        transfer_args += ["--holdout_gene", args.holdout_gene]
    run_stage(transfer_args, env)

    if args.full_finetune:
        stage(f"train, stage 2b: full ESM-2 backbone fine-tune "
              f"({args.finetune_mode}, device={device}) -- the actual DL "
              f"training stage")
        ft_args = [
            py, "scripts/finetune_esm_mmr.py",
            "--mode", args.finetune_mode, "--esm_model", args.esm_model,
            "--n_unfrozen_layers", str(args.n_unfrozen_layers),
            "--eval", args.eval, "--n_bootstrap", str(args.n_bootstrap),
            "--batch_size", str(args.batch_size),
            "--grad_accum", str(args.grad_accum),
            "--max_residues", str(args.max_residues),
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--clinical_weight", str(args.clinical_weight),
        ]
        if args.gradient_checkpointing:
            ft_args.append("--gradient_checkpointing")
        if not args.amp:
            ft_args.append("--no_amp")
        if args.eval == "holdout":
            ft_args += ["--holdout_gene", args.holdout_gene]
        run_stage(ft_args, env)
    else:
        stage.skip("train, stage 2b (--no-full_finetune): stopping at the "
                   "frozen-embedding stages, no backbone gradient fine-tuning")

    banner("DONE")
    print("Broad panel  : data/processed/extended/extended_dataset_train.csv")
    print("MMR dataset  : data/mmr/processed/extended/extended_dataset.csv")
    print("Pretrain ckpt: " + str(checkpoint_path))
    print("Fine-tune out: data/processed/mmr_transfer/mmr_transfer_results_*.csv")
    if args.full_finetune:
        print("Full FT out  : data/processed/esm_finetune/esm_finetune_results_*.csv")
    print("Run log      : append a dated entry to docs/RUNLOG.md")


if __name__ == "__main__":
    main()
