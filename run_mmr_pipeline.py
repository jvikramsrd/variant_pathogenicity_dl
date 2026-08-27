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
        help="Explicit verified protein-coordinate span of the homology "
             "region; overrides --exclude_pms2.")
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
        "print('cuda' if torch.cuda.is_available() else "
        "('mps' if getattr(torch.backends,'mps',None) and "
        "torch.backends.mps.is_available() else 'cpu'))"
    )
    try:
        out = subprocess.run([py, "-c", probe], capture_output=True,
                             text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if out.returncode != 0:
        return "unknown"
    device = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    return device if device in ("cuda", "mps", "cpu") else "unknown"


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
    device = "unknown"
    if args.full_finetune:
        device = detect_accelerator(py)
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

    stage(f"train, stage 1: pretrain the head on the broad panel "
          f"({args.features} mode, {args.pretrain_mode}) -- frozen embeddings")
    pretrain_args = [
        py, "scripts/pretrain_esm_80.py",
        "--features", args.features, "--mode", args.pretrain_mode,
        "--checkpoint_name", PRETRAIN_CHECKPOINT_NAME,
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
        ]
        if args.gradient_checkpointing:
            ft_args.append("--gradient_checkpointing")
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
