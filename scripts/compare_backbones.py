#!/usr/bin/env python3
"""Compare ESM-1b vs ESM2-650M zero-shot masked-marginal scores on MMR data.

PROJECT_PLAN.md Phase 3 step 2: "implement true masked-marginal zero-shot
scoring for both ESM-1b and ESM2-650M as simpler baselines ... don't assume
ESM2 wins; two independent papers found ESM-1b better for clinical
pathogenicity specifically. Compare on our MMR data."

Uses :class:`src.mvmamba_features.MaskedMarginalScorer` (a single forward pass
per sequence; PLLR read from one masked-context pass, Meier et al. 2021), so
this is cheap even on CPU relative to full feature extraction or fine-tuning.

Example
-------
    python scripts/compare_backbones.py \
        --mmr_csv data/mmr/processed/extended/extended_dataset.csv \
        --panel_json data/mmr/processed/extended/panel_sequences.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.esm_extractor import get_device  # noqa: E402
from src.eval_utils import bootstrap_ci  # noqa: E402
from src.mvmamba_features import MaskedMarginalScorer  # noqa: E402

logger = logging.getLogger("compare_backbones")

MMR_GENES = ("MLH1", "MSH2", "MSH6", "PMS2")
DEFAULT_BACKBONES = {
    "esm1b": "facebook/esm1b_t33_650M_UR50S",
    "esm2_650m": "facebook/esm2_t33_650M_UR50D",
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mmr_csv", type=Path,
                   default=ROOT / "data/mmr/processed/extended/extended_dataset.csv")
    p.add_argument("--panel_json", type=Path,
                   default=ROOT / "data/mmr/processed/extended/panel_sequences.json")
    p.add_argument("--out_dir", type=Path, default=ROOT / "data/processed/backbone_comparison")
    p.add_argument("--backbones", type=str,
                   default=",".join(f"{k}={v}" for k, v in DEFAULT_BACKBONES.items()),
                   help="Comma-separated name=hf_checkpoint pairs.")
    p.add_argument("--clinical_only", action="store_true", default=True,
                   help="Score only ClinVar/PG-clinical labelled rows (default; "
                        "DMS-only labels are not clinical pathogenicity ground truth).")
    p.add_argument("--n_bootstrap", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    device = get_device()

    master = pd.read_csv(args.mmr_csv, low_memory=False)
    master["label"] = pd.to_numeric(master["label"], errors="coerce")
    df = master[master["gene"].isin(MMR_GENES) & master["label"].notna()].copy()
    if args.clinical_only and "label_source" in df.columns:
        df = df[df["label_source"].isin(["clinvar", "pg_clinical"])]
    if "pms2_homology_excluded" in df.columns:
        df = df[pd.to_numeric(df["pms2_homology_excluded"], errors="coerce").fillna(0) != 1]

    panel = json.loads(Path(args.panel_json).read_text())
    sequence_by_gene = {g.upper(): d["sequence"] for g, d in panel.items()}

    backbones = dict(pair.split("=", 1) for pair in args.backbones.split(","))
    rows = []
    for name, model_name in backbones.items():
        logger.info("=== backbone: %s (%s) ===", name, model_name)
        scorer = MaskedMarginalScorer(model_name, device=device)
        for gene in MMR_GENES:
            sub = df[df["gene"] == gene]
            seq = sequence_by_gene.get(gene)
            if sub.empty or not seq:
                continue
            scores = scorer.score(sub, seq)
            y = sub["label"].astype(int).to_numpy()
            if len(set(y)) < 2:
                logger.warning("%s/%s: single-class labels, skipping.", name, gene)
                continue
            auc_ci = bootstrap_ci(y, scores, metric="roc_auc",
                                  n_bootstrap=args.n_bootstrap, seed=args.seed)
            prauc_ci = bootstrap_ci(y, scores, metric="pr_auc",
                                    n_bootstrap=args.n_bootstrap, seed=args.seed)
            row = {
                "backbone": name, "checkpoint": model_name, "gene": gene,
                "n_variants": int(len(sub)),
                "roc_auc": auc_ci["point"], "roc_auc_ci_low": auc_ci["lower"],
                "roc_auc_ci_high": auc_ci["upper"],
                "pr_auc": prauc_ci["point"], "pr_auc_ci_low": prauc_ci["lower"],
                "pr_auc_ci_high": prauc_ci["upper"],
            }
            rows.append(row)
            logger.info("[%s/%s] n=%d ROC-AUC %.3f [%.3f-%.3f] | PR-AUC %.3f",
                        name, gene, row["n_variants"], row["roc_auc"],
                        row["roc_auc_ci_low"], row["roc_auc_ci_high"], row["pr_auc"])
        del scorer

    results = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "backbone_comparison.csv"
    results.to_csv(out_path, index=False)
    (args.out_dir / "backbone_comparison_meta.json").write_text(json.dumps({
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "backbones": backbones, "clinical_only": args.clinical_only,
    }, indent=2))

    print("\n=================== ESM-1b vs ESM2 (masked-marginal PLLR) ============")
    if len(results):
        print(results.pivot_table(index="gene", columns="backbone", values="roc_auc")
              .to_string(float_format=lambda v: f"{v:.4f}"))
    print("=" * 74)
    print(f"Full results -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
