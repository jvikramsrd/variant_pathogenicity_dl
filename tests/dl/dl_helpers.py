"""Synthetic builders shared by the DL-branch tests. No data files, no network.

They live here, not in ``conftest.py``, because tests/dl and tests/slm both have
a ``conftest.py`` and neither directory is a package: whichever loads first owns
the module name ``conftest``, so ``from conftest import PANEL`` in a DL test
picked up tests/slm/conftest.py whenever the whole suite ran together (10 DL
tests failed that way while passing on their own). A unique module name cannot
collide.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
PANEL = {"MLH1": "P40692", "MSH2": "P43246", "MSH6": "P52701", "PMS2": "P54278"}


def random_sequences(length: int = 80, seed: int = 0) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    return {acc: "".join(rng.choice(list(AA), length)) for acc in PANEL.values()}


def assembled_table(sequences: dict[str, str], per_position: int = 2, seed: int = 0,
                    signal: float = 0.8) -> pd.DataFrame:
    """A table shaped like `vpdl build` output: key, label__clinvar, features."""
    rng = np.random.default_rng(seed)
    rows = []
    for gene, acc in PANEL.items():
        sequence = sequences[acc]
        for position in range(1, len(sequence) + 1):
            wt = sequence[position - 1]
            for mut in rng.choice([a for a in AA if a != wt], per_position, replace=False):
                label = float(rng.random() < 0.5)
                rows.append({
                    "uniprot_id": acc, "position": position, "wt_aa": wt, "mut_aa": str(mut),
                    "gene": gene, "label": label, "label_source": "clinvar",
                    "evidence_tier": "criteria provided, multiple submitters, no conflicts",
                    "label__clinvar": label,
                    "feature_alphamissense_score": np.clip(label * 0.4 + rng.normal(0.3, 0.2),
                                                           0, 1),
                    "feature_gnomad_log10_af": rng.normal(-5, 1) - signal * label,
                    "feature_gnomad_observed": 1.0,
                    "feature_acmg_pm2": float(rng.random() < 0.5),
                    "feature_in_domain": float(rng.random() < 0.5),
                })
    return pd.DataFrame(rows)


def clinvar_record(gene, position, wt, mut, label, stars=2, variation="1", assembly="GRCh38",
                   name=None, chrom="3", pos_vcf="100", ref="C", alt="T"):
    from vpdl.sources.clinvar import _review_stars  # noqa: F401  (shared helper, untouched)

    status = {0: "no assertion criteria provided", 1: "criteria provided, single submitter",
              2: "criteria provided, multiple submitters, no conflicts",
              3: "reviewed by expert panel", 4: "practice guideline"}[stars]
    return {"gene": gene, "position": position, "wt_aa": wt, "mut_aa": mut,
            "name": name or f"NM_000249.4({gene}):c.{3 * position - 2}C>T (p.Xaa{position}Yaa)",
            "significance": "Pathogenic" if label == 1 else "Benign", "label": float(label),
            "review_status": status, "stars": stars, "variation_id": variation,
            "assembly": assembly, "chromosome": chrom, "position_vcf": pos_vcf,
            "ref_vcf": ref, "alt_vcf": alt}
