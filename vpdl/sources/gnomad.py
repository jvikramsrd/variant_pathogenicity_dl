"""gnomAD v4 — allele frequency, ACMG frequency evidence, and gene constraint.

The highest-value feature family on this panel, measured rather than assumed:
v1's largest single result improvement (mean ROC-AUC 0.9229 -> 0.9445, MCC
0.4626 -> 0.6894) came from *un-dropping* these columns after a warm start had
silently discarded them, and the `ablate_gnomad_and_scores` arm was the largest
ablation effect in the grid at -0.0526 AUROC. Allele frequency's increment over
the external prior scores was 0.0211 against a standard error of 0.0052, so it
is not redundant with AlphaMissense despite AlphaMissense having seen population
data.

Two kinds of feature come out of here and they behave very differently under
leave-one-gene-out:

* **Per-variant** (allele frequency, the ACMG frequency flags) — vary within a
  gene, usable.
* **Gene-level constraint** (pLI, o/e missense, missense Z) — CONSTANT within a
  gene, therefore pure gene identity under a leave-one-gene-out split. See
  :func:`vpdl.features.drop_gene_constant`; v1 dropped these for exactly this
  reason and the same guard is enforced here.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from vpdl.sources.base import SourceCapabilities

logger = logging.getLogger(__name__)

__all__ = [
    "GNOMAD_API",
    "GNOMAD_DATASET",
    "ACMG_THRESHOLDS",
    "provides",
    "acmg_frequency_flags",
    "fetch_gene",
    "load",
]

GNOMAD_API = "https://gnomad.broadinstitute.org/api"
GNOMAD_DATASET = "gnomad_r4"

# Generic ACMG/AMP frequency thresholds. DELIBERATELY generic.
#
# The InSiGHT MMR expert panel specifies gene-specific thresholds for
# MLH1/MSH2/MSH6/PMS2 that are stricter than these, and using a genome-wide
# default where a VCEP threshold exists is exactly the miscalibration this
# project set out to study. Supply `thresholds=` from the VCEP specification
# rather than relying on these, and record which was used in the run summary.
ACMG_THRESHOLDS: dict[str, float] = {
    "ba1": 0.05,    # stand-alone benign: >5% in a general population
    "bs1": 0.01,    # strong benign: greater than expected for the disorder
    "pm2": 1e-4,    # moderate pathogenic: absent / extremely low frequency
}

_VARIANT_QUERY = """
query PanelVariants($symbol: String!, $dataset: DatasetId!) {
  gene(gene_symbol: $symbol, reference_genome: GRCh38) {
    gene_id
    symbol
    variants(dataset: $dataset) {
      variant_id
      genome { af }
      exome { af }
      hgvsp
      consequence
    }
  }
}
"""

_CONSTRAINT_QUERY = """
query GeneConstraint($symbol: String!) {
  gene(gene_symbol: $symbol, reference_genome: GRCh38) {
    symbol
    gnomad_constraint { pLI oe_mis oe_mis_upper mis_z }
  }
}
"""


def provides() -> SourceCapabilities:
    return SourceCapabilities(
        name="gnomad",
        supplies_labels=False,
        feature_columns=(
            "feature_gnomad_log10_af",
            "feature_acmg_ba1",
            "feature_acmg_bs1",
            "feature_acmg_pm2",
            "feature_gnomad_pli",
            "feature_gnomad_oe_mis",
            "feature_gnomad_mis_z",
        ),
        licence="ODbL / free for any use with attribution",
        notes=(
            "Per-variant AF and ACMG flags are usable under LOPO; the three "
            "gene-level constraint columns are gene-constant and MUST be "
            "dropped for leave-one-gene-out or they encode gene identity."
        ),
    )


def acmg_frequency_flags(
    allele_frequency: float | np.ndarray,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """BA1 / BS1 / PM2 flags from allele frequency.

    PM2 fires on absent-or-vanishingly-rare, which includes AF that is missing
    entirely — a variant unobserved in gnomAD is the PM2 case, not a missing
    value to impute. That distinction is easy to lose to a `fillna`.
    """
    limits = dict(ACMG_THRESHOLDS) | dict(thresholds or {})
    af = np.asarray(allele_frequency, dtype=float)
    observed = np.isfinite(af)

    return {
        "feature_acmg_ba1": np.where(observed & (af > limits["ba1"]), 1.0, 0.0),
        "feature_acmg_bs1": np.where(observed & (af > limits["bs1"]), 1.0, 0.0),
        # Unobserved counts as rare, which is the whole point of PM2.
        "feature_acmg_pm2": np.where(~observed | (af < limits["pm2"]), 1.0, 0.0),
    }


def _post(query: str, variables: dict, retries: int = 3, pause: float = 2.0) -> dict:
    """POST a GraphQL query, retrying on transient failure.

    The public endpoint rate-limits; a failed fetch must raise rather than
    return empty, because an empty gnomAD join is indistinguishable downstream
    from "this gene has no observed variants" and that is how a whole feature
    family goes missing without anyone noticing.
    """
    import urllib.error
    import urllib.request

    payload = json.dumps({"query": query, "variables": variables}).encode()
    request = urllib.request.Request(
        GNOMAD_API, data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "vpdl/0.1"},
    )

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read())
            if body.get("errors"):
                raise RuntimeError(f"gnomAD API errors: {body['errors']}")
            return body["data"]
        except (urllib.error.URLError, TimeoutError, RuntimeError) as error:
            last_error = error
            logger.warning("gnomAD attempt %d/%d failed: %s",
                           attempt + 1, retries, error)
            time.sleep(pause * (attempt + 1))

    raise RuntimeError(
        f"gnomAD fetch failed after {retries} attempts: {last_error}. "
        "Refusing to continue with an empty join — a silently absent feature "
        "family is worse than a failed build."
    )


def fetch_gene(symbol: str, cache_dir: Path, dataset: str = GNOMAD_DATASET) -> Path:
    """Fetch and cache one gene's variants and constraint."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{symbol}_{dataset}.json"
    if target.exists():
        return target

    logger.info("Fetching gnomAD %s for %s", dataset, symbol)
    variants = _post(_VARIANT_QUERY, {"symbol": symbol, "dataset": dataset})
    constraint = _post(_CONSTRAINT_QUERY, {"symbol": symbol})
    target.write_text(json.dumps(
        {"variants": variants, "constraint": constraint}, indent=1
    ))
    return target


def load(
    cache_dir: Path | str,
    genes: Sequence[str],
    uniprot_by_gene: Mapping[str, str],
    thresholds: Mapping[str, float] | None = None,
    include_constraint: bool = True,
) -> pd.DataFrame:
    """Emit the record schema with gnomAD feature columns.

    Supplies no labels: every row carries ``label = NaN`` and joins onto
    variants other sources contribute. Population frequency is evidence about
    a variant, never supervision.
    """
    from vpdl.sources.clinvar import parse_hgvs_p

    rows: list[dict] = []

    for gene in genes:
        payload = json.loads(Path(fetch_gene(gene, Path(cache_dir))).read_text())
        gene_block = (payload.get("variants") or {}).get("gene") or {}
        constraint = (
            ((payload.get("constraint") or {}).get("gene") or {})
            .get("gnomad_constraint") or {}
        )

        for variant in gene_block.get("variants") or []:
            parsed = parse_hgvs_p(variant.get("hgvsp") or "")
            if parsed is None:
                continue
            wt_aa, position, mut_aa = parsed

            exome = (variant.get("exome") or {}).get("af")
            genome = (variant.get("genome") or {}).get("af")
            frequencies = [f for f in (exome, genome) if f is not None]
            allele_frequency = max(frequencies) if frequencies else np.nan

            record = {
                "uniprot_id": uniprot_by_gene.get(gene),
                "position": position,
                "wt_aa": wt_aa,
                "mut_aa": mut_aa,
                "gene": gene,
                "label": np.nan,
                "label_source": "gnomad",
                "evidence_tier": "population",
                "feature_gnomad_log10_af": (
                    float(np.log10(allele_frequency))
                    if np.isfinite(allele_frequency) and allele_frequency > 0
                    else np.nan
                ),
            }
            flags = acmg_frequency_flags(allele_frequency, thresholds)
            record.update({key: float(value) for key, value in flags.items()})

            if include_constraint:
                # Gene-constant by construction. Emitted because they are real
                # evidence, and dropped for LOPO by drop_gene_constant().
                record["feature_gnomad_pli"] = constraint.get("pLI", np.nan)
                record["feature_gnomad_oe_mis"] = constraint.get("oe_mis", np.nan)
                record["feature_gnomad_mis_z"] = constraint.get("mis_z", np.nan)

            rows.append(record)

    frame = pd.DataFrame(rows)
    logger.info(
        "gnomAD: %d variants over %d genes, %d with observed AF",
        len(frame), len(genes),
        int(frame["feature_gnomad_log10_af"].notna().sum()) if len(frame) else 0,
    )
    return frame
