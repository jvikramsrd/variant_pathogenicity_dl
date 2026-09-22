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
    "UNOBSERVED_LOG10_AF",
    "fill_unobserved",
    "provides",
    "acmg_frequency_flags",
    "fetch_gene",
    "load",
]

GNOMAD_API = "https://gnomad.broadinstitute.org/api"
GNOMAD_DATASET = "gnomad_r4"

# Frequency thresholds, taken from v1 (src/gnomad.py), which chose them for an
# autosomal-dominant, early-onset cancer syndrome. An earlier draft of this
# module used generic values (BS1 1e-2, PM2 1e-4) — ten times too permissive on
# both flags for Lynch syndrome. The InSiGHT MMR expert panel publishes
# gene-specific values; confirm against the VCEP specification before a
# headline run and pass them via `thresholds=` if they differ.
ACMG_THRESHOLDS: dict[str, float] = {
    "ba1": 0.05,    # stand-alone benign
    "bs1": 1e-3,    # strong benign; AD, early onset -> conservative
    "pm2": 1e-5,    # moderate pathogenic; absent or vanishingly rare
}

_VARIANT_QUERY = """
query PanelVariants($symbol: String!, $dataset: DatasetId!) {
  gene(gene_symbol: $symbol, reference_genome: GRCh38) {
    gene_id
    symbol
    variants(dataset: $dataset) {
      variant_id
      consequence
      hgvsp
      transcript_id
      genome { af ac an }
      exome  { af ac an }
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


# log10 allele frequency assigned to variants gnomAD never observed. Below the
# rarest frequency the v4 joint call set can report (~1 in 1.6M alleles, about
# -6.2), so "absent" sorts as rarer than anything seen rather than being
# imputed to a typical observed frequency.
UNOBSERVED_LOG10_AF = -7.0


def fill_unobserved(table: pd.DataFrame) -> pd.DataFrame:
    """Give variants gnomAD never returned the evidence their absence implies.

    gnomAD only reports observed variants, so after assembly most substitutions
    have no gnomAD row and every gnomAD column is NaN. Left alone, median
    imputation would make an unobserved variant look like a typical observed
    one — erasing precisely the signal PM2 encodes. Absent means: not BA1, not
    BS1, PM2, and rarer than any observed allele. An explicit
    ``feature_gnomad_observed`` indicator keeps "absent" separable from
    "observed at a very low frequency".
    """
    if "feature_acmg_pm2" not in table.columns:
        return table

    table = table.copy()
    # The gnomAD source sets every flag on every row it emits, so a missing flag
    # after the merge means gnomAD returned nothing for that variant.
    unobserved = table["feature_acmg_pm2"].isna()
    table["feature_gnomad_observed"] = (~unobserved).astype(float)
    table.loc[unobserved, "feature_acmg_ba1"] = 0.0
    table.loc[unobserved, "feature_acmg_bs1"] = 0.0
    table.loc[unobserved, "feature_acmg_pm2"] = 1.0
    table["feature_gnomad_log10_af"] = table["feature_gnomad_log10_af"].fillna(
        UNOBSERVED_LOG10_AF
    )
    logger.info("gnomAD: %d of %d variants unobserved -> PM2, AF floor %.1f",
                int(unobserved.sum()), len(table), UNOBSERVED_LOG10_AF)
    return table


def provides() -> SourceCapabilities:
    return SourceCapabilities(
        name="gnomad",
        supplies_labels=False,
        feature_columns=(
            "feature_gnomad_log10_af",
            "feature_gnomad_observed",
            "feature_acmg_ba1",
            "feature_acmg_bs1",
            "feature_acmg_pm2",
            "feature_gnomad_pli",
            "feature_gnomad_oe_mis",
            "feature_gnomad_mis_z",
        ),
        licence="CC0 1.0 (gnomad.broadinstitute.org/policies, checked 2026-09-21; attribution requested)",
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

            # Joint allele frequency: sum AC and AN across the exome and genome
            # call sets, as v1 did. Taking max(af_exome, af_genome) instead
            # overstates frequency whenever the two disagree, which moves
            # variants across the BS1/PM2 lines.
            parts = [p for p in (variant.get("exome"), variant.get("genome")) if p]
            allele_count = sum(float(p.get("ac") or 0) for p in parts)
            allele_number = sum(float(p.get("an") or 0) for p in parts)
            allele_frequency = (
                allele_count / allele_number if allele_number > 0 else np.nan
            )

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
