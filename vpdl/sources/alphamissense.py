"""AlphaMissense pathogenicity priors — feature source and orientation anchor.

Licence — the downloaded file and the upstream repository DISAGREE, and a
reviewer who opens the file will see the older one:

* The predictions were released in 2023 under **CC BY-NC-SA 4.0**. The header
  of ``AlphaMissense_aa_substitutions.tsv.gz`` still says so ("Copyright 2023").
* On **2024-03-13** DeepMind relicensed them to **CC BY 4.0**
  (google-deepmind/alphamissense commit fe2dc845, "Update AlphaMissense
  predictions database license"). The data file was not regenerated, so its
  header is stale.

Cite as CC BY 4.0 and state the relicensing with the commit, so the
contradiction a reader will find in the file is explained rather than looking
like an error. v1's manuscript said CC BY-NC-SA 4.0, which was accurate for the
file it downloaded; it was superseded, not wrong.

This source does double duty: besides supplying a feature, it is the
independent anchor :func:`vpdl.assemble.assert_label_orientation` uses to catch
an inverted label mapping in some *other* source. That is why it is worth
including even in runs where its feature is ablated away.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from vpdl.sources.base import SourceCapabilities

logger = logging.getLogger(__name__)

__all__ = ["ALPHAMISSENSE_URL", "provides", "load"]

ALPHAMISSENSE_URL = (
    "https://storage.googleapis.com/dm_alphamissense/"
    "AlphaMissense_aa_substitutions.tsv.gz"
)


def provides() -> SourceCapabilities:
    return SourceCapabilities(
        name="alphamissense",
        supplies_labels=False,
        feature_columns=("feature_alphamissense_score",),
        licence="CC BY 4.0 (relicensed 2024-03-13; file header still says CC BY-NC-SA 4.0)",
        notes=(
            "Trained on population and clinical data, so it carries allele-"
            "frequency signal: an ablation dropping gnomAD while this remains "
            "is uninterpretable (see vpdl.features.PROXY_FOR)."
        ),
    )


def load(
    path: Path | str,
    accessions: Sequence[str],
    gene_by_accession: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Stream the substitutions file, keeping only the panel's accessions.

    The full file is ~1.2 GB compressed and covers every human missense
    substitution, so it is filtered during the read rather than loaded and
    subset afterwards.
    """
    wanted = set(accessions)
    gene_by_accession = dict(gene_by_accession or {})
    rows: list[dict] = []

    # Every data line begins with its UniProt accession, so a prefix test rejects
    # the ~200M off-panel lines before any splitting or dict construction.
    prefixes = tuple(f"{accession}\t" for accession in wanted)

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        header: list[str] | None = None
        for line in handle:
            if line.startswith("#"):
                continue
            if header is None:
                header = line.rstrip("\n").split("\t")
                continue
            if not line.startswith(prefixes):
                continue
            fields = line.rstrip("\n").split("\t")
            record = dict(zip(header, fields))
            accession = record.get("uniprot_id")
            if accession not in wanted:
                continue
            variant = record.get("protein_variant", "")
            if len(variant) < 3:
                continue
            wt_aa, mut_aa = variant[0], variant[-1]
            try:
                position = int(variant[1:-1])
            except ValueError:
                continue
            rows.append({
                "uniprot_id": accession,
                "position": position,
                "wt_aa": wt_aa,
                "mut_aa": mut_aa,
                "gene": gene_by_accession.get(accession, ""),
                "label": float("nan"),
                "label_source": "alphamissense",
                "evidence_tier": "prior",
                "feature_alphamissense_score": float(
                    record.get("am_pathogenicity", "nan")
                ),
            })

    frame = pd.DataFrame(rows)
    logger.info("AlphaMissense: %d rows over %d accessions", len(frame), len(wanted))
    return frame
