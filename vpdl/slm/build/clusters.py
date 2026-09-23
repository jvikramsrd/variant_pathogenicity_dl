"""Duplicate, near-duplicate and template clusters over the documents the model will READ.

Clusters are computed on the conclusion-masked text (what a model is given
under the ``classify`` policy), not the raw text: two narratives that differ
only in their verdict sentence are the same input.

    exact_group        identical normalised masked text
    near_dup_cluster   Jaccard >= near_threshold on word 5-shingles, over ALL documents
                       (labs copy text across laboratories too)
    template_cluster   Jaccard >= template_threshold on the placeholder-normalised text,
                       computed within each submitter (a template is a lab's)

Large corpora: MinHash + LSH keeps this near-linear; the cost is dominated by
shingling (docs/slm/GENOMIC_SLM_DGX_RUNBOOK.md gives the command; runtime on
the full ClinVar narrative set is TBD — RUN ON DGX SPARK).
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from vpdl.slm.parallel import pmap
from vpdl.slm.text.conclusion import mask_conclusions
from vpdl.slm.text.dedup import cluster_documents, exact_group, template_text

logger = logging.getLogger(__name__)

__all__ = ["document_clusters"]


def _masked_text(text: str) -> str:
    return mask_conclusions(text).text


def document_clusters(documents: pd.DataFrame, near_threshold: float = 0.8,
                      template_threshold: float = 0.6, genes: Sequence[str] | None = None,
                      num_perm: int = 128, seed: int = 1, workers: int | None = 1) -> pd.DataFrame:
    frame = documents.loc[documents["text"].fillna("").str.len() > 0,
                          ["document_id", "text", "submitter", "gene"]].reset_index(drop=True)
    masked = pd.Series(pmap(_masked_text, frame["text"].tolist(), workers), index=frame.index)
    out = pd.DataFrame({"document_id": frame["document_id"]})
    out["exact_group"] = masked.map(exact_group)
    near = cluster_documents(masked.tolist(), near_threshold, num_perm=num_perm, seed=seed,
                             workers=workers)
    out["near_dup_cluster"] = [f"nd{label}" for label in near.labels]
    logger.info("near-duplicate clusters: %d over %d documents (largest %d)",
                near.n_clusters, len(frame), near.largest)
    # Each document's own gene symbol is replaced by name; other symbols it
    # mentions are caught by template_text's all-caps rule. (Passing every gene
    # of the corpus would run 35,000 substitutions per document.)
    template = np.empty(len(frame), dtype=object)
    own_gene = frame["gene"].fillna("").astype(str).tolist()
    for submitter, rows in frame.groupby("submitter").groups.items():
        rows = np.asarray(list(rows))
        texts = [template_text(masked.iloc[i], [own_gene[i], *(genes or [])]) for i in rows]
        result = cluster_documents(texts, template_threshold, num_perm=num_perm, seed=seed,
                                   workers=workers)
        for i, label in zip(rows, result.labels):
            template[i] = f"tp:{submitter}:{rows[label]}"
    out["template_cluster"] = template
    missing = documents.loc[~documents["document_id"].isin(out["document_id"]), ["document_id"]]
    return pd.concat([out, missing.assign(exact_group=None, near_dup_cluster=None,
                                          template_cluster=None)], ignore_index=True)
