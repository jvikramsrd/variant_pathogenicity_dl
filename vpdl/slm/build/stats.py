"""Corpus quantification: how much clinical text there is, of what kind, about what.

Every number the docs quote about the clinical corpus comes from here, run on
the real tables (``vpdl-slm stats``). Until that has run on the DGX the docs
say ``TBD — RUN ON DGX SPARK``; the corpus is not called sufficient before it
has been measured.

Lengths are reported in characters and whitespace words; ``tokenizer=`` adds
model tokens for one tokenizer (token counts depend on it — BiomedBERT's
WordPiece and our 32k BPE split "c.199G>A" differently).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from vpdl.slm.variants import MMR_GENES

__all__ = ["corpus_stats", "stats_markdown"]


def _lengths(values: pd.Series) -> dict[str, float]:
    if values.empty:
        return {"n": 0}
    array = values.to_numpy()
    return {"n": int(len(array)), "total": int(array.sum()), "median": float(np.median(array)),
            "p25": float(np.percentile(array, 25)), "p75": float(np.percentile(array, 75)),
            "max": int(array.max()), "mean": round(float(array.mean()), 1)}


def _breakdown(frame: pd.DataFrame, column: str, top: int) -> list[dict[str, Any]]:
    if column not in frame.columns or frame.empty:
        return []
    grouped = frame.groupby(column, dropna=False).agg(documents=("document_id", "size"),
                                                       characters=("chars", "sum"),
                                                       variants=("variant_id", "nunique"))
    grouped = grouped.sort_values("documents", ascending=False).head(top).reset_index()
    return [{column: (str(r[column]) if pd.notna(r[column]) else "missing"),
             "documents": int(r["documents"]), "characters": int(r["characters"]),
             "variants": int(r["variants"])} for _, r in grouped.iterrows()]


def corpus_stats(tables: Mapping[str, pd.DataFrame], top: int = 30,
                 tokenizer: Callable[[str], list] | None = None,
                 token_sample: int = 20_000, seed: int = 0) -> dict[str, Any]:
    documents = tables["documents"].copy()
    variants = tables["variants"]
    units = tables.get("evidence_units", pd.DataFrame())
    citations = tables.get("citations", pd.DataFrame())
    documents["chars"] = documents["text"].str.len().fillna(0).astype(int)
    documents["words"] = documents["text"].str.split().str.len().fillna(0).astype(int)
    documents["has_text"] = documents["chars"] > 0
    documents["year"] = documents["date"].fillna("").str[:4].replace("", "undated")
    documents = documents.merge(variants[["variant_id", "consequence", "variant_type"]],
                                on="variant_id", how="left")
    documents["disease"] = documents["disease_names"].map(
        lambda names: names[0] if len(names) else "none given")
    documents["mmr"] = documents["gene"].isin(MMR_GENES)

    flags = pd.DataFrame(index=documents["document_id"])
    if not units.empty:
        # Row-level flags, then one vectorised groupby: per-group Python lambdas
        # take hours over the full corpus's millions of units.
        role = units["sentence_role"]
        rows = pd.DataFrame({"document_id": units["document_id"],
                             "conclusion": role == "conclusion",
                             "external": role == "external_classification",
                             "codes": units["acmg_codes"].map(len) > 0,
                             "evidence_units": (role == "evidence").astype(int)})
        per_doc = rows.groupby("document_id").agg(
            conclusion=("conclusion", "any"), external=("external", "any"),
            codes=("codes", "any"), evidence_units=("evidence_units", "sum"))
        types = units[["document_id", "evidence_types"]].explode("evidence_types")
        for kind in ("phenotype", "functional", "population", "segregation", "case", "computational",
                     "splicing", "de_novo", "allelic"):
            hits = types.loc[types["evidence_types"] == kind, "document_id"].unique()
            per_doc[f"has_{kind}"] = per_doc.index.isin(hits)
        flags = per_doc
    documents = documents.merge(flags, left_on="document_id", right_index=True, how="left")
    with_text = documents[documents["has_text"]]

    def count(mask) -> int:
        return int(mask.fillna(False).astype(bool).sum())

    labels = documents["label5"].fillna(documents["label_category"])
    result: dict[str, Any] = {
        "documents": int(len(documents)),
        "documents_with_text": int(documents["has_text"].sum()),
        "clinical_records": count(documents["source_id"].isin(["clinvar_submission_summary", "clingen_erepo"])),
        "unique_variants_with_documents": int(documents["variant_id"].nunique()),
        "unique_variants_with_text": int(with_text["variant_id"].nunique()),
        "variants_in_table": int(len(variants)),
        "unique_genes": int(documents["gene"].replace("", np.nan).nunique()),
        "unique_diseases": int(len({d for names in documents["disease_names"] for d in names})),
        "laboratories_submitters": int(documents["submitter"].nunique()),
        "expert_panel_records": count(documents["tier"] == 1),
        "vcep_records": count(documents["source_id"] == "clingen_erepo")
        + count(documents["submitter"].str.contains("expert panel|vcep", case=False, na=False)
                & (documents["source_id"] != "clingen_erepo")),
        "restricted_documents": count(documents["restricted"]),
        "documents_by_label": labels.value_counts().to_dict(),
        "documents_with_direct_conclusion": count(documents.get("conclusion", pd.Series(dtype=bool))),
        "documents_with_external_classification": count(documents.get("external", pd.Series(dtype=bool))),
        "documents_with_acmg_codes": count(documents.get("codes", pd.Series(dtype=bool))),
        "documents_with_pmids_in_text": count(documents["text_pmids"].map(len) > 0),
        "variants_with_citations": int(citations["variation_id"].nunique()) if not citations.empty else 0,
        "distinct_cited_publications": int(citations.drop_duplicates(
            ["citation_source", "citation_id"]).shape[0]) if not citations.empty else 0,
        "documents_with_evidence_type": {
            kind: count(documents.get(f"has_{kind}", pd.Series(dtype=bool)))
            for kind in ("phenotype", "functional", "population", "segregation", "case",
                         "computational", "splicing", "de_novo", "allelic")},
        "characters": _lengths(with_text["chars"]),
        "words": _lengths(with_text["words"]),
        "evidence_units": int(len(units)),
        "evidence_units_by_role": units["sentence_role"].value_counts().to_dict() if not units.empty else {},
        "tier_counts": documents["tier"].value_counts().sort_index().to_dict(),
        "mmr_share": {
            "documents": round(float(documents["mmr"].mean()), 4) if len(documents) else None,
            "characters": round(float(with_text.loc[with_text["mmr"], "chars"].sum()
                                      / max(1, with_text["chars"].sum())), 4)},
        "breakdowns": {name: _breakdown(documents, column, top) for name, column in [
            ("gene", "gene"), ("disease", "disease"), ("variant_type", "consequence"),
            ("source", "source_id"), ("laboratory", "submitter"), ("year", "year"),
            ("review_status", "review_status"), ("classification", "label5"), ("tier", "tier")]},
        "variant_consequences_in_table": variants["consequence"].value_counts().to_dict(),
    }
    if tokenizer is not None and len(with_text):
        sample = with_text.sample(min(token_sample, len(with_text)), random_state=seed)
        tokens = sample["text"].map(lambda t: len(tokenizer(t)))
        ratio = tokens.sum() / max(1, sample["chars"].sum())
        result["tokens"] = {"sampled_documents": int(len(sample)),
                            "per_document": _lengths(tokens),
                            "tokens_per_character": round(float(ratio), 4),
                            "estimated_total": int(ratio * with_text["chars"].sum())}
    else:
        result["tokens"] = "not computed (pass a tokenizer)"
    return result


def stats_markdown(stats: Mapping[str, Any]) -> str:
    lines = ["# Clinical text corpus — measured", "",
             f"- documents: {stats['documents']:,} ({stats['documents_with_text']:,} with text)",
             f"- unique variants with text: {stats['unique_variants_with_text']:,}; genes: "
             f"{stats['unique_genes']:,}; diseases: {stats['unique_diseases']:,}; submitters: "
             f"{stats['laboratories_submitters']:,}",
             f"- expert-panel records: {stats['expert_panel_records']:,}; with ACMG codes: "
             f"{stats['documents_with_acmg_codes']:,}; with a direct conclusion: "
             f"{stats['documents_with_direct_conclusion']:,}",
             f"- characters: {json.dumps(stats['characters'])}",
             f"- tokens: {json.dumps(stats['tokens'])}",
             f"- MMR share: {json.dumps(stats['mmr_share'])}",
             f"- labels: {json.dumps(stats['documents_by_label'])}",
             f"- evidence types (documents): {json.dumps(stats['documents_with_evidence_type'])}", ""]
    for name, rows in stats["breakdowns"].items():
        lines += [f"## by {name}", "", "| value | documents | variants | characters |", "|---|---|---|---|"]
        key = next(iter(rows[0])) if rows else name
        lines += [f"| {r[key]} | {r['documents']:,} | {r['variants']:,} | {r['characters']:,} |"
                  for r in rows]
        lines.append("")
    return "\n".join(lines)
