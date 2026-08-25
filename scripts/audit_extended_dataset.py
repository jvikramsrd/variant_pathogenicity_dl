#!/usr/bin/env python3
"""Independent integrity audit of data/processed/extended/extended_dataset.csv.

Re-derives every join invariant from first principles (canonical sequences,
per-source raw columns, label precedence) instead of trusting the builder.
Writes a machine-readable report to extended/audit_report.json and prints a
human summary.  Optionally emits a train-ready CSV of resolved-label rows.

Checks
------
C1  key uniqueness / completeness      no dup or NaN (uniprot_id,position,wt_aa,mut_aa)
C2  sequence consistency               wt_aa == canonical[position], in bounds
C3  mutation validity                  mut_aa != wt_aa, valid residue letters
C4  gene<->accession bijection         matches panel records exactly
C5  hgvs_p consistency                 p.{three(wt)}{pos}{three(mut)}
C6  label precedence                   label == f(clinvar, clinical, single-assay DMS)
C7  DMS aggregate coherence            n_assays>0 <=> score present <=> ids present
C8  AlphaMissense sanity               score in [0,1]; class vocabulary known
C9  zero-shot coverage                 per-model fill rates; rows with scores
C10 domain coherence                   names non-empty <=> in_domain==1
C11 provenance coherence               sources tokens == observed evidence; n_sources
C12 conflict quarantine                contradictory clinical evidence has no target

Exit code 0 iff every hard check passes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXT_DIR = PROJECT_ROOT / "data" / "processed" / "extended"
PANEL_FILE = PROJECT_ROOT / "data" / "raw" / "uniprot" / "expanded_panel.json"

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
THREE = {"A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
         "Q": "Gln", "E": "Glu", "G": "Gly", "H": "His", "I": "Ile",
         "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
         "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val"}
AM_CLASSES = {"benign", "likely benign", "ambiguous", "likely pathogenic",
              "pathogenic"}


def main() -> int:
    df = pd.read_csv(EXT_DIR / "extended_dataset.csv",
                     dtype={"wt_aa": str, "mut_aa": str, "gene": str,
                            "uniprot_id": str, "hgvs_p": str})
    panel = json.loads(PANEL_FILE.read_text())
    seq_of = {v["accession"]: v["sequence"] for v in panel.values()}
    acc_of = {g.upper(): v["accession"] for g, v in panel.items()}
    report: dict = {"rows": len(df), "cols": df.shape[1], "checks": {},
                    "train_summary": {}, "conflicts": {}}
    failures: list[str] = []

    key = ["uniprot_id", "position", "wt_aa", "mut_aa"]

    # C1 ------------------------------------------------------------------ #
    n_dup = int(df.duplicated(subset=key).sum())
    n_nan_key = int(df[key].isna().any(axis=1).sum())
    report["checks"]["C1_key_uniqueness"] = {
        "duplicate_keys": n_dup, "nan_keys": n_nan_key}
    if n_dup or n_nan_key:
        failures.append("C1")

    # C2+C3: sequence-level validation ------------------------------------- #
    sub = df[df["uniprot_id"].isin(seq_of)]
    lens = sub["uniprot_id"].map({a: len(s) for a, s in seq_of.items()})
    chars = sub["uniprot_id"].map(
        {a: np.array(list(s)) for a, s in seq_of.items()})
    pos_ok = (sub["position"] >= 1) & (sub["position"] <= lens)
    wt_ok = np.array([
        c[p - 1] == w if 1 <= p <= len(c) else False
        for c, p, w in zip(chars, sub["position"], sub["wt_aa"])], dtype=bool)
    bad_pos = int((~pos_ok).sum())
    bad_wt = int((~wt_ok).sum())
    bad_mut_letter = int((~df["mut_aa"].isin(VALID_AA)).sum())
    same_aa = int((df["wt_aa"] == df["mut_aa"]).sum())
    unknown_acc = int((~df["uniprot_id"].isin(seq_of)).sum())
    report["checks"]["C2_sequence_consistency"] = {
        "out_of_bounds": bad_pos, "wt_mismatch": bad_wt,
        "unknown_accessions": unknown_acc}
    report["checks"]["C3_mutation_validity"] = {
        "invalid_mut_letters": bad_mut_letter, "wt_eq_mut": same_aa}
    if bad_pos or bad_wt or unknown_acc:
        failures.append("C2")
    if bad_mut_letter or same_aa:
        failures.append("C3")

    # C4 ------------------------------------------------------------------- #
    pairs = df[["gene", "uniprot_id"]].drop_duplicates()
    mismatched = int(sum(acc_of.get(str(g)) != u for g, u in
                         pairs.itertuples(index=False)))
    multi_gene_per_acc = int((pairs.groupby("uniprot_id")["gene"]
                              .nunique() > 1).sum())
    report["checks"]["C4_gene_accession_map"] = {
        "mismatched_pairs": mismatched, "acc_with_multiple_genes":
            multi_gene_per_acc}
    if mismatched or multi_gene_per_acc:
        failures.append("C4")

    # C5 ------------------------------------------------------------------- #
    expected_hgvs = ("p." + df["wt_aa"].map(THREE)
                     + df["position"].astype("Int64").astype(str)
                     + df["mut_aa"].map(THREE))
    has_hgvs = df["hgvs_p"].notna()
    bad_hgvs = int((has_hgvs & (df["hgvs_p"] != expected_hgvs)).sum())
    missing_hgvs = int((~has_hgvs).sum())
    report["checks"]["C5_hgvs"] = {"mismatched": bad_hgvs,
                                   "missing": missing_hgvs}
    if bad_hgvs:
        failures.append("C5")
    if missing_hgvs:
        report["checks"]["C5_hgvs"]["note"] = "missing values backfilled by builder"

    # C6: label precedence -------------------------------------------------- #
    # ProteinGym DMS_score_bin=1 == top fitness half (tolerated), so the
    # DMS-derived pathogenic label is the flip; single-assay rows only.
    dms_single_ok = ((df["dms_bin_median"].notna()) & (df["n_dms_assays"] == 1)
                     & df["dms_bin_median"].isin([0, 1]))
    dms_pathogenic = 1 - df["dms_bin_median"]
    expected_label = np.where(
        df["clinvar_label"].notna(), df["clinvar_label"],
        np.where(df["clinical_label"].notna(), df["clinical_label"],
                 np.where(dms_single_ok, dms_pathogenic, np.nan)))
    conflict_col = df.get("label_conflict", pd.Series(0, index=df.index)).fillna(0).astype(int)
    expected_label = np.where(conflict_col == 1, np.nan, expected_label)
    got = pd.to_numeric(df["label"], errors="coerce").to_numpy(dtype="float64")
    exp = pd.to_numeric(pd.Series(expected_label), errors="coerce") \
        .to_numpy(dtype="float64")
    both_nan = np.isnan(got) & np.isnan(exp)
    precedence_bad = int(((got != exp) & ~both_nan).sum())
    invalid_labels = int((~pd.isna(got) & ~np.isin(got, [0.0, 1.0])).sum())
    report["checks"]["C6_label_precedence"] = {
        "violations": precedence_bad, "non_binary": invalid_labels}
    if precedence_bad or invalid_labels:
        failures.append("C6")

    # C7 -------------------------------------------------------------------- #
    has_dms = df["n_dms_assays"].fillna(0) > 0
    coh = ((has_dms == df["dms_score_median"].notna())
           & (has_dms == df["dms_ids"].notna())).all()
    multi_no_bin = int((has_dms & (df["n_dms_assays"] > 1)
                        & df["dms_bin_median"].isna()).sum())
    report["checks"]["C7_dms_coherence"] = {
        "coherent": bool(coh),
        "multi_assay_without_bin(median-masked)": multi_no_bin}
    if not coh:
        failures.append("C7")

    # C8 -------------------------------------------------------------------- #
    am = df["am_pathogenicity"]
    am_bad_range = int((am.notna() & ((am < 0) | (am > 1))).sum())
    bad_class = int((df["am_class"].notna()
                     & ~df["am_class"].isin(AM_CLASSES)).sum())
    class_score_conflict = int((am.notna() & df["am_class"].notna()
                                & (((am < 0.5) & (df["am_class"].isin(
                                    {"pathogenic", "likely pathogenic"})))
                                   | ((am > 0.5) & (df["am_class"].isin(
                                       {"benign", "likely benign"}))))).sum())
    report["checks"]["C8_alphamissense"] = {
        "score_out_of_range": am_bad_range, "unknown_class": bad_class,
        "class_vs_score_contradictions": class_score_conflict}
    if am_bad_range or bad_class:
        failures.append("C8")

    # C9 -------------------------------------------------------------------- #
    zs_cols = [c for c in df.columns if c.startswith("zs_")]
    fill = {c: round(float(df[c].notna().mean()), 4) for c in zs_cols}
    any_zs = float(df[zs_cols].notna().any(axis=1).mean())
    report["checks"]["C9_zeroshot_coverage"] = {
        "any_model_row_fraction": round(any_zs, 4), "per_model_fill": fill}

    # C10 ------------------------------------------------------------------- #
    dom_incoh = int(((df["in_domain"] == 1) != df["domain_names"].notna()
                     & (df["domain_names"] != "")).sum())
    # empty-string vs NaN: normalise then compare
    dn = df["domain_names"].fillna("").astype(str)
    dom_incoh = int((((df["in_domain"] == 1) & (dn == ""))
                     | ((df["in_domain"] == 0) & (dn != ""))).sum())
    report["checks"]["C10_domains"] = {"incoherent_rows": dom_incoh}
    if dom_incoh:
        failures.append("C10")

    # C11 ------------------------------------------------------------------- #
    def _expected_sources(r) -> str:
        tags = []
        if pd.notna(r["clinvar_label"]) or pd.notna(r.get("review_status")):
            tags.append("clinvar")
        if pd.notna(r["clinical_label"]):
            tags.append("pg_clinical")
        if pd.notna(r["n_dms_assays"]) and r["n_dms_assays"] > 0:
            tags.append("dms")
        if pd.notna(r["am_pathogenicity"]):
            tags.append("alphamissense")
        return "|".join(tags)

    src_tokens = df["sources"].fillna("").astype(str)
    n_src_bad = 0
    clin_mask = (df["clinvar_label"].notna()
                 | df["review_status"].notna().fillna(False))
    pg_mask = df["clinical_label"].notna()
    dms_mask = has_dms
    am_mask = df["am_pathogenicity"].notna()
    n_src_bad = int(
        (src_tokens.str.contains("clinvar") != clin_mask).sum()
        + (src_tokens.str.contains("pg_clinical") != pg_mask).sum()
        + (src_tokens.str.contains("dms") != dms_mask).sum()
        + (src_tokens.str.contains("alphamissense") != am_mask).sum())
    token_counts = src_tokens.apply(lambda s: len([t for t in s.split("|") if t]))
    n_count_bad = int((token_counts != df["n_sources"]).sum())
    report["checks"]["C11_provenance"] = {"token_mismatches": n_src_bad,
                                          "count_mismatches": n_count_bad}
    if n_src_bad or n_count_bad:
        failures.append("C11")

    # C12 conflicts ---------------------------------------------------------- #
    both_lab = df["clinvar_label"].notna() & df["clinical_label"].notna()
    disagree = int((both_lab & (df["clinvar_label"] != df["clinical_label"])).sum())
    quarantined = int(((conflict_col == 1) & pd.isna(df["label"])).sum())
    unquarantined = int((both_lab & (df["clinvar_label"] != df["clinical_label"])
                         & df["label"].notna()).sum())
    report["conflicts"]["clinvar_vs_pg_clinical"] = {
        "overlap_rows": int(both_lab.sum()), "disagreements": disagree,
        "quarantined_rows": quarantined,
        "unquarantined_disagreements": unquarantined,
        "note": "contradictory clinical evidence is excluded from supervision"}
    if unquarantined:
        failures.append("C12")

    # Trainability summary --------------------------------------------------- #
    lab = df[pd.notna(got) & (conflict_col == 0)]
    by_source = {
        "clinvar_only": int((lab["clinvar_label"].notna()
                             & lab["clinical_label"].isna()).sum()),
        "pg_clinical_only": int((lab["clinvar_label"].isna()
                                 & lab["clinical_label"].notna()).sum()),
        "single_assay_dms": int((lab["clinvar_label"].isna()
                                 & lab["clinical_label"].isna()).sum()),
    }
    per_gene = (lab.groupby(["gene"])["label"]
                .agg(total="size", pathogenic=lambda s: int((s == 1).sum()),
                     benign=lambda s: int((s == 0).sum()))
                .sort_values("total", ascending=False))
    feature_cov = {
        "alphamissense": round(float(lab["am_pathogenicity"].notna().mean()), 3),
        "any_zeroshot": round(float(lab[zs_cols].notna().any(axis=1).mean()), 3),
        "dms_score": round(float(lab["dms_score_median"].notna().mean()), 3),
        "in_domain": round(float((lab["in_domain"] == 1).mean()), 3),
    }
    report["train_summary"] = {
        "labelled_rows": len(lab),
        "pathogenic": int((lab["label"] == 1).sum()),
        "benign": int((lab["label"] == 0).sum()),
        "labels_by_source": by_source,
        "feature_coverage_on_labelled": feature_cov,
        "genes": int(lab["gene"].nunique()),
        "per_gene": per_gene.reset_index().to_dict("records"),
    }

    report["passed"] = not failures

    # Emit the train-ready CSV from the verified labelled rows so that the
    # whole artefact chain is reproducible from code alone.
    if not failures:
        lab_df = df[pd.notna(pd.to_numeric(df["label"], errors="coerce"))
                    & (conflict_col == 0)].copy()
        lab_df["label"] = pd.to_numeric(lab_df["label"], errors="coerce") \
            .astype("Int64")
        zs_cols = [c for c in lab_df.columns if c.startswith("zs_")]
        order = ["gene", "uniprot_id", "position", "wt_aa", "mut_aa", "hgvs_p",
                 "label", "label_source", "label_weight", "label_conflict",
                 "cross_source_conflict", "clinvar_label", "clinvar_conflict",
                 "stars", "review_status", "clinical_label", "clinical_conflict", "np_accession",
                 "dms_score_median", "dms_bin_median", "dms_bin_nunique", "n_dms_assays", "dms_ids",
                 "dms_selection_types", "am_pathogenicity", "am_class",
                 "in_domain", "domain_names"]
        order = [c for c in order if c in lab_df.columns] \
            + zs_cols + ["sources", "n_sources"]
        lab_df = lab_df.sort_values(["gene", "position", "mut_aa"])[order] \
            .reset_index(drop=True)
        train_path = EXT_DIR / "extended_dataset_train.csv"
        lab_df.to_csv(train_path, index=False)
        report["train_csv"] = {
            "path": str(train_path), "rows": len(lab_df),
            "pathogenic": int((lab_df["label"] == 1).sum()),
            "benign": int((lab_df["label"] == 0).sum()),
        }

    out = EXT_DIR / "audit_report.json"
    out.write_text(json.dumps(report, indent=2))

    print(json.dumps({k: v for k, v in report.items()
                      if k in ("rows", "cols", "passed", "checks",
                               "conflicts", "train_summary")},
                     indent=2)[:4000])
    print(f"\nAudit report -> {out}")
    if failures:
        print("FAILED CHECKS:", ", ".join(failures))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
