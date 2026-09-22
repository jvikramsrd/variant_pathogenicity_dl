"""The canonical variant table: one row per protein substitution, with QC.

Built ON TOP of :func:`vpdl.assemble.assemble`, not beside it. Assembly already
does coordinate validation, label precedence, cross-source conflict quarantine,
gnomAD-absence-as-PM2 and the orientation check; this layer adds what the DL
branch needs and assembly does not carry:

* identity — ``variant_id``, normalised HGVS p., transcript, c., genomic
  coordinates (GRCh38) from every ClinVar record behind the protein change;
* order-independent ClinVar labels — every record is considered, discordant
  records withhold the label (``vpdl.sources.clinvar.load`` keeps the first by
  file order; see CODEBASE_AUDIT finding 2);
* explicit QC — ``qc_status``, ``qc_flags``, ``exclusion_reason`` per row, and
  the PMS2CL homology rule with orthogonal confirmation;
* validation-only functional values, structure availability, homology clusters
  and split assignments.

Rows are never dropped for QC. A failed check withholds supervision (sets the
row's ``label__*`` to NaN) and says why, exactly as the PMS2 gate does.

`position` keeps its v2 meaning — PROTEIN position — so every existing layer
reads this table unchanged; the genomic coordinate is ``genomic_position``.
The full protein sequences live in a sidecar (``<table>.sequences.json``); rows
carry ``protein_sequence_sha256``. Repeating a 1,360-residue string on 74k rows
would add ~75 MB and carry no information the sidecar does not.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from vpdl.assemble import label_sources_in, resolve_labels, LABEL_PRECEDENCE
from vpdl.dl import hgvs
from vpdl.dl.homology import PROTEIN_FAMILIES, family_of, homology_groups, sequence_sha
from vpdl.sources.base import validate_against_sequence
from vpdl.sources.clinvar import PMS2_HOMOLOGY_CODONS
from vpdl.splits import group_keys, variant_keys

logger = logging.getLogger(__name__)

__all__ = [
    "CANONICAL_SCHEMA",
    "MISSING_VALUE_POLICY",
    "WITHHOLDING_FLAGS",
    "clinvar_protein_labels",
    "gnomad_protein_frequencies",
    "load_confirmations",
    "load_expert_labels",
    "CanonicalReport",
    "build_canonical",
    "validate_canonical",
    "attach_sequences",
    "write_canonical",
]

# Column -> meaning. The order is the order written. Every column is present
# in every build; a column a build had no source for is NaN, and the report
# says which sources were supplied.
CANONICAL_SCHEMA: dict[str, str] = {
    "variant_id": "uniprot:position:wt>mut — identical to vpdl.splits.variant_keys",
    "gene": "HGNC symbol, resolved from the accession",
    "chromosome": "GRCh38 chromosome from ClinVar (NaN without a ClinVar record)",
    "genomic_position": "GRCh38 VCF POS; ';'-joined when several nucleotide changes give this protein change",
    "ref": "VCF REF allele(s)",
    "alt": "VCF ALT allele(s)",
    "genome_build": "GRCh38 when genomic fields are filled",
    "transcript": "RefSeq transcript of the ClinVar HGVS",
    "transcript_status": "mane | mane_version_mismatch | non_mane | absent",
    "hgvs_c": "coding change(s) from ClinVar, ';'-joined",
    "hgvs_p": "normalised three-letter protein change",
    "protein_position": "1-based UniProt residue (same as `position`)",
    "wt_residue": "reference residue, verified against the UniProt sequence",
    "mutant_residue": "substituted residue",
    "clinical_label": "1 pathogenic / 0 benign / NaN — ClinVar, >= min stars, after QC",
    "label_confidence": "high (>=3 stars) | medium (2) | low (1) | none",
    "clinvar_review_status": "review status of the highest-starred record",
    "clinvar_star_rating": "0-4, highest over the records",
    "clinvar_n_records": "distinct ClinVar VariationIDs behind this protein change",
    "clinvar_variation_ids": "';'-joined VariationIDs",
    "expert_label": "label from expert-panel review (ClinVar >=3 stars or supplied expert CSV)",
    "expert_source": "where expert_label came from",
    "gnomad_af": "protein-level allele frequency (sum over alleles giving this change)",
    "gnomad_ac": "allele count, summed over alleles",
    "gnomad_an": "allele number (max over alleles)",
    "gnomad_observed": "1 when gnomAD reports any allele for this change",
    "population_reliable": "False where short-read frequencies are untrustworthy (PMS2CL region)",
    "protein_accession": "UniProt accession (same as `uniprot_id`)",
    "protein_sequence_sha256": "12-hex sequence version; sequences in the sidecar JSON",
    "structure_id": "e.g. AF-P40692-F1-model_v6",
    "structure_available": "residue has coordinates in structure_id",
    "functional_assay_available": "any validation-only functional value present",
    "cimra_value": "CIMRA OddsPath (validation only)",
    "cimra_strength": "Tavtigian PS3/BS3 bucket of cimra_value",
    "dms_score": "continuous ProteinGym DMS_score (validation only; higher = fitter)",
    "mavedb_score": "MaveDB score (validation only; see assay for direction)",
    "pms2_pseudogene_risk": "PMS2 residue inside the PMS2CL homology codons",
    "orthogonal_confirmation": "none | method supplied in the confirmation CSV",
    "qc_status": "pass | withheld (see exclusion_reason)",
    "qc_flags": "';'-joined QC flags, withholding and warning",
    "exclusion_reason": "why supervision was withheld, human-readable",
    "protein_cluster": "sequence family (MutL | MutS) — the cluster split unit",
    "cluster_id": "homology group: aligned residues across paralogs share it",
    "split_logo": "held-out fold under leave-one-gene-out (= gene)",
    "split_family": "held-out fold under the family (sequence-cluster) split",
    "split_random_debug": "fold index of the residue-grouped random split (debug only)",
    "split_functional": "validation when a functional value exists, else none",
    "split_assignment": "primary protocol fold (leave-one-gene-out)",
}

MISSING_VALUE_POLICY: dict[str, str] = {
    "labels": "NaN means no supervision. Never imputed, never defaulted.",
    "gnomad": "Absent from gnomAD is evidence (PM2, AF floor) in feature columns; "
              "gnomad_af stays NaN and gnomad_observed = 0.",
    "structure": "NaN features with structure_available = False; never imputed from neighbours.",
    "functional": "NaN with functional_assay_available = False; never a feature.",
    "genomic": "NaN when no ClinVar record exists (DMS/AlphaMissense-only rows).",
    "features": "Median of the TRAINING fold (vpdl.features.build_feature_matrix), per fold.",
}

# Flags that withhold supervision. Everything else in qc_flags is a warning.
WITHHOLDING_FLAGS: dict[str, str] = {
    "reference_mismatch": "reported wild-type residue differs from the UniProt sequence",
    "clinvar_discordant": "ClinVar records for this protein change disagree (P/LP vs B/LB)",
    "hgvs_inconsistent": "ClinVar c. and p. place the change in different codons",
    "pms2_homology_unconfirmed": "PMS2CL homology region (exons 11-15) without orthogonal confirmation",
    "expert_conflict": "expert-panel label contradicts the ClinVar aggregate",
}

_CONFIDENCE = {4: "high", 3: "high", 2: "medium", 1: "low", 0: "none"}


def _key(record: Mapping[str, Any]) -> tuple[str, int, str, str]:
    return (record["gene"], int(record["position"]), record["wt_aa"], record["mut_aa"])


def clinvar_protein_labels(records: Iterable[Mapping[str, Any]],
                           min_stars: int = 2,
                           expert_min_stars: int = 3) -> pd.DataFrame:
    """Summarise every ClinVar record behind each protein change, order-free.

    Records come from :func:`vpdl.sources.clinvar.iter_variant_summary` with
    ``min_stars=0``, so low-star records are counted but only records at
    ``>= min_stars`` supply the label. Two eligible records with opposite
    classifications make the change ``clinvar_discordant`` (label withheld);
    an eligible VUS beside a P/LP record is noted, not a conflict.
    """
    groups: dict[tuple, list[Mapping[str, Any]]] = {}
    for record in records:
        groups.setdefault(_key(record), []).append(record)

    rows = []
    for (gene, position, wt, mut), recs in groups.items():
        eligible = [r for r in recs if r["stars"] >= min_stars]
        labels = {r["label"] for r in eligible if pd.notna(r["label"])}
        expert = {r["label"] for r in recs
                  if r["stars"] >= expert_min_stars and pd.notna(r["label"])}
        best = max(recs, key=lambda r: r["stars"])
        grch38 = [r for r in recs if (r.get("assembly") or "") == "GRCh38"]
        genomic = sorted({(r.get("chromosome"), r.get("position_vcf"),
                           r.get("ref_vcf"), r.get("alt_vcf")) for r in grch38})
        names = [hgvs.parse_clinvar_name(r["name"]) for r in (grch38 or recs)]
        transcripts = sorted({n.transcript for n in names if n.transcript})
        coding = sorted({n.hgvs_c for n in names if n.hgvs_c})
        consistency = {hgvs.hgvs_consistency(c, position) for c in coding}
        variation_ids = sorted({str(r.get("variation_id")) for r in recs
                                if r.get("variation_id")})

        def joined(index: int) -> str | float:
            values = [str(g[index]) for g in genomic if g[index] not in (None, "", "na")]
            return ";".join(values) if values else np.nan

        label = next(iter(labels)) if len(labels) == 1 else np.nan
        rows.append({
            "gene": gene, "position": position, "wt_aa": wt, "mut_aa": mut,
            "clinvar_label": label,
            "clinvar_discordant": len(labels) > 1,
            "clinvar_eligible_vus": bool(labels) and any(pd.isna(r["label"]) for r in eligible),
            "expert_label_clinvar": next(iter(expert)) if len(expert) == 1 else np.nan,
            "clinvar_star_rating": int(best["stars"]),
            "clinvar_review_status": best["review_status"],
            "clinvar_n_records": len(variation_ids) or len(recs),
            "clinvar_variation_ids": ";".join(variation_ids) or np.nan,
            "chromosome": joined(0) if genomic else np.nan,
            "genomic_position": joined(1) if genomic else np.nan,
            "ref": joined(2) if genomic else np.nan,
            "alt": joined(3) if genomic else np.nan,
            "genome_build": "GRCh38" if genomic else np.nan,
            "transcript": ";".join(transcripts) if transcripts else np.nan,
            "hgvs_c": ";".join(coding) if coding else np.nan,
            "hgvs_inconsistent": "inconsistent" in consistency,
            "multi_nucleotide": len(genomic) > 1,
        })
    return pd.DataFrame(rows, columns=_CLINVAR_SUMMARY_COLUMNS)


_CLINVAR_SUMMARY_COLUMNS = [
    "gene", "position", "wt_aa", "mut_aa", "clinvar_label", "clinvar_discordant",
    "clinvar_eligible_vus", "expert_label_clinvar", "clinvar_star_rating",
    "clinvar_review_status", "clinvar_n_records", "clinvar_variation_ids", "chromosome",
    "genomic_position", "ref", "alt", "genome_build", "transcript", "hgvs_c",
    "hgvs_inconsistent", "multi_nucleotide",
]


def gnomad_protein_frequencies(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Protein-level gnomAD counts from :func:`vpdl.sources.gnomad.iter_gnomad_records`.

    Several alleles can encode one protein change; its frequency is the sum of
    theirs (distinct alleles essentially never co-occur on one haplotype).
    Assembly's feature column keeps the first allele instead, which is left
    alone so existing tables stay reproducible; the difference is counted in
    the canonical report.
    """
    groups: dict[tuple, list[Mapping[str, Any]]] = {}
    for record in records:
        groups.setdefault(_key(record), []).append(record)
    rows = []
    for (gene, position, wt, mut), recs in groups.items():
        afs = [r["af"] for r in recs if np.isfinite(r["af"])]
        rows.append({
            "gene": gene, "position": position, "wt_aa": wt, "mut_aa": mut,
            "gnomad_af": float(np.sum(afs)) if afs else np.nan,
            "gnomad_ac": float(np.sum([r["ac"] for r in recs])),
            "gnomad_an": float(np.max([r["an"] for r in recs])),
            "gnomad_n_alleles": len(recs),
        })
    return pd.DataFrame(rows)


def load_confirmations(path: Path | str) -> pd.DataFrame:
    """Orthogonal confirmations for PMS2 homology-region variants.

    CSV with ``gene, position, wt_aa, mut_aa, method`` (e.g. long-range PCR,
    cDNA). Only what is listed is trusted; there is no default confirmation.
    """
    frame = pd.read_csv(path)
    required = {"gene", "position", "wt_aa", "mut_aa", "method"}
    if missing := required - set(frame.columns):
        raise ValueError(f"{path}: confirmation CSV lacks {sorted(missing)}")
    frame["position"] = frame["position"].astype(int)
    return frame[list(required)]


def load_expert_labels(path: Path | str, source: str = "insight") -> pd.DataFrame:
    """Expert classifications (InSiGHT / ClinGen VCEP export) as labels.

    CSV with ``gene`` and either ``hgvs_p`` or ``position, wt_aa, mut_aa``,
    plus ``classification`` (Pathogenic, Likely pathogenic, Benign, Likely
    benign; anything else carries no label). No values are bundled: the
    InSiGHT database's terms govern redistribution.
    """
    from vpdl.sources.clinvar import significance_to_label

    frame = pd.read_csv(path)
    if "classification" not in frame.columns or "gene" not in frame.columns:
        raise ValueError(f"{path}: expert CSV needs gene and classification columns")
    if "hgvs_p" in frame.columns:
        parsed = frame["hgvs_p"].map(hgvs.normalize_hgvs_p)
        frame = frame[parsed.notna()].copy()
        parsed = parsed[parsed.notna()]
        frame["wt_aa"] = [p[0] for p in parsed]
        frame["position"] = [p[1] for p in parsed]
        frame["mut_aa"] = [p[2] for p in parsed]
    frame["label"] = frame["classification"].map(significance_to_label)
    frame["expert_source"] = f"{source}:{Path(path).name}"
    frame["position"] = frame["position"].astype(int)
    return frame[["gene", "position", "wt_aa", "mut_aa", "label", "expert_source"]]


@dataclass
class CanonicalReport:
    rows: int = 0
    rows_per_gene: dict[str, int] = field(default_factory=dict)
    sources_supplied: dict[str, bool] = field(default_factory=dict)
    clinical_labels: dict[str, dict[str, int]] = field(default_factory=dict)
    qc_flag_counts: dict[str, int] = field(default_factory=dict)
    withheld_labels: int = 0
    label_changes_vs_assembled: int = 0
    clinvar_records: int = 0
    clinvar_multi_record_changes: int = 0
    clinvar_discordant_changes: int = 0
    transcript_status: dict[str, int] = field(default_factory=dict)
    flagged_records: int = 0
    gnomad_first_allele_differs: int = 0
    pms2: dict[str, int] = field(default_factory=dict)
    structure_coverage: dict[str, float] = field(default_factory=dict)
    functional_coverage: dict[str, int] = field(default_factory=dict)
    families: dict[str, str] = field(default_factory=dict)
    sequence_versions: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _flag(flags: pd.Series, mask: pd.Series | np.ndarray, name: str) -> pd.Series:
    mask = pd.Series(np.asarray(mask, dtype=bool), index=flags.index)
    return flags.where(~mask, flags.map(lambda f: f + [name]))


def build_canonical(
    table: pd.DataFrame,
    sequences: Mapping[str, str],
    clinvar_records: Iterable[Mapping[str, Any]] | None = None,
    gnomad_records: Iterable[Mapping[str, Any]] | None = None,
    functional: pd.DataFrame | None = None,
    structure_index: Mapping[str, Mapping[str, Any]] | None = None,
    expert: pd.DataFrame | None = None,
    confirmations: pd.DataFrame | None = None,
    min_stars: int = 2,
    pms2_codon_range: tuple[int, int] = PMS2_HOMOLOGY_CODONS,
    families: Mapping[str, str] = PROTEIN_FAMILIES,
    n_debug_folds: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, CanonicalReport, pd.DataFrame]:
    """Assembled table -> (canonical table, report, flagged ClinVar records).

    `clinvar_records` should be ``iter_variant_summary(path, genes, min_stars=0)``.
    Without it, labels stay as assembled and the genomic fields are NaN — the
    report records that, rather than the table pretending otherwise.
    """
    report = CanonicalReport()
    key_columns = ["uniprot_id", "position", "wt_aa", "mut_aa", "gene"]
    if missing := set(key_columns) - set(table.columns):
        raise ValueError(f"not an assembled table: missing {sorted(missing)}")

    out = table.copy().reset_index(drop=True)
    out["position"] = out["position"].astype(int)
    out["variant_id"] = variant_keys(out)
    duplicated = out["variant_id"].duplicated()
    if duplicated.any():
        raise ValueError(
            f"{int(duplicated.sum())} duplicate variant ids in the assembled table, e.g. "
            f"{out.loc[duplicated, 'variant_id'].head(3).tolist()}. Assembly guarantees one "
            "row per variant; a duplicate means the table was edited or concatenated.")
    flags = pd.Series([[] for _ in range(len(out))], index=out.index, dtype=object)
    reasons: dict[str, str] = dict(WITHHOLDING_FLAGS)

    # -- identity and the reference-residue check --------------------------
    out["protein_position"] = out["position"]
    out["wt_residue"], out["mutant_residue"] = out["wt_aa"], out["mut_aa"]
    out["protein_accession"] = out["uniprot_id"]
    out["hgvs_p"] = [hgvs.format_hgvs_p(w, p, m)
                     for w, p, m in zip(out["wt_aa"], out["position"], out["mut_aa"])]
    versions = {acc: sequence_sha(seq) for acc, seq in sequences.items()}
    report.sequence_versions = versions
    out["protein_sequence_sha256"] = out["uniprot_id"].map(versions)
    reference_ok = np.array([
        acc in sequences and validate_against_sequence(sequences[acc], int(pos), wt)
        for acc, pos, wt in zip(out["uniprot_id"], out["position"], out["wt_aa"])])
    flags = _flag(flags, ~reference_ok, "reference_mismatch")

    # -- ClinVar: every record, order-independent --------------------------
    label_columns = [c for c in out.columns if c.startswith("label__")]
    assembled_clinvar = out.get("label__clinvar", pd.Series(np.nan, index=out.index)).copy()
    report.sources_supplied["clinvar_records"] = clinvar_records is not None
    flagged = pd.DataFrame()
    if clinvar_records is not None:
        records = list(clinvar_records)
        report.clinvar_records = len(records)
        summary = clinvar_protein_labels(records, min_stars=min_stars)
        merged = out[key_columns].merge(summary, on=["gene", "position", "wt_aa", "mut_aa"],
                                        how="left")
        for column in summary.columns:
            if column not in ("gene", "position", "wt_aa", "mut_aa"):
                out[column] = merged[column].to_numpy()
        present = set(zip(out["gene"], out["position"], out["wt_aa"], out["mut_aa"]))
        orphan_mask = np.array([k not in present for k in
                                zip(summary["gene"], summary["position"],
                                    summary["wt_aa"], summary["mut_aa"])], dtype=bool)
        orphans = summary.loc[orphan_mask]
        flagged = _explain_orphans(orphans, sequences, table)
        report.flagged_records = len(flagged)
        report.clinvar_multi_record_changes = int((summary["clinvar_n_records"] > 1).sum())
        report.clinvar_discordant_changes = int(summary["clinvar_discordant"].sum())
        out["clinical_label"] = out["clinvar_label"]
        flags = _flag(flags, out["clinvar_discordant"].fillna(False).astype(bool),
                      "clinvar_discordant")
        flags = _flag(flags, out["hgvs_inconsistent"].fillna(False).astype(bool),
                      "hgvs_inconsistent")
        flags = _flag(flags, out["multi_nucleotide"].fillna(False).astype(bool),
                      "multi_nucleotide")
        flags = _flag(flags, out["clinvar_eligible_vus"].fillna(False).astype(bool),
                      "clinvar_vus_alongside_classification")
        out = out.drop(columns=["clinvar_label", "clinvar_discordant", "hgvs_inconsistent",
                                "multi_nucleotide", "clinvar_eligible_vus"])
    else:
        out["clinical_label"] = assembled_clinvar
        for column in ("chromosome", "genomic_position", "ref", "alt", "genome_build",
                       "transcript", "hgvs_c", "clinvar_review_status",
                       "clinvar_variation_ids", "expert_label_clinvar"):
            out[column] = np.nan
        out["clinvar_star_rating"] = np.nan
        out["clinvar_n_records"] = np.nan

    out["transcript_status"] = [
        hgvs.transcript_status(g, t.split(";")[0] if isinstance(t, str) else None)
        for g, t in zip(out["gene"], out["transcript"])]
    non_mane = out["transcript_status"].isin(["non_mane", "mane_version_mismatch"])
    flags = _flag(flags, non_mane, "transcript_not_mane")
    out["label_confidence"] = out["clinvar_star_rating"].map(
        lambda s: _CONFIDENCE.get(int(s), "none") if pd.notna(s) else "none")

    # -- expert labels ------------------------------------------------------
    out["expert_label"] = out["expert_label_clinvar"]
    out["expert_source"] = np.where(out["expert_label"].notna(),
                                    "clinvar:reviewed_by_expert_panel", None)
    report.sources_supplied["expert_csv"] = expert is not None
    if expert is not None:
        joined = out[key_columns].merge(_one_per_key(expert, "label"),
                                        on=["gene", "position", "wt_aa", "mut_aa"], how="left")
        external = joined["label"]
        disagree = (external.notna() & out["expert_label"].notna()
                    & (external != out["expert_label"])).to_numpy()
        take = external.notna().to_numpy() & ~disagree
        out.loc[take, "expert_label"] = external[take].to_numpy()
        out.loc[take, "expert_source"] = joined.loc[take, "expert_source"].to_numpy()
        out.loc[disagree, "expert_label"] = np.nan
        flags = _flag(flags, disagree, "expert_conflict")
    clinical_vs_expert = (out["expert_label"].notna() & out["clinical_label"].notna()
                          & (out["expert_label"] != out["clinical_label"]))
    flags = _flag(flags, clinical_vs_expert, "expert_conflict")

    # -- PMS2 CL homology rule ------------------------------------------------
    start, end = pms2_codon_range
    risk = (out["gene"] == "PMS2") & out["position"].between(start, end)
    out["pms2_pseudogene_risk"] = risk
    out["orthogonal_confirmation"] = "none"
    report.sources_supplied["pms2_confirmations"] = confirmations is not None
    if confirmations is not None:
        joined = out[key_columns].merge(_one_per_key(confirmations, "method"),
                                        on=["gene", "position", "wt_aa", "mut_aa"], how="left")
        confirmed = joined["method"].notna().to_numpy()
        out.loc[confirmed, "orthogonal_confirmation"] = joined.loc[confirmed, "method"].to_numpy()
    unconfirmed = risk & (out["orthogonal_confirmation"] == "none")
    flags = _flag(flags, unconfirmed, "pms2_homology_unconfirmed")
    flags = _flag(flags, risk & ~unconfirmed, "pms2_homology_confirmed")
    out["population_reliable"] = ~risk
    report.pms2 = {"rows": int((out["gene"] == "PMS2").sum()),
                   "in_homology_region": int(risk.sum()),
                   "confirmed": int((risk & ~unconfirmed).sum()),
                   "codon_start": start, "codon_end": end}

    # -- apply withholding to every training/evaluation label --------------
    withholding = flags.map(lambda f: [x for x in f if x in WITHHOLDING_FLAGS])
    withheld = withholding.map(bool).to_numpy()
    out["qc_status"] = np.where(withheld, "withheld", "pass")
    out["qc_flags"] = flags.map(lambda f: ";".join(dict.fromkeys(f)))
    out["exclusion_reason"] = withholding.map(
        lambda f: "; ".join(reasons[x] for x in dict.fromkeys(f)) if f else "")
    out.loc[withheld, "clinical_label"] = np.nan
    out.loc[withheld, "expert_label"] = np.nan
    out["label__clinvar"] = out["clinical_label"]
    if out["expert_label"].notna().any():
        out["label__clinvar_expert"] = out["expert_label"]
    for column in label_columns:
        if column != "label__clinvar":
            out.loc[withheld, column] = np.nan
    report.withheld_labels = int((withheld & (assembled_clinvar.notna().to_numpy()
                                              | out["expert_label"].notna().to_numpy())).sum())
    changed = ~((out["label__clinvar"] == assembled_clinvar)
                | (out["label__clinvar"].isna() & assembled_clinvar.isna()))
    report.label_changes_vs_assembled = int(changed.sum())

    sources = label_sources_in(out)
    if sources:
        out["label"] = resolve_labels(out, sources)
        stacked = out[[f"label__{s}" for s in sources]]
        winner = stacked.notna().idxmax(axis=1).str.replace("label__", "", regex=False)
        conflicted = stacked.nunique(axis=1) > 1
        out["label_source"] = np.where(out["label"].notna(), winner,
                                       np.where(conflicted, "conflict_quarantined", "unlabelled"))

    # -- population (protein-level) ----------------------------------------
    report.sources_supplied["gnomad_records"] = gnomad_records is not None
    observed = out.get("feature_gnomad_observed")
    out["gnomad_observed"] = observed if observed is not None else np.nan
    if gnomad_records is not None:
        freq = gnomad_protein_frequencies(gnomad_records)
        merged = out[key_columns].merge(freq, on=["gene", "position", "wt_aa", "mut_aa"],
                                        how="left")
        for column in ("gnomad_af", "gnomad_ac", "gnomad_an"):
            out[column] = merged[column].to_numpy()
        out["gnomad_observed"] = out["gnomad_af"].notna().astype(float)
        if "feature_gnomad_log10_af" in out.columns:
            first = 10 ** out["feature_gnomad_log10_af"]
            both = out["gnomad_af"].notna() & (out["feature_gnomad_observed"] == 1)
            report.gnomad_first_allele_differs = int(
                (both & ~np.isclose(first, out["gnomad_af"], rtol=1e-6)).sum())
    else:
        log_af = out.get("feature_gnomad_log10_af")
        observed_mask = (out["gnomad_observed"] == 1) if observed is not None else False
        out["gnomad_af"] = np.where(observed_mask, 10 ** log_af, np.nan) \
            if log_af is not None else np.nan
        out["gnomad_ac"] = np.nan
        out["gnomad_an"] = np.nan

    # -- structure ----------------------------------------------------------
    report.sources_supplied["structure"] = structure_index is not None
    out["structure_id"] = np.nan
    out["structure_available"] = False
    if structure_index is not None:
        out["structure_id"] = out["uniprot_id"].map(
            {acc: entry.get("structure_id") for acc, entry in structure_index.items()})
        out["structure_available"] = [
            pos in structure_index.get(acc, {}).get("residues", ())
            for acc, pos in zip(out["uniprot_id"], out["position"])]
        report.structure_coverage = {
            gene: round(float(rows["structure_available"].mean()), 4)
            for gene, rows in out.groupby("gene")}

    # -- functional (validation only) --------------------------------------
    report.sources_supplied["functional"] = functional is not None
    for column in ("cimra_value", "cimra_strength", "dms_score", "mavedb_score"):
        out[column] = np.nan
    if functional is not None and len(functional):
        fkeys = ["uniprot_id", "position", "wt_aa", "mut_aa"]
        for assay, rows in functional.groupby("assay"):
            target = ("cimra_value" if assay == "cimra" else
                      "dms_score" if assay == "pg_dms_continuous" else "mavedb_score")
            merged = out[fkeys].merge(rows[fkeys + ["raw_score"]
                                           + (["cimra_strength"] if "cimra_strength" in rows else [])]
                                      .drop_duplicates(fkeys), on=fkeys, how="left")
            fill = merged["raw_score"].notna().to_numpy() & out[target].isna().to_numpy()
            out.loc[fill, target] = merged.loc[fill, "raw_score"].to_numpy()
            if target == "cimra_value" and "cimra_strength" in merged:
                out.loc[fill, "cimra_strength"] = merged.loc[fill, "cimra_strength"].to_numpy()
            report.functional_coverage[assay] = int(merged["raw_score"].notna().sum())
    out["functional_assay_available"] = out[["cimra_value", "dms_score",
                                             "mavedb_score"]].notna().any(axis=1)

    # -- clusters and split assignments -------------------------------------
    out["protein_cluster"] = out["uniprot_id"].map(lambda acc: family_of(acc, families))
    report.families = {acc: family_of(acc, families) for acc in sequences}
    groups = homology_groups({acc: seq for acc, seq in sequences.items()
                              if acc in set(out["uniprot_id"])}, families)
    out["cluster_id"] = [groups.get((acc, int(pos)), f"{acc}:{int(pos)}")
                         for acc, pos in zip(out["uniprot_id"], out["position"])]
    out["split_logo"] = out["gene"]
    out["split_family"] = out["protein_cluster"]
    out["split_random_debug"] = _hash_folds(group_keys(out), n_debug_folds, seed)
    out["split_functional"] = np.where(out["functional_assay_available"], "validation", "none")
    out["split_assignment"] = out["split_logo"]

    # -- report --------------------------------------------------------------
    report.rows = len(out)
    report.rows_per_gene = out["gene"].value_counts().sort_index().to_dict()
    report.clinical_labels = {
        gene: {"pathogenic": int((rows["clinical_label"] == 1).sum()),
               "benign": int((rows["clinical_label"] == 0).sum())}
        for gene, rows in out.groupby("gene")}
    report.qc_flag_counts = dict(sorted(Counter(
        f for fl in flags for f in dict.fromkeys(fl)).items()))
    report.transcript_status = out["transcript_status"].value_counts().to_dict()

    out = out.drop(columns=["expert_label_clinvar"])
    ordered = list(CANONICAL_SCHEMA)
    rest = [c for c in out.columns if c not in ordered]
    return out[ordered + rest], report, flagged


def _one_per_key(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    """Collapse a user-supplied CSV to one row per variant before merging.

    Duplicate keys would multiply rows in a left merge and misalign every
    column after it. Duplicates that agree collapse; duplicates that disagree
    on `value` keep no value — the conflict is not resolved by guessing.
    """
    keys = ["gene", "position", "wt_aa", "mut_aa"]
    frame = frame.copy()
    frame["position"] = frame["position"].astype(int)
    agree = frame.groupby(keys)[value].transform(lambda s: s.nunique(dropna=True) <= 1)
    frame.loc[~agree, value] = np.nan
    return frame.drop_duplicates(subset=keys, keep="first")


def _hash_folds(groups: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Deterministic fold per residue group (stable across processes and machines)."""
    import hashlib

    return np.array([int.from_bytes(hashlib.sha256(f"{seed}:{g}".encode()).digest()[:4],
                                    "big") % k for g in groups])


def _explain_orphans(orphans: pd.DataFrame, sequences: Mapping[str, str],
                     table: pd.DataFrame) -> pd.DataFrame:
    """ClinVar protein changes with no row in the table, and why. Never repaired."""
    from vpdl.sources.uniprot import MMR_ACCESSIONS

    accession = {gene: acc for gene, (acc, _) in MMR_ACCESSIONS.items()}
    accession.update(dict(zip(table["gene"], table["uniprot_id"])))
    reasons = []
    for row in orphans.itertuples():
        acc = accession.get(row.gene)
        seq = sequences.get(acc) if acc else None
        if seq is None:
            reason = "no_canonical_sequence"
        elif not 1 <= row.position <= len(seq):
            reason = "position_out_of_range"
        elif seq[row.position - 1] != row.wt_aa:
            reason = f"reference_mismatch (UniProt has {seq[row.position - 1]})"
        else:
            reason = "valid_but_not_in_assembled_table (below min stars and no feature source row)"
        reasons.append(reason)
    return orphans.assign(uniprot_id=[accession.get(g) for g in orphans["gene"]],
                          flag_reason=reasons)


def validate_canonical(frame: pd.DataFrame) -> list[str]:
    """Schema problems as messages; empty list means the table is well-formed."""
    problems = [f"missing column {c}" for c in CANONICAL_SCHEMA if c not in frame.columns]
    if problems:
        return problems
    if frame["variant_id"].duplicated().any():
        problems.append("duplicate variant_id")
    bad = ~frame["qc_status"].isin(["pass", "withheld"])
    if bad.any():
        problems.append(f"{int(bad.sum())} rows with unknown qc_status")
    leaked = (frame["qc_status"] == "withheld") & frame["clinical_label"].notna()
    if leaked.any():
        problems.append(f"{int(leaked.sum())} withheld rows still carry a clinical label")
    gate = frame["pms2_pseudogene_risk"] & (frame["orthogonal_confirmation"] == "none") \
        & frame["label__clinvar"].notna()
    if gate.any():
        problems.append(f"{int(gate.sum())} unconfirmed PMS2 homology rows are labelled")
    for column in ("label__clinvar", "clinical_label"):
        values = frame[column].dropna()
        if not values.isin([0.0, 1.0]).all():
            problems.append(f"{column} has values other than 0/1")
    return problems


def attach_sequences(frame: pd.DataFrame, sequences: Mapping[str, str]) -> pd.DataFrame:
    """Materialise ``protein_sequence`` in memory (never written per row)."""
    return frame.assign(protein_sequence=frame["uniprot_id"].map(sequences))


def write_canonical(frame: pd.DataFrame, report: CanonicalReport, flagged: pd.DataFrame,
                    sequences: Mapping[str, str], out_path: Path | str,
                    meta: Mapping[str, Any] | None = None,
                    extra_artefacts: Sequence[Path | str] = ()) -> dict[str, Path]:
    """Write table, sequence sidecar, flagged records, report; then the manifest.

    The manifest is written LAST, after every file it names is final, so its
    checksums describe the files on disk (landmine L10).
    """
    from vpdl.provenance import write_manifest

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.with_suffix("")
    paths = {
        "table": out_path,
        "sequences": Path(f"{stem}.sequences.json"),
        "flagged": Path(f"{stem}.flagged.csv"),
        "report": Path(f"{stem}.report.json"),
    }
    frame.to_csv(paths["table"], index=False)
    paths["sequences"].write_text(json.dumps(
        {acc: {"sequence": seq, "sha256_12": sequence_sha(seq)}
         for acc, seq in sorted(sequences.items())}, indent=1))
    flagged.to_csv(paths["flagged"], index=False)
    paths["report"].write_text(json.dumps(report.as_dict(), indent=2, default=str))
    manifest = Path(f"{stem}.manifest.json")
    write_manifest(manifest, artefacts=list(paths.values()) + [Path(p) for p in extra_artefacts],
                   meta={"kind": "canonical", "schema": list(CANONICAL_SCHEMA),
                         "missing_value_policy": MISSING_VALUE_POLICY,
                         "withholding_flags": WITHHOLDING_FLAGS,
                         "label_precedence": LABEL_PRECEDENCE, **dict(meta or {})})
    paths["manifest"] = manifest
    return paths
