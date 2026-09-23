"""A tiny SYNTHETIC ClinVar release, in ClinVar's own file layouts, for tests and smoke runs.

Every value is invented: gene/variant combinations, narratives, submitters,
dates and citations. The narratives are assembled from sentence templates
written for this file (no ClinVar text is copied); PubMed ids are in a range
reserved for tests (9,900,000+) except one deliberate mention of the
MSH2 DMS holdout paper's PMID, so the functional-holdout leakage check has
something to find. Nothing here is evidence about any real variant.
"""

from __future__ import annotations

import gzip
import random
from pathlib import Path

__all__ = ["write_synthetic_clinvar", "SYNTHETIC_GENES", "HOLDOUT_PMID"]

HOLDOUT_PMID = "33357406"      # MSH2 DMS (MaveDB urn:mavedb:00000050-a-1) — in the catalogue

SYNTHETIC_GENES = {
    # gene: (chromosome, transcript, disease name, disease id)
    "MLH1": ("3", "NM_000249.4", "Lynch syndrome 2", "MedGen:C1333991"),
    "MSH2": ("2", "NM_000251.3", "Lynch syndrome 1", "MedGen:C1333990"),
    "MSH6": ("2", "NM_000179.3", "Lynch syndrome 5", "MedGen:C1864935"),
    "PMS2": ("7", "NM_000535.7", "Lynch syndrome 4", "MedGen:C1837336"),
    "BRCA1": ("17", "NM_007294.4", "Hereditary breast ovarian cancer syndrome", "MedGen:C0677776"),
    "TP53": ("17", "NM_000546.6", "Li-Fraumeni syndrome", "MedGen:C0085390"),
    "KCNQ1": ("11", "NM_000218.3", "Long QT syndrome 1", "MedGen:C1141890"),
    "SCN5A": ("3", "NM_198056.3", "Brugada syndrome", "MedGen:C1142166"),
    "CFTR": ("7", "NM_000492.4", "Cystic fibrosis", "MedGen:C0010674"),
    "PAH": ("12", "NM_000277.3", "Phenylketonuria", "MedGen:C0031485"),
    "GENEA": ("1", "NM_900001.1", "Synthetic disorder A", "MedGen:C9000001"),
    "GENEB": ("5", "NM_900002.1", "Synthetic disorder B", "MedGen:C9000002"),
}
_AA = ["Ala", "Arg", "Asn", "Asp", "Cys", "Gln", "Glu", "Gly", "His", "Ile", "Leu", "Lys",
       "Met", "Phe", "Pro", "Ser", "Thr", "Trp", "Tyr", "Val"]
_AA_NAME = {"Ala": "alanine", "Arg": "arginine", "Asn": "asparagine", "Asp": "aspartic acid",
            "Cys": "cysteine", "Gln": "glutamine", "Glu": "glutamic acid", "Gly": "glycine",
            "His": "histidine", "Ile": "isoleucine", "Leu": "leucine", "Lys": "lysine",
            "Met": "methionine", "Phe": "phenylalanine", "Pro": "proline", "Ser": "serine",
            "Thr": "threonine", "Trp": "tryptophan", "Tyr": "tyrosine", "Val": "valine"}
_CLASSES = ["Pathogenic", "Likely pathogenic", "Uncertain significance", "Likely benign", "Benign"]
_LABS = ["Synthetic Lab Alpha", "Synthetic Lab Beta", "Synthetic Lab Gamma",
         "Synthetic Expert Panel", "OMIM"]

_PATHOGENIC_EVIDENCE = [
    "This variant was not observed in the gnomAD population database.",
    "It co-segregated with disease in {n} affected relatives from one family (PMID: {pmid}).",
    "Tumours from carriers showed loss of {gene} protein expression and high microsatellite instability.",
    "A functional assay showed markedly reduced activity compared with the wild-type protein.",
    "Computational prediction tools suggest the change is damaging to protein function.",
    "The affected residue is highly conserved across species.",
    "It was identified in trans with a known pathogenic variant in a patient with an early-onset phenotype.",
    "RNA analysis demonstrated exon skipping in patient-derived cells.",
]
_BENIGN_EVIDENCE = [
    "The variant is present in the general population at an allele frequency of {af}% in gnomAD.",
    "It did not segregate with disease in a family with {n} affected members.",
    "A functional assay showed normal activity indistinguishable from the wild-type protein.",
    "In silico tools predict the change to be tolerated.",
    "The residue is poorly conserved and a different amino acid is found in several species.",
    "It was observed in cis with a pathogenic variant in the same individual.",
]
_NEUTRAL = [
    "The significance of the residue's biochemical properties is unclear.",
    "Limited clinical information was provided with the sample.",
]
_OPENERS = {
    "Synthetic Lab Alpha": "This sequence change replaces {aa1} with {aa2} at codon {pos} of the {gene} protein ({p}).",
    "Synthetic Lab Beta": "The {c} variant in {gene} is a single nucleotide substitution predicted to change {aa1} to {aa2}.",
    "Synthetic Lab Gamma": "Variant summary: {gene} {c} ({p}) results in a {aa1} to {aa2} substitution.",
    "Synthetic Expert Panel": "The {gene} {p} variant was reviewed by the synthetic expert panel.",
    "OMIM": "In a synthetic family with {disease}, a {gene} {p} change was described.",
}
_CLOSERS = {
    "Pathogenic": "Therefore, this variant has been classified as Pathogenic.",
    "Likely pathogenic": "Based on the evidence outlined above, the variant was classified as likely pathogenic.",
    "Uncertain significance": "In summary, the available evidence is currently insufficient to determine the role of this variant in disease.",
    "Likely benign": "We classify this variant as likely benign.",
    "Benign": "Therefore, this variant is classified as benign.",
}
_CODES = {
    "Pathogenic": "The following criteria were applied: PM2_Supporting, PP1, PS3.",
    "Likely pathogenic": "The following criteria were applied: PM2_Supporting, PP3, PS3_Moderate.",
    "Uncertain significance": "The following criteria were applied: PM2_Supporting.",
    "Likely benign": "The following criteria were applied: BS3_Supporting, BP4.",
    "Benign": "The following criteria were applied: BA1, BS3.",
}
_DATE = ["Jan 15, 2018", "Mar 03, 2019", "Jul 21, 2020", "Nov 30, 2021", "Feb 11, 2022",
         "Aug 09, 2023", "Oct 17, 2024", "May 05, 2025", "Jan 20, 2026"]


def _narrative(rng: random.Random, lab: str, classification: str, gene: str, c: str, p: str,
               aa1: str, aa2: str, pos: int, disease: str) -> str:
    opener = _OPENERS[lab].format(aa1=_AA_NAME[aa1], aa2=_AA_NAME[aa2], pos=pos, gene=gene, p=p,
                                  c=c, disease=disease)
    pathogenic = classification in ("Pathogenic", "Likely pathogenic")
    benign = classification in ("Benign", "Likely benign")
    pool = (_PATHOGENIC_EVIDENCE if pathogenic else _BENIGN_EVIDENCE if benign
            else rng.sample(_PATHOGENIC_EVIDENCE, 2) + rng.sample(_BENIGN_EVIDENCE, 1) + _NEUTRAL)
    evidence = rng.sample(pool, k=min(len(pool), rng.randint(2, 4)))
    text = [opener] + [e.format(n=rng.randint(2, 6), pmid=9_900_000 + rng.randint(0, 99_999),
                                gene=gene, af=round(rng.uniform(0.2, 3.0), 2)) for e in evidence]
    if rng.random() < 0.4:
        text.append(_CODES[classification])
    if gene == "MSH2" and rng.random() < 0.3:
        text.append(f"A multiplexed functional study reported a loss-of-function score (PMID: {HOLDOUT_PMID}).")
    if rng.random() < 0.15:
        text.append(f"This variant has been reported as {classification.lower()} by other laboratories.")
    text.append(_CLOSERS[classification])
    return " ".join(text)


def write_synthetic_clinvar(directory: Path | str, variants_per_gene: int = 12, seed: int = 7) -> dict[str, Path]:
    rng = random.Random(seed)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    vs_header = ["#AlleleID", "Type", "Name", "GeneID", "GeneSymbol", "HGNC_ID", "ClinicalSignificance",
                 "ClinSigSimple", "LastEvaluated", "RS# (dbSNP)", "nsv/esv (dbVar)", "RCVaccession",
                 "PhenotypeIDS", "PhenotypeList", "Origin", "OriginSimple", "Assembly",
                 "ChromosomeAccession", "Chromosome", "Start", "Stop", "ReferenceAllele",
                 "AlternateAllele", "Cytogenetic", "ReviewStatus", "NumberSubmitters", "Guidelines",
                 "TestedInGTR", "OtherIDs", "SubmitterCategories", "VariationID", "PositionVCF",
                 "ReferenceAlleleVCF", "AlternateAlleleVCF"]
    ss_header = ["#VariationID", "ClinicalSignificance", "DateLastEvaluated", "Description",
                 "SubmittedPhenotypeInfo", "ReportedPhenotypeInfo", "ReviewStatus", "CollectionMethod",
                 "OriginCounts", "Submitter", "SCV", "SubmittedGeneSymbol", "ExplanationOfInterpretation",
                 "SomaticClinicalImpact", "Oncogenicity"]
    variant_rows, submission_rows, citation_rows = [], [], []
    variation_id, allele_id, scv = 100_000, 200_000, 900_000
    for gene_index, (gene, (chrom, transcript, disease, disease_id)) in enumerate(SYNTHETIC_GENES.items()):
        for k in range(variants_per_gene):
            variation_id += 1
            allele_id += 1
            aa1, aa2 = rng.sample(_AA, 2)
            pos = rng.randint(10, 900)
            c_pos = pos * 3 - rng.randint(0, 2)
            ref, alt = rng.sample("ACGT", 2)
            kind = rng.random()
            if kind < 0.7:
                c, p, vtype = f"c.{c_pos}{ref}>{alt}", f"p.{aa1}{pos}{aa2}", "single nucleotide variant"
            elif kind < 0.8:
                c, p, vtype = f"c.{c_pos}{ref}>{alt}", f"p.{aa1}{pos}Ter", "single nucleotide variant"
            elif kind < 0.9:
                c, p, vtype = f"c.{c_pos}+{rng.randint(1, 2)}{ref}>{alt}", "", "single nucleotide variant"
            else:
                c, p, vtype = f"c.{c_pos}del", f"p.{aa1}{pos}fs", "Deletion"
            name = f"{transcript}({gene}):{c}" + (f" ({p})" if p else "")
            labels = rng.choices(_CLASSES, weights=[2, 2, 3, 2, 2], k=rng.randint(1, 3))
            if rng.random() < 0.7:
                labels = [labels[0]] * len(labels)
            aggregate = labels[0] if len(set(labels)) == 1 else "Conflicting classifications of pathogenicity"
            status = ("reviewed by expert panel" if "Synthetic Expert Panel" in labels
                      else "criteria provided, multiple submitters, no conflicts" if len(labels) > 1
                      else "criteria provided, single submitter")
            if aggregate.startswith("Conflicting"):
                status = "criteria provided, conflicting classifications"
            position = 1_000_000 * (gene_index + 1) + c_pos
            date = rng.choice(_DATE)
            for assembly, shift in (("GRCh37", 500), ("GRCh38", 0)):
                variant_rows.append([str(allele_id), vtype, name, str(4000 + gene_index), gene,
                                     f"HGNC:{7000 + gene_index}", aggregate, "1", date, str(10_000 + variation_id),
                                     "-", f"RCV{variation_id:09d}", f"{disease_id}|MedGen:C3661900",
                                     f"{disease}|not provided", "germline", "germline", assembly,
                                     f"NC_0000{chrom}", chrom, str(position + shift), str(position + shift),
                                     "na", "na", "-", status, str(len(labels)), "-", "N", "-", "2",
                                     str(variation_id), str(position + shift), ref, alt])
            for index, label in enumerate(labels):
                scv += 1
                lab = rng.choice(_LABS[:3]) if index or rng.random() < 0.8 else rng.choice(_LABS[3:])
                review = ("reviewed by expert panel" if lab == "Synthetic Expert Panel"
                          else "no assertion criteria provided" if lab == "OMIM"
                          else "criteria provided, single submitter")
                method = "literature only" if lab == "OMIM" else "clinical testing"
                text = _narrative(rng, lab, label, gene, c, p or c, aa1, aa2, pos, disease)
                submission_rows.append([str(variation_id), label, rng.choice(_DATE), text,
                                        f"{disease}", f"{disease_id.split(':')[1]}:{disease}", review,
                                        method, "germline:1", lab, f"SCV{scv:09d}.1", gene, "-", "-", "-"])
            for _ in range(rng.randint(0, 2)):
                citation_rows.append([str(allele_id), str(variation_id), "-", "-", "PubMed",
                                      str(9_900_000 + rng.randint(0, 99_999))])
            if gene == "MSH2" and k % 4 == 0:
                citation_rows.append([str(allele_id), str(variation_id), "-", "-", "PubMed", HOLDOUT_PMID])

    paths = {"variant_summary": directory / "variant_summary.txt.gz",
             "submission_summary": directory / "submission_summary.txt.gz",
             "var_citations": directory / "var_citations.txt"}
    with gzip.open(paths["variant_summary"], "wt", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(vs_header) + "\n")
        for row in variant_rows:
            handle.write("\t".join(row) + "\n")
    with gzip.open(paths["submission_summary"], "wt", encoding="utf-8", newline="\n") as handle:
        handle.write("##Overview of interpretation of all variants in ClinVar (SYNTHETIC TEST FILE)\n")
        handle.write("##Columns follow NCBI's README\n")
        handle.write("\t".join(ss_header) + "\n")
        for row in submission_rows:
            handle.write("\t".join(row) + "\n")
    with paths["var_citations"].open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("#AlleleID\tVariationID\trs\tnsv\tcitation_source\tcitation_id\n")
        for row in citation_rows:
            handle.write("\t".join(row) + "\n")
    return paths
