# Canonical variant dataset (DL branch)

Built by `vpdl-dl canonical` from the assembled table (`vpdl build`), written to
`data/built/canonical.csv` with a sequence sidecar, a QC report, a flagged-records
file and a manifest written last (landmine L10). One row per protein
substitution in MLH1 / MSH2 / MSH6 / PMS2. **No row is dropped for QC**: a failed
check withholds supervision and records why.

Counts below are not yet known — they come from the first DGX build and go in
the report (`canonical.report.json`), not in this file by hand.

## Sources

| source | role | licence | status in this build |
|---|---|---|---|
| UniProt P40692 / P43246 / P52701 / P54278 | coordinate authority (756 / 934 / 1360 / 862 aa) | CC BY 4.0 | existing |
| ClinVar `variant_summary.txt.gz` | labels (≥2 stars), genomic coordinates, HGVS, review status | public domain | existing; all records now read |
| ClinVar "reviewed by expert panel" (3★) | `expert_label`, `label__clinvar_expert` | public domain | new |
| InSiGHT / ClinGen VCEP export | `expert_label` via `--expert` CSV | per database terms (not bundled) | interface only |
| gnomAD v4 | AF features (existing) + protein-level AF/AC/AN | CC0 | existing + AC/AN |
| AlphaMissense | prior feature + orientation anchor | CC BY 4.0 | existing |
| ProteinGym DMS (MSH2 Jia 2020) | labels (existing arms) + continuous `dms_score` (validation) | MIT | existing + validation copy |
| CIMRA OddsPath (Drost; Rayner 2022) | validation only | paper supplements | interface only (`--cimra` CSV) |
| MaveDB score sets | validation only | per score set | interface only (`--mavedb`) |
| AlphaFold DB models (v6) | structure features, `structure_id` | CC BY 4.0 | cached locally for all four |
| Ensembl MANE transcripts | exon-structure features, PMS2 range check | Ensembl terms | fetched on DGX |

Atlas of Variant Effects: no MMR-gene dataset is bundled; a score set published
there enters through the same `--mavedb`-style validation interface, never as
labels.

## Schema

Required fields (see `CANONICAL_SCHEMA` in `vpdl/dl/canonical.py` for the
authoritative list and one-line meanings):

| group | columns |
|---|---|
| identity | `variant_id`, `gene`, `protein_accession`, `protein_position`, `wt_residue`, `mutant_residue`, `hgvs_p` |
| genomic | `chromosome`, `genomic_position`, `ref`, `alt`, `genome_build`, `transcript`, `transcript_status`, `hgvs_c` |
| labels | `clinical_label`, `label_confidence`, `clinvar_review_status`, `clinvar_star_rating`, `clinvar_n_records`, `clinvar_variation_ids`, `expert_label`, `expert_source` |
| population | `gnomad_af`, `gnomad_ac`, `gnomad_an`, `gnomad_observed`, `population_reliable` |
| protein | `protein_sequence_sha256` (+ sidecar `canonical.sequences.json`) |
| structure | `structure_id`, `structure_available` |
| functional (validation only) | `functional_assay_available`, `cimra_value`, `cimra_strength`, `dms_score`, `mavedb_score` |
| QC | `pms2_pseudogene_risk`, `orthogonal_confirmation`, `qc_status`, `qc_flags`, `exclusion_reason` |
| splits | `protein_cluster`, `cluster_id`, `split_logo`, `split_family`, `split_random_debug`, `split_functional`, `split_assignment` |

Plus every assembled column (`feature_*`, `label__*`, `label`, `label_source`,
`evidence_tier`, `uniprot_id`, `position`, `wt_aa`, `mut_aa`).

Two deliberate deviations from the requested field list, both documented in
the module:

* **`position` keeps its v2 meaning (protein position)**; the requested
  genomic "position" is `genomic_position`. Renaming `position` would break every
  existing layer and the LLM branch's evidence reader.
* **`protein_sequence` is not written per row.** It is ~1,000 characters that
  would repeat on ~74k rows (~75 MB) and carry nothing the sidecar does not.
  Rows carry `protein_sequence_sha256`; `attach_sequences()` materialises the
  column in memory.

## Quality control

| check | rule | effect |
|---|---|---|
| reference residue | `wt_residue` must equal UniProt at `protein_position` | mismatch → `reference_mismatch`, withheld; ClinVar records off the reference go to `canonical.flagged.csv` with the residue UniProt has. Never repositioned. |
| HGVS normalisation | three-/one-letter p. spellings → one identity; non-missense refused | — |
| transcript | ClinVar transcript vs pinned MANE (`NM_000249.4`, `NM_000251.3`, `NM_000179.3`, `NM_000535.7`) | `transcript_not_mane` warning (protein coordinates are validated regardless) |
| c./p. consistency | an exonic SNV's codon must equal the protein position | `hgvs_inconsistent`, withheld |
| duplicates | duplicate `variant_id` rows | build fails |
| multi-record changes | several ClinVar records → one protein change | counted; labels resolved over **all** eligible records |
| conflicts | eligible records disagree (P/LP vs B/LB) | `clinvar_discordant`, withheld (the assembled table kept the first by file order) |
| cross-source conflicts | ClinVar vs DMS disagree | quarantined per arm by `resolve_labels` (existing) |
| expert conflicts | expert label contradicts the ClinVar aggregate | `expert_conflict`, withheld |
| PMS2CL | PMS2 codons 382–862 (exons 11–15) without orthogonal confirmation | `pms2_homology_unconfirmed`, withheld; `population_reliable = False` |
| missing values | see `MISSING_VALUE_POLICY` | labels never imputed; structure/functional absent = NaN + availability flag |

### PMS2 homology rule

`pms2_pseudogene_risk` marks PMS2 residues 382–862 — exons 11–15 of
NM_000535.7, derived from Ensembl's exon table (v1 `derive_pms2_homology_range.py`,
now re-derived by `vpdl-dl genomic`, which refuses to continue if the derived
range differs from the constant). Supervision there is withheld unless the
variant is listed in `--confirmations` (`gene, position, wt_aa, mut_aa, method`,
e.g. long-range PCR or cDNA), recorded in `orthogonal_confirmation`. An
expert-panel classification is not treated as sequencing confirmation. The gene
itself is never dropped (landmine L9). gnomAD frequencies in the same region
come from the same short-read mapping problem, so `population_reliable` is
False there.

## Functional data stay out of training

`cimra_value`, `dms_score`, `mavedb_score` are never `feature_*` or `label__*`
columns; they are written again as `canonical.functional.csv` (oriented
`damage_score`, higher = more damaging, direction checked against
AlphaMissense) and used only by `vpdl-dl functional` after training. A feature
whose name carries a functional value is a **critical** leakage finding and
stops training. Training on assay labels remains possible only as its own arm
(`--train-sources clinvar pg_dms`), exactly as before.

## Derived tables

`vpdl-dl structure` adds `feature_struct_*` (see `vpdl/dl/structure.py`) and
`vpdl-dl genomic` adds `feature_genomic_*`; each writes a manifest and carries
the sequence sidecar forward. All DL training cells read **one** final table
(`data/built/canonical_full.csv`) so every arm shares a dataset hash and a test
set. Zero-shot scores are written to the feature store and, for the
`vpdl paired --feature-baseline` floor only, to a separate copy
(`canonical_zs.csv`) — never into the training table.
