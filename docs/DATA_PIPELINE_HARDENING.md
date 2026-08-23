# Dataset pipeline hardening

*Implemented 2026-08-23. The pipeline was not run after these code changes.*

## Guarantees added

- Downloads are now staged in a `.part` file and promoted atomically only
  after size and optional checksum validation. A server that ignores an HTTP
  Range request no longer corrupts a resumed download by appending a full
  response.
- ProteinGym ZIP archives are integrity-checked before parsing.
- Parsers fail clearly when required source columns are absent or no usable
  source files are found. Invalid DMS bins and within-assay contradictory
  labels are dropped rather than passed through.
- Clinical benchmark duplicates with conflicting labels are removed. DMS and
  zero-shot duplicates are reduced deterministically rather than depending on
  archive order.
- ClinVar and ProteinGym-clinical evidence are resolved deterministically.
  Any contradictory labels at the same variant key are marked explicitly.
- Cross-source clinical disagreement is quarantined: raw evidence remains in
  the master CSV, but `label` is blank and `label_conflict=1`, so it cannot
  enter the train CSV.
- Each label has `label_source` and `label_weight`. The defaults encode source
  reliability (ClinVar weighted by review stars, PG clinical 0.75, DMS 0.20)
  and are combined with the trainer's clinical/DMS experiment weights.
- The builder refuses to export a master table with null/duplicate keys,
  unresolved metadata, non-binary labels, unquarantined conflicts, or labels
  without a recognised source.
- The manifest now records checksums and byte counts of generated artefacts,
  not only raw downloads.

## Default label policy

The build CLI now defaults to ClinVar assertions with at least **two stars**.
Use `--min_stars 1` only for an intentionally broader, lower-confidence
training set. This is a data-quality choice, not a claim that one-star
assertions are always wrong.

## Remaining scientific limits

No merger can establish clinical truth from heterogeneous public evidence.
DMS labels are still functional-assay proxies, and AlphaMissense or published
model scores may have training-data overlap with clinical benchmarks. Use the
documented leave-one-protein-out and temporally external evaluations before
claiming prospective clinical performance.
