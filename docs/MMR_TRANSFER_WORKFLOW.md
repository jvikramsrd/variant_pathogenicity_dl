# MMR transfer-learning workflow (Phases 1–3)

No data build, extraction, training, or evaluation was run when this workflow
was added.

## Stage 1 — broad pretraining

`scripts/train_transfer.py pretrain` uses the 80-gene table and deliberately
excludes DMS-derived feature columns. In `leave_gene_out` mode it excludes all
four MMR genes, preventing their variants from contaminating the general
representation before a transfer experiment.

## Stage 2 — MMR clinical fine-tuning

`scripts/train_transfer.py finetune` accepts only rows with a ClinVar or
ProteinGym-clinical label; DMS-only labels and functional-assay values remain
outside the final clinical fine-tuning target. The checkpoint feature schema
is checked exactly before weight transfer.

`--mode practical` produces an adapted MMR model. It can be useful in practice
but is not an unseen-gene claim. `--mode leave_gene_out --holdout_gene G`
pretrains without MMR data, fine-tunes without gene `G`, and reserves `G` for
the final downstream evaluation.

## Dedicated MMR build

Use `scripts/build_mmr_dataset.py` in a separate `data/mmr` directory. The
script requires a validated PMS2 exon 11–15 homology-region table with an
`orthogonally_confirmed` flag, or `--exclude_pms2`. It never guesses a
protein-coordinate mapping for the pseudogene region.

## Still required before a final performance claim

- Add InSiGHT/ClinGen VCEP and gnomAD adapters to the dedicated MMR table.
- Add CIMRA OddsPath and MSH2 DMS as held-out evaluation sets.
- Run leave-one-MMR-gene-out evaluation with bootstrap confidence intervals.
- Add the structure-based MVmamba branch only after the transfer baseline is
  established; current ESM extraction is sequence-only.
