# DGX Spark runbook — DL branch

Every command below runs **on the DGX Spark**, in the repository, with the
project venv active. Nothing here was run on the Windows PC except the tests.

Rule for every phase: if the **check** line does not match, stop and send the
output. Results go in `docs/RUNLOG.md` (newest first), with the cell names.

Shorthands used throughout — paste once per shell:

```bash
T=data/built/canonical_full.csv
S="clinvar pg_dms alphamissense gnomad"
```

---

## Phase 0 — ship the code (Windows PC)

1. Stage only the DL-branch paths.

```bash
git add vpdl/dl vpdl/models vpdl/sources/clinvar.py vpdl/sources/gnomad.py vpdl/evaluate.py vpdl/experiment.py vpdl/features.py tests/dl configs/dl scripts/check_hardware.py docs/dl pyproject.toml .gitignore
```

2. Look before committing.

```bash
git status
```

Check: no `deleted:` lines; nothing under `vpdl/kb`, `vpdl/slm`, `data/`.

3. Commit and push (the message is yours to choose).

```bash
git commit -m "DL branch: canonical table, leakage gate, PLM/sequence/fusion models, pretraining, DGX runbook"
```

```bash
git push
```

---

## Phase 1 — environment and tests (DGX)

1. Get the code.

```bash
git pull
```

2. See what is installed before installing anything.

```bash
python -c "import torch, transformers; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)"
```

Check: CUDA `True`. If `slm-train` is running, do not change packages under it.

3. Install the DL extra (same transformers floor as the SLM, so nothing is forced to upgrade).

```bash
pip install -e ".[dev,gbm,dl]"
```

4. Record the hardware.

```bash
python scripts/check_hardware.py --smoke --out runs/dl/hardware.json
```

Check: `bf16: true`, `smoke_bf16_matmul: true`, `smoke_esm_forward: true`; note `torch_compile_ready`.

5. Run the tests.

```bash
python -m pytest tests/dl tests/regression -q
```

Check: all pass (107 DL tests; the regression suite also runs its GPU test here).

---

## Phase 2 — data

1. AlphaFold models for the four proteins (~2.5 MB).

```bash
mkdir -p data/raw/alphafold && for a in P40692 P43246 P52701 P54278; do curl -sf https://alphafold.ebi.ac.uk/api/prediction/$a -o data/raw/alphafold/${a}_metadata.json && curl -sf "$(python -c "import json;print(json.load(open('data/raw/alphafold/${a}_metadata.json'))[0]['pdbUrl'])")" -o data/raw/alphafold/$a.pdb; done; ls -la data/raw/alphafold
```

Check: 8 files; each `.pdb` 0.5–0.9 MB.

2. Canonical table (reads every ClinVar record, gnomAD cache, structures, continuous DMS).

```bash
vpdl-dl canonical --table data/built/mmr.csv --clinvar data/raw/variant_summary.txt.gz --gnomad-cache data/cache/gnomad --structures data/raw/alphafold --pg-dms data/raw/DMS_ProteinGym_substitutions.zip --out data/built/canonical.csv
```

Check: `functional orientation ... pg_dms_continuous` is **positive**; per-gene clinical labels close to 174 / 177 / 78 / 21; note `label_changes_vs_assembled`, `clinvar_discordant_changes`, `flagged_records`.

3. Structure columns (Shrake–Rupley SASA runs on CPU, ~1 min).

```bash
vpdl-dl structure --data data/built/canonical.csv --models data/raw/alphafold --out data/built/canonical_struct.csv
```

Check: four `structure_id`s printed.

4. Genomic columns (fetches four Ensembl transcripts once).

```bash
vpdl-dl genomic --data data/built/canonical_struct.csv --fetch --out data/built/canonical_full.csv
```

Check: `PMS2CL homology codons from the exon table: (382, 862)`; exit 0. Any other range stops here.

5. Leakage report for every split scheme.

```bash
for s in logo logo_purged family random_debug; do vpdl-dl leakage --data $T --split $s --out runs/dl/leakage; echo "$s exit=$?"; done
```

Check: every line `exit=0` (`Critical findings: 0`). Paste the four `.md` files into `docs/dl/DATA_LEAKAGE_REPORT.md`.

---

## Phase 3 — existing models on the canonical table (the regression baseline)

Same six features as the published arms (`population` + `external_priors`).

```bash
for m in gbm mlp bilstm; do vpdl-dl train --data $T --sources $S --model $m --modalities population,external_priors --out runs/dl/main; done
```

Check: mean scoreable ROC-AUC near the RUNLOG's 0.947 / 0.963 / 0.962 (differences are explained by canonical QC — `label_changes_vs_assembled`).

GBM with early stopping (the fair cross-model comparison the RUNLOG asked for):

```bash
vpdl-dl train --data $T --sources $S --model gbm --modalities population,external_priors --model-kwargs '{"early_stopping_rounds": 50}' --tag es50 --out runs/dl/main
```

---

## Phase 4 — sequence baselines

```bash
for m in aa_mlp cnn bilstm_attn transformer; do vpdl-dl train --data $T --sources $S --model $m --modalities population,external_priors --out runs/dl/main; done
```

Sequence-only (no tabular features at all):

```bash
for m in bilstm cnn transformer; do vpdl-dl train --data $T --sources $S --model $m --modalities none --out runs/dl/seqonly; done
```

List the exact arm names, then compare every model against the existing MLP:

```bash
vpdl-dl arms --runs runs/dl/main
```

```bash
vpdl paired --runs runs/dl/main --reference train-clinvar__drop-domains-genomic-structure --reference-model mlp --cross-model
```

---

## Phase 5 — zero-shot baselines (no training)

```bash
vpdl-dl zeroshot --data $T --backbone esm2_650m --policy full --method masked_marginal --table-out data/built/canonical_zs.csv
```

```bash
vpdl-dl zeroshot --data $T --backbone esm1b --policy centered --method masked_marginal --table-out data/built/canonical_zs.csv
```

```bash
vpdl-dl zeroshot --data $T --backbone esm2_650m --policy full --method wt_marginal --table-out data/built/canonical_zs.csv
```

```bash
vpdl-dl zeroshot --data $T --backbone esm1b --policy centered --method wt_marginal --table-out data/built/canonical_zs.csv
```

```bash
vpdl paired --runs runs/dl/main --reference train-clinvar__drop-domains-genomic-structure --reference-model mlp --data data/built/canonical_zs.csv --feature-baseline zeroshot_esm2_650m_P0_masked_marginal zeroshot_esm1b_P0_masked_marginal zeroshot_esm2_650m_P0_wt_marginal zeroshot_esm1b_P0_wt_marginal
```

Check: every zero-shot arm has AUC > 0.5 (a sign error raises instead of printing).

---

## Phase 6 — embeddings and WT/VT representations

1. Extract once per backbone and context policy.

```bash
vpdl-dl embed --data $T --backbone esm2_650m --policy full
```

```bash
vpdl-dl embed --data $T --backbone esm1b --policy hierarchical
```

```bash
for p in centered asymmetric hierarchical sliding; do vpdl-dl embed --data $T --backbone esm2_650m --policy $p; done
```

Check: each prints `pointer: features/latest/<backbone>@P0-<policy>.txt`.

2. Representations, ESM-2 (the existing backbone) — frozen-PLM probe, no tabular features.

```bash
E2=$(cat features/latest/esm2_650m@P0-full.txt); for r in site.wt site.vt site.vt_minus_wt site.abs_diff site.wt_plus_vt site.concat4 local.concat4 global.concat4; do vpdl-dl train --data $T --sources $S --model mlp --modalities none --embedding-blocks "$E2:$r" --out runs/dl/probes; done
```

3. ESM-1b versus ESM-2 at the best representation (replace `site.concat4` if step 2 says otherwise).

```bash
E1=$(cat features/latest/esm1b@P0-hierarchical.txt); vpdl-dl train --data $T --sources $S --model mlp --modalities none --embedding-blocks "$E1:site.concat4" --out runs/dl/probes
```

4. MSH6 context policies.

```bash
for p in centered asymmetric hierarchical sliding; do vpdl-dl train --data $T --sources $S --model mlp --modalities none --embedding-blocks "$(cat features/latest/esm2_650m@P0-$p.txt):site.concat4" --out runs/dl/probes; done
```

---

## Phase 7 — population ablation and DL fusion

`E2` from Phase 6. Model A has no population features; Model B has them.

```bash
vpdl-dl train --data $T --sources $S --model fusion --tag concat --modalities structure,genomic,annotation --embedding-blocks "$E2:site.concat4" --out runs/dl/fusion
```

```bash
vpdl-dl train --data $T --sources $S --model fusion --tag concat --modalities population,structure,genomic,annotation --embedding-blocks "$E2:site.concat4" --out runs/dl/fusion
```

Modality matrix (single modalities, pairs, all):

```bash
for m in population structure genomic; do vpdl-dl train --data $T --sources $S --model fusion --tag concat --modalities $m --out runs/dl/fusion; done
```

```bash
for m in none structure population genomic structure,population structure,population,genomic,annotation; do vpdl-dl train --data $T --sources $S --model fusion --tag concat --modalities $m --embedding-blocks "$E2:site.concat4" --out runs/dl/fusion; done
```

Gated fusion, only after concat results exist:

```bash
vpdl-dl train --config configs/dl/fusion_dgx.toml --data $T --sources $S --model fusion --tag gated --modalities structure,population,genomic,annotation --embedding-blocks "$E2:site.concat4" --model-kwargs '{"mode": "gated"}'
```

---

## Phase 8 — PLM fine-tuning (most conservative first)

```bash
for k in frozen lora adapters last_n; do vpdl-dl train --data $T --sources $S --model plm_finetune --modalities none --tag $k --model-kwargs "{\"backbone\": \"esm2_650m\", \"strategy\": {\"kind\": \"$k\"}, \"context\": \"centered\", \"epochs\": 10, \"batch_size\": 16}" --out runs/dl/finetune; done
```

Check: each cell logs `fine-tune strategy <k>: N / M parameters trainable`. Full
fine-tuning is refused unless `"allow_full": true` is added — only with evidence
from these four.

---

## Phase 9 — pretraining P0–P4 (strict, per fold)

1. Corpus: download once, then one strict corpus per held-out gene.

```bash
vpdl-dl corpus --fetch --mode transductive --out data/dl/corpus/transductive
```

```bash
for g in MLH1 MSH2 MSH6 PMS2; do vpdl-dl corpus --fasta data/dl/corpus/raw/*.fasta --mode strict --holdout $g --out data/dl/corpus/strict-$g; done
```

Check: each prints `n_excluded_homologs` ≥ 1 (the gene itself plus orthologs).

2. P1 first (cheapest); resume any interrupted run by re-running with `--resume`.

```bash
for g in MLH1 MSH2 MSH6 PMS2; do vpdl-dl pretrain --config configs/dl/pretrain_dgx.toml --arm P1 --corpus data/dl/corpus/strict-$g --out runs/dl/pretrain/P1-$g && vpdl-dl embed --data $T --backbone esm2_650m --policy full --arm P1 --fold $g --adapted-from runs/dl/pretrain/P1-$g --blocks-file runs/dl/pretrain/P1_blocks.json; done
```

3. Downstream: P0 and P1 under the identical protocol.

```bash
vpdl-dl train --data $T --sources $S --model mlp --modalities none --embedding-blocks "$E2:site.concat4" --out runs/dl/pretrain_eval
```

```bash
vpdl-dl train --data $T --sources $S --model mlp --modalities none --embedding-blocks "perfold=runs/dl/pretrain/P1_blocks.json:site.concat4" --out runs/dl/pretrain_eval
```

```bash
vpdl-dl arms --runs runs/dl/pretrain_eval
```

Then `vpdl paired --runs runs/dl/pretrain_eval --reference <the P0 arm> --reference-model mlp`.

4. P2–P4 only if P1 is not a regression: repeat step 2 with `--arm P2` / `P3` / `P4` and `P2_blocks.json` etc., then step 3.

---

## Phase 10 — split schemes, calibration, functional validation, failure, export

1. Same models under the stricter splits.

```bash
for s in family logo_purged; do vpdl-dl train --data $T --sources $S --model mlp --modalities population,external_priors --split $s --out runs/dl/splits; vpdl-dl train --data $T --sources $S --model mlp --modalities none --embedding-blocks "$E2:site.concat4" --split $s --out runs/dl/splits; done
```

2. Calibration (fitted on inner-validation predictions only).

```bash
vpdl-dl calibrate --runs runs/dl/main --method temperature
```

3. Independent functional validation (held-out-gene scores vs the MSH2 assay / CIMRA).

```bash
vpdl-dl train --config configs/dl/fusion_dgx.toml --data $T --sources $S --model fusion --tag concat --modalities structure,population,genomic,annotation --embedding-blocks "$E2:site.concat4"
```

```bash
vpdl-dl functional --runs runs/dl/fusion --functional data/built/canonical.functional.csv
```

4. Failure analysis for any cell (name from `vpdl-dl arms`, plus `__seed42`).

```bash
vpdl-dl failure --data $T --runs runs/dl/fusion --cell <cell slug>
```

5. DL output records for the future integration.

```bash
vpdl-dl export --runs runs/dl/fusion --cell <cell slug> --export-dir runs/dl/fusion/export --out runs/dl/export
```

Check: `runs/dl/export/dl_outputs_<cell>.jsonl` + `.embeddings.npy` + `.schema.json`.

---

## What to send back

The `Check:` lines that did not match, the four leakage reports, and the
`vpdl paired` tables. Every run is also indexed in `runs/dl/registry.jsonl`.
