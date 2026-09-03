# GPU run playbook — Stage-2b ablation grid

Operator's document for the **Windows** CUDA box (~15 GiB card). The dev box is
CPU-only, so everything from §2 onward runs on that machine and the artefacts
come back through git.

Design rationale: `docs/superpowers/specs/2026-09-02-accuracy-ablation-paper-design.md`.
Grid definition in code: `src/finetune_grid.py`.

**Budget:** ~2 h setup and data rebuild, then ~31 GPU-hours in two sittings.

---

## 0. Why this grid, in one paragraph

`docs/PAPER_DRAFT.md` §6.12 proposed, until 2026-09-02, a freeze-depth × PLLR
ablation and a decision rule: if the frozen floor matches the full fine-tune,
backbone gradients bought nothing. That comparison could not have settled the
question, because Stage 2b read **none** of the prior columns the frozen probe
reads — so the measured 6.5-point gap (0.880 vs 0.945) confounded freeze depth
with feature set. This grid adds a **branch** axis (`esm` vs `esm+priors`) so
freeze depth is measured with the feature set held constant. §6.12 and
`PAPER.md` §7.2/Table 6 describe the grid below.

---

## 1. Prerequisites

PowerShell on the CUDA box. On a Linux GPU box substitute `.venv/bin/python`,
forward slashes, and `\` line continuations; everything else is identical.

```powershell
nvidia-smi                 # the card, and ~15 GiB
git --version
py -0p                     # must list 3.12
```

**Python 3.12, not 3.14.** The cu126 wheels publish no 3.14 build and pip falls
back to the CPU wheel *without failing* — which surfaces an hour into a run as a
job that never touches the GPU.

Free disk: **~10 GB** — ~1.7 GB raw downloads (ClinVar 442 MB, AlphaMissense
1.21 GB, ProteinGym ~60 MB), ~1.5 GB processed tables, ~2.5 GB Hugging Face
cache for the 650M checkpoint, plus headroom. The rebuild also calls the
Ensembl, UniProt, InterPro, AlphaFold and gnomAD REST APIs, so it needs an
unrestricted connection.

---

## 2. Get the code

```powershell
cd $HOME
git -c core.autocrlf=false clone https://github.com/jvikramsrd/variant_pathogenicity_dl.git
cd variant_pathogenicity_dl
```

Stay on `main`; there is no feature branch to check out.

`core.autocrlf=false` matters: Git for Windows rewrites text files to CRLF on
checkout by default, and this repo carries no `.gitattributes`. With autocrlf on,
every CSV that passes through git changes bytes, so hash checks fail and diffs
against the dev box are meaningless.

---

## 3. Environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements-cuda.txt
.\.venv\Scripts\pip.exe install pytest        # not in requirements-cuda.txt
```

Verify CUDA before anything else. **If this prints `False`, stop** — the grid
would run on CPU and take weeks.

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 4. Rebuild the datasets

`data/mmr/processed/**` and `data/processed/**` are gitignored, so the tables are
not in the clone. This section rebuilds them from source with **gnomAD included**
on both panels.

Read §4.5 before running: a rebuild changes the data under the paper's existing
numbers, and the comparator has to move with it.

### 4.1 Derive the PMS2 homology range

```powershell
.\.venv\Scripts\python.exe scripts\derive_pms2_homology_range.py
```

Expect **382 862**. The script derives the exon 11–15 span from Ensembl's exon
table for MANE Select ENST00000265849 and self-validates against the 862 aa
pinned for UniProt P54278, failing loudly if the transcript moved. Use whatever
it prints in §4.3 rather than the remembered number.

### 4.2 Broad 80-gene panel (~40–60 min)

```powershell
.\.venv\Scripts\python.exe scripts\make_expanded_panel.py
.\.venv\Scripts\python.exe scripts\build_extended_dataset.py --panel_file data\raw\uniprot\expanded_panel.json --min_stars 2 --all_sources
.\.venv\Scripts\python.exe scripts\audit_extended_dataset.py
```

`--all_sources` = `--include_gnomad --include_structure --include_interpro
--include_functional_sites`. Most of the wall clock is the one-off AlphaMissense
scan (216,175,352 lines, ~7 min), the downloads, and 80 gnomAD GraphQL
round-trips. Reference from the 2026-08-23 build: **1,156,625 master rows × 68
columns**, audit **12/12**. If the audit fails, stop and read
`data/processed/extended/audit_report.json`.

The Stage-2b grid does **not** read this panel — it is needed for the
warm-started probe (§4.4, optional) and the paper's broad-panel results. Skip
this step if you only want the grid.

### 4.3 MMR panel with gnomAD (~10 min)

```powershell
.\.venv\Scripts\python.exe scripts\build_mmr_dataset.py --min_stars 2 --pms2_codon_range 382 862
```

No `--skip_gnomad`: stage 3 joins variant-level allele frequencies and stage 4
fetches gene-level constraint. Both cache under `data\raw\gnomad\`
(`<GENE>_gnomad_v4.csv`, `<GENE>_constraint.json`); the client retries 5 times
with backoff on a 120 s timeout, so if the API flakes, re-run the identical
command and it resumes from cache.

The PMS2 gate is fail-closed — without `--pms2_codon_range` the gene is dropped
entirely rather than silently trusted.

### 4.4 Verify the build

The gnomAD failure mode is silent: `--skip_gnomad` fills the columns with NaN/0
rather than omitting them, so "the columns exist" proves nothing. The previous
build shipped `gnomad_rows_panel: 0` for exactly that reason.

```powershell
.\.venv\Scripts\python.exe -c "import pandas as pd; d=pd.read_csv(r'data\mmr\processed\extended\extended_dataset.csv',low_memory=False); c=d[d.label_source.isin(['clinvar','pg_clinical'])]; print('rows',len(d)); print(c.groupby(['gene','pms2_homology_excluded']).size()); print('AF joined:', d.gnomad_log10_af.notna().sum(), 'of', len(d)); print('constraint:', d[['gnomad_pli','gnomad_oe_lof']].notna().sum().to_dict())"
```

Check, in order:

- `AF joined` is **well above zero**, and `manifest.json → stats.gnomad_rows_panel`
  is non-zero with `gnomad_genes_failed` empty. This is the point of the rebuild.
- Row count and clinical split. Reference from the table the paper currently
  quotes: **74,328 rows**; clinical **208 MLH1 / 335 MSH2 / 119 MSH6 / 21 PMS2**
  = 683, with PMS2 also showing 21 rows excluded by the homology gate.
- **Record whatever you actually get.** ClinVar grows weekly, so a modest
  increase is expected and is not an error — but the new count replaces 683
  everywhere in the paper, and goes in `RUNLOG.md`.

### 4.5 Regenerate the comparator (~1 min) — not optional

Table 6's comparator row (frozen priors probe, mean ROC-AUC 0.945 over
MLH1/MSH2/MSH6) was scored on the *old* table without gnomAD. The rebuild
changes both the label snapshot and the prior feature set, so that number no
longer describes the data the grid will run on.

```powershell
.\.venv\Scripts\python.exe scripts\run_mmr_transfer.py --scratch --features priors --eval lopo --n_bootstrap 10000 --out_dir data\processed\mmr_transfer_scratch
```

That is exactly the run behind the 0.945 row (MLH1 0.964 / MSH2 0.901 / MSH6
0.969 / PMS2 1.000). Two things to confirm in
`mmr_transfer_summary_lopo.json`:

- `gene_constant_priors_dropped: true` — see §7.
- `n_bootstrap: 10000`. The archived run used 2,000 while the paper's protocol
  section claims 10,000; this command fixes that discrepancy.

Optional, for the warm-started probe as well (needs §4.2):

```powershell
.\.venv\Scripts\python.exe run_mmr_pipeline.py --no-full_finetune --features priors --eval lopo --min_stars 2 --pms2_codon_range 382 862 --skip_build --n_bootstrap 10000
```

`--skip_build` reuses what §4.2–4.3 built; `--no-full_finetune` stops before
Stage 2b so it does not collide with the grid.

### 4.6 Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests\ -q      # expect 149 passed
```

Failures here are far cheaper to fix than fifteen hours into a run.

---

## 5. Run the grid

Resumable by design: re-running skips cells that already have all three
artefacts, so an interruption is continued by repeating the identical command,
not restarted. Every artefact is cell-tagged, so the old "move the results CSV
aside between cells" step is gone.

### 5.1 Dry run

```powershell
.\.venv\Scripts\python.exe scripts\run_stage2b_grid.py `
  --tiers 1 2 3 4 5 `
  --esm_model facebook/esm2_t33_650M_UR50D `
  --mode siamese --eval lopo `
  --batch_size 1 --grad_accum 8 --gradient_checkpointing `
  --epochs 10 --n_bootstrap 10000 `
  --out_dir data\processed\stage2b_grid `
  --dry_run
```

Prints 16 cells in 5 tiers and one command line per cell. Runs nothing.

### 5.2 Stop the machine sleeping

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
# on a laptop, the same two with -dc
```

A suspend mid-cell loses that cell — the driver resumes at cell granularity,
not mid-training.

### 5.3 Tier 1 first (~15.75 h, 6 cells)

```powershell
$env:PYTHONUNBUFFERED = "1"
# $env:HF_HOME = "D:\hf_cache"    # if C: is tight; the 650M checkpoint is ~2.5 GB

Start-Transcript -Path data\processed\stage2b_grid\grid_tier1.log
.\.venv\Scripts\python.exe scripts\run_stage2b_grid.py `
  --tiers 1 `
  --esm_model facebook/esm2_t33_650M_UR50D `
  --mode siamese --eval lopo `
  --batch_size 1 --grad_accum 8 --gradient_checkpointing `
  --epochs 10 --n_bootstrap 10000 `
  --out_dir data\processed\stage2b_grid
Stop-Transcript
```

`Start-Transcript` rather than `2>&1 | Tee-Object`: the driver logs to stderr,
which PowerShell 5.1 renders as red ErrorRecords through a pipe and writes as
UTF-16.

`--batch_size 1 --grad_accum 8 --gradient_checkpointing` is what makes a 650M
siamese full fine-tune fit ~10 GiB on a 15 GiB card while keeping ProPath's
effective batch of 8. Below ~11 GiB a full fine-tune is impossible at all: the
static AdamW state alone is ~9.7 GiB.

Check before continuing:

```powershell
Get-Content data\processed\stage2b_grid\stage2b_grid_manifest.json
Get-ChildItem data\processed\stage2b_grid\*.csv | Select-Object Name, Length
```

Six cells complete, zero failed, three artefacts each.

### 5.4 The remaining tiers (~15 h)

```powershell
Start-Transcript -Path data\processed\stage2b_grid\grid_tiers2to5.log
.\.venv\Scripts\python.exe scripts\run_stage2b_grid.py `
  --tiers 1 2 3 4 5 `
  --esm_model facebook/esm2_t33_650M_UR50D `
  --mode siamese --eval lopo `
  --batch_size 1 --grad_accum 8 --gradient_checkpointing `
  --epochs 10 --n_bootstrap 10000 `
  --out_dir data\processed\stage2b_grid
Stop-Transcript
```

Tier 1 is listed again deliberately — those six are reported complete and
skipped. This is also the exact command to repeat after any crash, OOM or
reboot.

### 5.5 Wall clock

| Tier | Cells | What it answers | Est. |
|---|---|---|---|
| 1 | 6 | **The headline.** Does ESM add anything on top of published priors, on unseen genes? 3 seeds × {full FT, frozen floor}, both `esm+priors` | ~15.75 h |
| 2 | 2 | Branch attribution: how much of any gain is the priors branch vs the backbone? | ~5.25 h |
| 3 | 5 | The PLLR axis, measured at the frozen floor where the backbone cannot relearn the term. Includes the GateWave-fusion cell | ~1 h |
| 4 | 2 | Freeze-depth middle ground (`--n_unfrozen_layers 2`) | ~4 h |
| 5 | 1 | PLLR off at full fine-tune | ~5 h |
| | **16** | | **~31 h** |

Tiers are ordered by scientific value, so an interruption degrades gracefully.
**Tier 1 alone settles the paper's headline claim.** If time runs short, stop at
a tier boundary rather than part-way through tier 1's seeds.

Frozen (`n_unfrozen_layers 0`) cells are cheap because the backbone output is
constant across epochs and is encoded once — that is why every floor cell can
carry three seeds.

### 5.6 If a cell fails

The driver logs the failure, continues to the next cell, and prints the failed
slugs at the end; one OOM does not cost the remaining tiers. Retry by re-running
the §5.4 command.

If a full-unfreeze cell OOMs repeatedly, the knobs in order of preference:

1. `--grad_accum 16` with `--batch_size 1`
2. Drop the cell's `--epochs` — early stopping usually beats the cap anyway
3. Skip tier 5 and the full-unfreeze cell of tier 2; they are the least
   load-bearing of the full-unfreeze cells

Do **not** raise `--batch_size` above 1 for a 650M full unfreeze on 15 GiB.

---

## 6. Send the results back

### 6.1 Capture provenance

```powershell
nvidia-smi > data\processed\stage2b_grid\nvidia_smi.txt
.\.venv\Scripts\pip.exe freeze > data\processed\stage2b_grid\pip_freeze.txt
```

`PAPER.md` §6.10 flags the Stage-2b environment as unverified provenance; these
two files close that.

### 6.2 Commit and push

Everything worth returning is small text. Most of it is gitignored on purpose,
so the `-f` list below is not optional:

```powershell
git checkout -b results/stage2b-grid
git add data\processed\stage2b_grid\esm_finetune_results_*.csv `
        data\processed\stage2b_grid\esm_finetune_summary_*.json `
        data\mmr\processed\extended\manifest.json `
        data\processed\mmr_transfer_scratch\mmr_transfer_results_lopo.csv `
        data\processed\mmr_transfer_scratch\mmr_transfer_summary_lopo.json
git add -f data\processed\stage2b_grid\esm_finetune_predictions_*.csv `
           data\processed\stage2b_grid\stage2b_grid_results.csv `
           data\processed\stage2b_grid\stage2b_grid_manifest.json `
           data\processed\stage2b_grid\nvidia_smi.txt `
           data\processed\stage2b_grid\pip_freeze.txt
git status      # confirm 16 cells x 3 artefacts before committing
git commit -m "results: Stage-2b ablation grid, 16 cells, 650M on CUDA"
git push -u origin results/stage2b-grid
```

Why each `-f` is needed:

| File | Why git ignores it |
|---|---|
| `esm_finetune_predictions_*.csv` | explicit `*_predictions_*.csv` rule |
| `stage2b_grid_results.csv` | no trailing `_`, so it misses the `*_results_*` exception |
| `stage2b_grid_manifest.json` | the exception matches only bare `manifest.json` |
| `nvidia_smi.txt`, `pip_freeze.txt` | no exception covers them |

`grid_tier1.log` is ignored by `*.log` — leave it out, transcripts are noisy.
First push prompts for GitHub credentials; `gh auth login` beforehand if the
Credential Manager is not already set up.

**The predictions CSVs are the important ones.** Metrics alone cannot be
seed-ensembled, recalibrated, or analysed per-gene after the fact; the previous
Stage-2b runs returned only metrics, which is why Run 1 and Run 2 cannot be
compared at the variant level.

### 6.3 On the dev box

```bash
git fetch origin && git checkout results/stage2b-grid
```

---

## 7. gnomAD: three things to get right

Including gnomAD (§4.2–4.3) is legitimate here, but it has consequences.

**Do not pass `--af_labels_active` to the grid.** That flag exists to enforce the
AF quarantine when allele-frequency-minted benign labels are in the training
pool: it forces the four AF-derived columns (`gnomad_log10_af`, `acmg_ba1`,
`acmg_bs1`, `acmg_pm2`) out of the feature set. This run's labels come from
ClinVar and PG-clinical only, so the quarantine is not triggered and the flag
would discard exactly what §4.3 just added. `src.transfer.assert_af_quarantine`
refuses to run if both are ever enabled at once.

**Confirm the five gene-level constraint columns are dropped under LOGO.**
`gnomad_pli`, `oe_lof`, `oe_mis`, `mis_z` and `syn_z` are constant within a gene,
so across genes they are a 5-dimensional gene identifier rather than variant
evidence; `RUNLOG.md` 2026-08-28 records MLH1 collapsing to ROC-AUC 0.500 in
every seed when they were kept. `--gene_constant_priors auto` resolves to `drop`
under `--eval lopo`, and the grid path uses
`prior_columns_of(..., drop_gene_constant=True)`. Verify, don't assume: the
summary JSON reports `gene_constant_priors_dropped`.

**Add a limitations line to the paper.** ClinVar benign classifications
frequently invoke BS1/BA1 — allele frequency — as evidence, so AF features
partially encode the evidence used to produce the labels. That is soft
circularity of Grimm's type 2 and should be stated plainly. It is *not* the hard
`acmg_bs1 == label` identity of Finding 2, which arises only when the labels are
themselves minted from AF.

---

## 8. Pre-registered reading of the results

Fixed **before** the numbers land, and not revised afterwards.

- If the **frozen floor** (`n_unfrozen_layers 0`) mean sits inside the 95% CI of
  the **full fine-tune** at the same branch, backbone gradients bought nothing
  measurable at this label budget (277–530 training rows per split). Report the
  frozen probe as the model and the fine-tune as a negative result.
- If **`esm+priors` does not beat `priors`-only** — the frozen Stage-2 probe as
  recomputed in §4.5, *not* the historical 0.945 — then ESM adds nothing on top
  of published priors for unseen-gene MMR prediction. That is a publishable
  negative result and the current best guess.
- If **`n_unfrozen_layers 2` matches `-1`** at a fraction of the compute,
  recommend the partial unfreeze and quote the VRAM saving.
- The **PLLR-on minus PLLR-off gap at the frozen floor** is the clean
  measurement of what the 2026-08-28 PLLR fix is worth. At full unfreeze the
  same gap is confounded with the backbone relearning the term.
- Holdouts are small (119–335; PMS2 21). **If the seed sd exceeds the
  between-cell differences, the ablation is underpowered — say so and do not
  rank cells.**
- PMS2's held-out rows stay out of every headline mean.
- Report each cell as mean ± sd over the seeds available for it.

**Historical rows in Table 6 are no longer comparable.** §4.7 Run 2 (mean 0.880)
and the old 0.945 probe were computed on the pre-rebuild table without gnomAD.
Mark both as historical, quote the §4.5 comparator as the live one, and do not
compute a delta across the rebuild boundary.

---

## 9. What is *not* in this grid, and why

**Label expansion** — gnomAD-AF-minted benign labels, 1-star ClinVar — stays out
of this run and lands afterwards as a single before/after delta on the winning
cell. Changing the labels, the model and the training recipe at once and
reporting the aggregate is precisely the practice this paper criticises. Note
the distinction from §4.3: joining gnomAD as *features* is a feature-set change
measured on both sides of the comparison (§4.5 regenerates the comparator on the
same table); minting *labels* from allele frequency is a supervision change and
triggers the quarantine in §7.

**CPU-only experiments** — leave-one-protein-out on the 80-gene panel, seed
averaging, the temporal ClinVar holdout — run on the dev box and consume none of
this budget.
