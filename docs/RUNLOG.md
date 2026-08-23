# Running Log

Append one entry per build / training run. Newest at the top.
Format: date · what ran (command) · outcome · artifacts.

---

## 2026-08-23

- **Leakage-clean full training** — `scripts/train_extended.py --no_dms_features`
  - 80 genes, 190,494 labelled rows, group-disjoint CV, DMS-derived features removed
  - Train-vs-val AUC gap ≈ 0.000 → no classic overfitting
  - Clinical-only slice: **ROC-AUC 0.963 / MCC 0.78** (AM baseline: 0.945 / 0.75)
  - All-labels: 0.716 (honest number; earlier 0.9987 was circular via `dms_bin_median`)
  - Artifacts: `data/processed/extended_train/ext80_*`, checkpoints

- **Overfitting diagnostic added** — `scripts/diagnose_overfitting.py`
  - Proved `dms_bin_median` == flipped label on 97.3% of rows (target leakage)

- **Label-inversion bug fixed** in `src/extended_builder.py`
  - ProteinGym `DMS_score_bin=1` = top fitness half (tolerated); builder had
    mapped it straight to "pathogenic" → ~185k labels were inverted
  - Verified vs AlphaMissense anchor + PG-clinical anchor; dataset rebuilt, re-audited 12/12

- **Expanded dataset build** — `make_expanded_panel.py` + `build_extended_dataset.py --panel_file`
  - Panel 10 → 80 genes; master table 110,124 → 1,156,625 rows
  - 50,304 isoform-mismatched DMS + 152 AlphaMissense rows dropped by wt-validation
  - Audit: 12/12 passed (`audit_report.json`); ClinVar↔PG-clinical agree on all 1,919 overlaps
  - Train CSV: `extended_dataset_train.csv` (190,494 rows: 82,149 P/LP · 108,345 B/LB)

## TODO / next runs

- [ ] GPU box: install `requirements-cuda.txt`, verify `torch.cuda.is_available()`
- [ ] Full model: `train_extended.py --features esm+priors --esm_model facebook/esm2_t33_650M_UR50D --no_dms_features`
- [ ] Leave-one-protein-out CV for transfer estimate
- [ ] Score VUS risk tiers from ESM-mode run (`ext80_vus_predictions.csv`)
