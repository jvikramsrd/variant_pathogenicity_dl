#!/usr/bin/env bash
# DL ablation grid on the DGX: every data source alone and combined, against every model.
# Runbook: docs/dl/ABLATION.md.
#
#   bash scripts/dl_ablation.sh dry     # check every arm (table, folds, leakage gate, models); trains nothing
#   bash scripts/dl_ablation.sh run     # train the grid (3 seeds), resumable, then rebuild the paper tables
#
# Evaluation never changes: leave-one-gene-out, scored on ClinVar labels of the held-out gene.
# Axis A varies the FEATURE sources (training labels = ClinVar).
# Axis B varies the TRAINING-LABEL sources (all features).
set -euo pipefail

MODE=${1:-dry}
PY=${PY:-python}
T=${T:-data/built/canonical_full.csv}
S="clinvar pg_dms alphamissense gnomad"          # what the table was built from
SEEDS=${SEEDS:-"42 43 44"}
JOBS=${JOBS:-6}
TAB="gbm mlp"                                     # feature-only models
SEQ="aa_mlp cnn bilstm bilstm_attn transformer"   # sequence-window models (+ any features given)
ALL="population,structure,genomic,annotation,external_priors"
LOG=${LOG:-runs/dl/ablation_$(date +%Y%m%d-%H%M%S).log}

mkdir -p runs/dl
case "$MODE" in dry|run) ;; *) echo "usage: $0 dry|run" >&2; exit 2 ;; esac

cell() {   # cell <label> <out dir> <vpdl-dl train options...>
    local label=$1 out=$2; shift 2
    if [ "$MODE" = dry ]; then
        if $PY -m vpdl.dl train --data "$T" --sources $S --seeds 42 --out "$out" --dry-run "$@" \
                >> "$LOG" 2>&1; then
            echo "ok    $label"
        else
            echo "FAIL  $label  (see $LOG)"; exit 1
        fi
    else
        echo "== $label" | tee -a "$LOG"
        set +e
        $PY -m vpdl.dl train --data "$T" --sources $S --seeds $SEEDS --jobs "$JOBS" --skip-existing \
            --out "$out" "$@" 2>&1 | tee -a "$LOG" | grep -E "^(done|FAILED|[0-9]+ cells)|skip-existing"
        local status=${PIPESTATUS[0]}
        set -e
        if [ "$status" -ne 0 ]; then
            echo "'$label' stopped with exit $status — see $LOG. Re-run the same command to resume." >&2
            exit 1
        fi
    fi
}

A=runs/dl/ablation_features
# A: one feature source at a time, then all together (training labels: ClinVar).
cell "A0 sequence only (no tabular features)"   $A --train-sources clinvar --modalities none            --model $SEQ
cell "A1 gnomAD population (+ ACMG AF flags)"   $A --train-sources clinvar --modalities population      --model $TAB $SEQ
# AlphaMissense carries allele-frequency signal, so "AlphaMissense without gnomAD" is a joint
# arm by the code's own rule; --allow-proxy-leak records that in the summary.
cell "A2 AlphaMissense"                         $A --train-sources clinvar --modalities external_priors --allow-proxy-leak --model $TAB $SEQ
cell "A3 AlphaFold structure"                   $A --train-sources clinvar --modalities structure       --model $TAB $SEQ
cell "A4 genomic / exon structure"              $A --train-sources clinvar --modalities genomic         --model $TAB $SEQ
cell "A5 all feature sources combined"          $A --train-sources clinvar --modalities "$ALL"          --model $TAB $SEQ
# (gnomAD + AlphaMissense together is runs/dl/main — already run, 3 seeds.)

B=runs/dl/ablation_labels
# B: which labels train the model (all features; always scored on ClinVar, held-out gene).
cell "B1 DMS labels only (MSH2 only: MSH2 fold is untrainable by design)" \
                                                $B --train-sources pg_dms          --modalities "$ALL" --model $TAB $SEQ
cell "B2 ClinVar + DMS combined"                $B --train-sources clinvar pg_dms  --modalities "$ALL" --model $TAB $SEQ
cell "B3 ClinVar expert-panel labels only"      $B --train-sources clinvar_expert  --modalities "$ALL" --model $TAB $SEQ

if [ "$MODE" = dry ]; then
    echo "all arms check out. Train with: bash $0 run"
    exit 0
fi

echo "== paper tables" | tee -a "$LOG"
$PY -m vpdl.dl results --runs runs/dl --out results/dl 2>&1 | tee -a "$LOG" | tail -15
echo "log: $LOG"
