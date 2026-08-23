#!/usr/bin/env bash
# ===========================================================================
# End-to-end pipeline for variant pathogenicity prediction.
#
# Running this single script from a fresh clone regenerates EVERYTHING
# (datasets, audit, train table, model, metrics). All stages are cached:
# re-running skips work that is already on disk.
#
# Usage
#   bash run_pipeline.sh                      # priors-only model (CPU OK)
#   FEATURES="esm+priors" bash run_pipeline.sh # full ESM-2 model (GPU advised)
#   ESM_MODEL=facebook/esm2_t12_35M_UR50D ...  # override checkpoint
#   SKIP_BUILD=1 bash run_pipeline.sh          # reuse existing dataset
#
# Requirements: python3 + venv; internet access for first run (ClinVar,
# ProteinGym, AlphaMissense downloads ~1.7 GB total).
# ===========================================================================
set -euo pipefail

PY="${PYTHON:-python3}"
FEATURES="${FEATURES:-priors}"
ESM_MODEL="${ESM_MODEL:-facebook/esm2_t33_650M_UR50D}"
K_FOLDS="${K_FOLDS:-5}"

cd "$(dirname "$0")"

banner() { printf '\n\033[1;34m==== %s ====\033[0m\n' "$1"; }

# --- 0. Environment --------------------------------------------------------
if [ ! -d .venv ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    banner "Stage 0/4 · creating virtualenv + installing requirements"
    "$PY" -m venv .venv
    if command -v nvidia-smi >/dev/null 2>&1; then
        .venv/bin/pip install -r requirements-cuda.txt      # NVIDIA GPU box
    else
        .venv/bin/pip install -r requirements.txt           # CPU fallback
    fi
fi
PYBIN=".venv/bin/python"
[ -n "${VIRTUAL_ENV:-}" ] && PYBIN="python"

banner "Stage 1/4 · unit tests"
"$PYBIN" tests/test_datasets.py

# --- 2. Expanded gene panel (all human ProteinGym DMS proteins) -------------
PANEL="data/raw/uniprot/expanded_panel.json"
if [ ! -f "$PANEL" ]; then
    banner "Stage 2a · resolving expanded panel (UniProt REST)"
    "$PYBIN" scripts/make_expanded_panel.py
else
    echo "Panel cache present: $PANEL"
fi

# --- 2b. Build the unified multi-source dataset ----------------------------
BUILD_ARGS=(--panel_file "$PANEL")
if [ -f data/processed/extended/extended_dataset.csv ] && [ -n "${SKIP_BUILD:-}" ]; then
    echo "SKIP_BUILD set -> reusing existing extended_dataset.csv"
else
    banner "Stage 2b · building extended dataset (downloads cached)"
    "$PYBIN" scripts/build_extended_dataset.py "${BUILD_ARGS[@]}"
fi

# --- 3. Independent integrity audit + train-table emission ------------------
banner "Stage 3/4 · auditing merge integrity (12 checks) + emitting train CSV"
"$PYBIN" scripts/audit_extended_dataset.py

# --- 4. Train + evaluate ----------------------------------------------------
banner "Stage 4/4 · training MLP head ($FEATURES mode, k=$K_FOLDS)"
TRAIN_ARGS=("--k_folds" "$K_FOLDS" "--no_dms_features")
if [ "$FEATURES" = "esm+priors" ]; then
    TRAIN_ARGS+=("--features" "esm+priors" "--esm_model" "$ESM_MODEL")
fi
"$PYBIN" scripts/train_extended.py "${TRAIN_ARGS[@]}"

banner "DONE"
echo "Dataset   : data/processed/extended/extended_dataset.csv (+ _train.csv)"
echo "Audit     : data/processed/extended/audit_report.json"
echo "Model/mets: data/processed/extended_train/"
echo "Run log   : append a dated entry to docs/RUNLOG.md"
