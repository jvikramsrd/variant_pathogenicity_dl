#!/usr/bin/env bash
# End-to-end Lynch-syndrome MMR pipeline wrapper for macOS/Linux:
# download -> clean/process -> train (Phases 0-3).
#
# Usage
#   bash run_mmr_pipeline.sh --dry_run                  # preview the sequence
#   bash run_mmr_pipeline.sh --no-full_finetune         # CPU-friendly run
#   FEATURES="esm+priors" bash run_mmr_pipeline.sh --all_sources   # GPU box
#   SKIP_BUILD=1 bash run_mmr_pipeline.sh               # re-run training only
#
# Backbone gradient fine-tuning (stage 2b) is ON by default and needs a
# CUDA/MPS device; the pipeline stops before it on a CPU-only box unless you
# pass --allow_cpu_finetune or --no-full_finetune.
#
# All flags are forwarded verbatim to run_mmr_pipeline.py -- see
# `python run_mmr_pipeline.py --help` for the full list.
#
# The pipeline implementation lives in run_mmr_pipeline.py so path handling
# and virtualenv discovery are shared with the Windows PowerShell wrapper
# and with run_pipeline.py/.ps1 (the separate, unmodified original-workflow
# pipeline).
set -euo pipefail

cd "$(dirname "$0")"

if [ -n "${PYTHON:-}" ]; then
    PY="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
else
    PY="python"
fi

exec "$PY" run_mmr_pipeline.py "$@"
