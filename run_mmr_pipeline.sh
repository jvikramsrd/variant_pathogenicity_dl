#!/usr/bin/env bash
# End-to-end Lynch-syndrome MMR pipeline wrapper for macOS/Linux:
# download -> clean/process -> train (Phases 0-3).
#
# Usage
#   bash run_mmr_pipeline.sh
#   bash run_mmr_pipeline.sh --dry_run                 # preview the command sequence
#   FEATURES="esm+priors" bash run_mmr_pipeline.sh --all_sources --full_finetune
#   SKIP_BUILD=1 bash run_mmr_pipeline.sh               # re-run training only
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
