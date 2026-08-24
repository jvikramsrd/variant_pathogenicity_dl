#!/usr/bin/env bash
# End-to-end pipeline wrapper for macOS/Linux.
#
# Usage
#   bash run_pipeline.sh
#   FEATURES="esm+priors" bash run_pipeline.sh
#   SKIP_BUILD=1 bash run_pipeline.sh
#
# The pipeline implementation lives in run_pipeline.py so path handling and
# virtualenv discovery are shared with the Windows PowerShell wrapper.
set -euo pipefail

cd "$(dirname "$0")"

if [ -n "${PYTHON:-}" ]; then
    PY="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
else
    PY="python"
fi

exec "$PY" run_pipeline.py "$@"
