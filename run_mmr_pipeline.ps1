[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args,

    [string]$Python = $(if ($env:PYTHON) { $env:PYTHON } else { "python" })
)

# End-to-end Lynch-syndrome MMR pipeline wrapper for Windows PowerShell:
# download -> clean/process -> train (Phases 0-3).
#
# Usage
#   .\run_mmr_pipeline.ps1 --dry_run                    # preview the sequence
#   .\run_mmr_pipeline.ps1 --no-full_finetune           # CPU-friendly run
#   $env:FEATURES = "esm+priors"; .\run_mmr_pipeline.ps1 --all_sources   # GPU box
#
# Backbone gradient fine-tuning (stage 2b) is ON by default and needs a
# CUDA/MPS device; the pipeline stops before it on a CPU-only box unless you
# pass --allow_cpu_finetune or --no-full_finetune.
#
# All positional/flag arguments are forwarded verbatim to
# run_mmr_pipeline.py -- see `python run_mmr_pipeline.py --help` for the
# full list. Env vars (FEATURES, ESM_MODEL, SKIP_BUILD, PYTHON) are read
# directly by run_mmr_pipeline.py, same convention as run_pipeline.ps1
# (the separate, unmodified original-workflow pipeline).

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Push-Location $scriptDir
try {
    & $Python "run_mmr_pipeline.py" @Args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
