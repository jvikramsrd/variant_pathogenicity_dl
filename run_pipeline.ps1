[CmdletBinding()]
param(
    [ValidateSet("priors", "esm+priors")]
    [string]$Features = $(if ($env:FEATURES) { $env:FEATURES } else { "priors" }),

    [string]$EsmModel = $(if ($env:ESM_MODEL) { $env:ESM_MODEL } else { "facebook/esm2_t33_650M_UR50D" }),

    [int]$KFolds = $(if ($env:K_FOLDS) { [int]$env:K_FOLDS } else { 5 }),

    [switch]$SkipBuild,

    [string]$Python = $(if ($env:PYTHON) { $env:PYTHON } else { "python" })
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$runnerArgs = @(
    "run_pipeline.py",
    "--features", $Features,
    "--esm-model", $EsmModel,
    "--k-folds", $KFolds
)

if ($SkipBuild) {
    $runnerArgs += "--skip-build"
}

Push-Location $scriptDir
try {
    & $Python @runnerArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
