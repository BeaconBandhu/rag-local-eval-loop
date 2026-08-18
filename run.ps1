# One-command launcher: resolves the target RAG project's own venv Python
# and runs the eval loop with it -- no manual env vars or venv path typing.
# Forwards every argument straight to eval.runner, e.g.:
#   .\run.ps1
#   .\run.ps1 --num-answerable 50 --num-unanswerable 50
#   .\run.ps1 --rag-root D:\path\to\RAG
#
# Resolution order for the target project root (same as eval/target.py):
#   1. RAG_PROJECT_ROOT environment variable, if set
#   2. ..\RAG next to this repo's own folder (sibling checkout)
# This script only needs to find that root's venv Python -- eval/target.py
# does the actual sys.path injection once the run starts.

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

if ($env:RAG_PROJECT_ROOT) {
    $ragRoot = $env:RAG_PROJECT_ROOT
} else {
    $ragRoot = Join-Path (Split-Path -Parent $here) "RAG"
}

if (-not (Test-Path (Join-Path $ragRoot "app\config.py"))) {
    Write-Error @"
'$ragRoot' doesn't look like the RAG project (no app\config.py found there).

Point at it with:
  `$env:RAG_PROJECT_ROOT = "C:\path\to\RAG"
before running this script, or pass --rag-root <path> as an argument.
"@
    exit 1
}

$venvPython = Join-Path $ragRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "No virtualenv found at '$venvPython'. Set up the target project's venv first (see its own README), or point RAG_PROJECT_ROOT at a project that has one."
    exit 1
}

Write-Host "Target project: $ragRoot"
Write-Host "Using venv:     $venvPython"
Write-Host ""

Push-Location $here
try {
    & $venvPython -m eval.runner @args
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
