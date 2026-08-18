# One-command launcher: resolves the target RAG project's own venv Python
# and runs the eval loop with it -- no manual env vars or venv path typing.
# Forwards every argument straight to eval.runner, e.g.:
#   .\run.ps1
#   .\run.ps1 --num-answerable 50 --num-unanswerable 50
#   .\run.ps1 --rag-root D:\path\to\your-project
#
# Resolution order for the target project root (same as eval/target.py):
#   1. RAG_PROJECT_ROOT environment variable, if set
#   2. ..\RAG next to this repo's own folder (sibling checkout)
#
# This script deliberately does NOT check for any particular file (like
# app\config.py) inside that root -- a filename check is a proxy for "is
# this a compatible project" that breaks the moment someone's project is
# laid out differently (a flat main.py instead of an app\ package, say),
# and it can silently drift out of sync with what's actually required
# (this repo shipped exactly that bug once: this script kept checking for
# app\config.py after eval/target.py itself had already stopped requiring
# it). Real verification -- actually importing the target's embedder/
# generator modules and checking the required functions exist on them --
# happens once, in Python, via eval.target.verify_target() when eval.runner
# starts. This script's only job is finding a directory and a Python
# interpreter to hand off to.

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

if ($env:RAG_PROJECT_ROOT) {
    $ragRoot = $env:RAG_PROJECT_ROOT
} else {
    $ragRoot = Join-Path (Split-Path -Parent $here) "RAG"
}

if (-not (Test-Path $ragRoot -PathType Container)) {
    Write-Error @"
'$ragRoot' is not a directory.

Point at your project's root with:
  `$env:RAG_PROJECT_ROOT = "C:\path\to\your-project"
before running this script, or pass --rag-root <path> as an argument.
"@
    exit 1
}

$venvPython = Join-Path $ragRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "No virtualenv found at '$venvPython'. Set up your project's venv first (python -m venv .venv, then .venv\Scripts\python.exe -m pip install -r requirements.txt inside it), or point RAG_PROJECT_ROOT at a project that already has one."
    exit 1
}

Write-Host "Target project: $ragRoot"
Write-Host "Using venv:     $venvPython"
Write-Host ""

Push-Location $here
try {
    & $venvPython -m eval.runner --rag-root $ragRoot @args
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
