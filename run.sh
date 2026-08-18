#!/usr/bin/env bash
# One-command launcher: resolves the target RAG project's own venv Python
# and runs the eval loop with it -- no manual env vars or venv path typing.
# Forwards every argument straight to eval.runner, e.g.:
#   ./run.sh
#   ./run.sh --num-answerable 50 --num-unanswerable 50
#   ./run.sh --rag-root /path/to/your-project
#
# Resolution order for the target project root (same as eval/target.py):
#   1. RAG_PROJECT_ROOT environment variable, if set
#   2. ../RAG next to this repo's own folder (sibling checkout)
#
# This script deliberately does NOT check for any particular file (like
# app/config.py) inside that root -- a filename check is a proxy for "is
# this a compatible project" that breaks the moment someone's project is
# laid out differently (a flat main.py instead of an app/ package, say),
# and it can silently drift out of sync with what's actually required (this
# repo shipped exactly that bug once: this script kept checking for
# app/config.py after eval/target.py itself had already stopped requiring
# it). Real verification -- actually importing the target's embedder/
# generator modules and checking the required functions exist on them --
# happens once, in Python, via eval.target.verify_target() when eval.runner
# starts. This script's only job is finding a directory and a Python
# interpreter to hand off to.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAG_ROOT="${RAG_PROJECT_ROOT:-$(dirname "$HERE")/RAG}"

if [ ! -d "$RAG_ROOT" ]; then
    echo "'$RAG_ROOT' is not a directory." >&2
    echo "" >&2
    echo "Point at your project's root with:" >&2
    echo "  export RAG_PROJECT_ROOT=/path/to/your-project" >&2
    echo "before running this script, or pass --rag-root <path> as an argument." >&2
    exit 1
fi

VENV_PYTHON="$RAG_ROOT/.venv/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
    echo "No virtualenv found at '$VENV_PYTHON'. Set up your project's venv first (python3 -m venv .venv && .venv/bin/pip install -r requirements.txt inside it), or point RAG_PROJECT_ROOT at a project that already has one." >&2
    exit 1
fi

echo "Target project: $RAG_ROOT"
echo "Using venv:     $VENV_PYTHON"
echo ""

cd "$HERE"
exec "$VENV_PYTHON" -m eval.runner --rag-root "$RAG_ROOT" "$@"
