#!/usr/bin/env bash
# One-command launcher: resolves the target RAG project's own venv Python
# and runs the eval loop with it -- no manual env vars or venv path typing.
# Forwards every argument straight to eval.runner, e.g.:
#   ./run.sh
#   ./run.sh --num-answerable 50 --num-unanswerable 50
#   ./run.sh --rag-root /path/to/RAG
#
# Resolution order for the target project root (same as eval/target.py):
#   1. RAG_PROJECT_ROOT environment variable, if set
#   2. ../RAG next to this repo's own folder (sibling checkout)
# This script only needs to find that root's venv Python -- eval/target.py
# does the actual sys.path injection once the run starts.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAG_ROOT="${RAG_PROJECT_ROOT:-$(dirname "$HERE")/RAG}"

if [ ! -f "$RAG_ROOT/app/config.py" ]; then
    echo "'$RAG_ROOT' doesn't look like the RAG project (no app/config.py found there)." >&2
    echo "" >&2
    echo "Point at it with:" >&2
    echo "  export RAG_PROJECT_ROOT=/path/to/RAG" >&2
    echo "before running this script, or pass --rag-root <path> as an argument." >&2
    exit 1
fi

VENV_PYTHON="$RAG_ROOT/.venv/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
    echo "No virtualenv found at '$VENV_PYTHON'. Set up the target project's venv first (see its own README), or point RAG_PROJECT_ROOT at a project that has one." >&2
    exit 1
fi

echo "Target project: $RAG_ROOT"
echo "Using venv:     $VENV_PYTHON"
echo ""

cd "$HERE"
exec "$VENV_PYTHON" -m eval.runner "$@"
