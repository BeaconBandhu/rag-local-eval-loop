"""Resolves and imports the RAG project under test.

This suite deliberately lives in its own repo, separate from the RAG
project it evaluates -- but it evaluates the *real* system, not a copy of
its logic. That means it needs to import that project's actual modules
(app.embedder, app.generator, app.chunking, app.config, training.data) and
run them in-process, using the real ai4bharat/MSMARCO-XI dataset loader
and the real fine-tuned embedding model / generation backend the target
project is currently configured with.

How the target is located, in order:
  1. --rag-root CLI flag (highest priority)
  2. RAG_PROJECT_ROOT environment variable
  3. A sibling directory named "RAG" next to this repo's own folder
     (i.e. ../RAG relative to this file) -- true on the machine this suite
     was built on, and a reasonable default for anyone who clones both
     repos side by side, but not assumed to be true anywhere else.

Whichever path resolves, it must contain an `app` package (app/config.py
specifically) -- verified before anything is imported, so a wrong path
fails with a clear message instead of a confusing ImportError three
modules later.

Run this suite using *the target project's own virtualenv* (e.g.
RAG/.venv/Scripts/python.exe on Windows), not a fresh one -- it imports
and executes that project's real code in-process, which means it needs
that project's exact dependencies (torch/transformers if
GENERATION_BACKEND="local", sentence-transformers, faiss-cpu, openai,
python-dotenv). See README.md's Setup section.
"""
import os
import sys
from pathlib import Path

_THIS_REPO_ROOT = Path(__file__).resolve().parent.parent
_injected = False


class TargetNotFound(RuntimeError):
    pass


def resolve_target_root(cli_arg: str | None = None) -> Path:
    candidate = cli_arg or os.environ.get("RAG_PROJECT_ROOT") or str(_THIS_REPO_ROOT.parent / "RAG")
    path = Path(candidate).resolve()
    if not (path / "app" / "config.py").exists():
        raise TargetNotFound(
            f"'{path}' doesn't look like the RAG project (no app/config.py found there).\n"
            f"Point at it with --rag-root <path>, or set the RAG_PROJECT_ROOT environment "
            f"variable, e.g.:\n"
            f'  $env:RAG_PROJECT_ROOT = "C:\\path\\to\\RAG"   # PowerShell\n'
            f"  export RAG_PROJECT_ROOT=/path/to/RAG          # bash"
        )
    return path


def load_target(cli_arg: str | None = None) -> Path:
    """Inserts the target project's root at the front of sys.path (once)
    so `import app.xxx` / `import training.xxx` resolve to it. Returns the
    resolved path for logging."""
    global _injected
    root = resolve_target_root(cli_arg)
    if not _injected:
        sys.path.insert(0, str(root))
        _injected = True
    return root
