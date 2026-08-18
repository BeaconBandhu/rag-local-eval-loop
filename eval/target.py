"""Resolves and imports the RAG project under test.

This suite lives in its own repo, separate from whatever RAG project it
evaluates -- and it's built to evaluate ANY RAG project, not just the one
it was originally developed against. What's fixed across every run is the
dataset (see eval/msmarco.py); what varies is the target's own retrieval
and generation code -- its own embedding model, its own vector index,
its own LLM API key or local SLM.

REQUIRED interface (this is the whole contract -- see also
TARGET_INTERFACE.md at this repo's root for the full spec with examples):

  app/embedder.py
    embed(texts: list[str]) -> array-like, shape (len(texts), dim)
    embed_one(text: str) -> array-like, shape (dim,)
    get_model() -> anything (called once; only its side effect of loading
                   the model matters -- the return value is unused)

  app/generator.py
    generate_answer(query: str, results: list[<context object>]) -> <answer object>
      Each context object needs `.text` and `.source` attributes (plain
      duck typing -- eval/pipeline.py builds its own simple object with
      those two fields; it does NOT import the target's own SearchResult
      class, so the target's context type doesn't need to match any
      particular class, just those two attribute names).
      The returned answer object needs `.text: str`, `.grounded: bool`,
      `.generation_ms: float`, `.model: str`.

OPTIONAL, with suite-owned fallbacks if absent (see each module's own
docstring for the exact fallback value and how to override it):

  app.config.GENERATION_BACKEND     -- worker-count safety clamp (see
                                        eval/pipeline.py); without it, no
                                        auto-clamp happens, so set
                                        --workers 1 yourself if your
                                        generation path holds one shared
                                        local-GPU model.
  app.config.LATENCY_BUDGET_MS      -- retrieval latency budget for the
                                        report (see eval/checks/latency.py)
  app.config.GENERATION_MODEL /
  app.config.LOCAL_GENERATION_MODEL -- cosmetic "model" label in the
                                        report only

Explicitly NOT required, unlike earlier versions of this suite: FAISS or
HNSW specifically, any particular chunking function, any particular
embedding dimension constant, or a training/data.py MSMARCO-XI loader --
this suite builds and chunks its own throwaway eval index using its own
defaults (see eval/index_build.py) and loads the dataset itself (see
eval/msmarco.py), neither borrowed from the target.

How the target is located, in order:
  1. --rag-root CLI flag (highest priority)
  2. RAG_PROJECT_ROOT environment variable
  3. A sibling directory named "RAG" next to this repo's own folder
     (i.e. ../RAG relative to this file) -- true on the machine this suite
     was built on, and a reasonable default for anyone who clones both
     repos side by side, but not assumed to be true anywhere else.

Whichever path resolves, it must contain app/embedder.py and
app/generator.py -- the two REQUIRED files above -- verified before
anything is imported, so a wrong or incompatible path fails with a clear
message instead of a confusing ImportError three modules later.

Run this suite using *the target project's own virtualenv*, not a fresh
one -- it imports and executes that project's real code in-process, which
means it needs that project's exact dependencies (whatever app/embedder.py
and app/generator.py themselves need). See README.md's Setup section.
"""
import os
import sys
from pathlib import Path
from typing import Any

_THIS_REPO_ROOT = Path(__file__).resolve().parent.parent
_injected = False


class TargetNotFound(RuntimeError):
    pass


def resolve_target_root(cli_arg: str | None = None) -> Path:
    candidate = cli_arg or os.environ.get("RAG_PROJECT_ROOT") or str(_THIS_REPO_ROOT.parent / "RAG")
    path = Path(candidate).resolve()
    missing = [f for f in ("app/embedder.py", "app/generator.py") if not (path / f).exists()]
    if missing:
        raise TargetNotFound(
            f"'{path}' doesn't look like a compatible RAG project -- missing {', '.join(missing)}.\n"
            f"See TARGET_INTERFACE.md for the full interface this suite needs.\n"
            f"Point at the right project with --rag-root <path>, or set the RAG_PROJECT_ROOT "
            f"environment variable, e.g.:\n"
            f'  $env:RAG_PROJECT_ROOT = "C:\\path\\to\\your-project"   # PowerShell\n'
            f"  export RAG_PROJECT_ROOT=/path/to/your-project          # bash"
        )
    return path


def load_target(cli_arg: str | None = None) -> Path:
    """Inserts the target project's root at the front of sys.path (once)
    so `import app.xxx` resolves to it. Returns the resolved path."""
    global _injected
    root = resolve_target_root(cli_arg)
    if not _injected:
        sys.path.insert(0, str(root))
        _injected = True
    return root


def optional_config(name: str, default: Any) -> Any:
    """Reads app.config.<name> from the target if it exists, else returns
    `default`. Used for every OPTIONAL interface item listed above --
    lets this suite run against targets that don't define config the same
    way this suite's own original target project does, without crashing."""
    load_target()
    try:
        import app.config as target_config
    except ImportError:
        return default
    return getattr(target_config, name, default)
