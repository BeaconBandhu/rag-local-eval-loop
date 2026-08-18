"""Resolves, imports, and verifies the RAG project under test.

This suite evaluates ANY RAG project -- what's fixed across every run is
the dataset (see eval/msmarco.py); what varies is the target's own
retrieval and generation code. Verification is done by actually
IMPORTING the target's embedder/generator modules and checking the
required functions exist on them -- not by checking for expected file
names on disk. A filename check is a proxy that breaks the moment
someone's project is laid out differently (a flat main.py instead of an
app/ package, for instance) even though the actual required functions
might be right there under a different name -- and, worse, it can drift
out of sync with what's actually required (this suite shipped exactly
that bug once already: the launcher scripts checked for app/config.py
after eval/target.py itself had already stopped requiring it). Importing
the real module and checking real attributes can't drift like that --
it's always checking the actual thing that matters.

REQUIRED interface (see also TARGET_INTERFACE.md at this repo's root for
the full spec with examples):

  <embedder module>  (default "app.embedder", override EVAL_EMBEDDER_MODULE)
    embed(texts: list[str]) -> array-like, shape (len(texts), dim)
    embed_one(text: str) -> array-like, shape (dim,)
    get_model() -> anything (called once; only its side effect of loading
                   the model matters -- the return value is unused)

  <generator module>  (default "app.generator", override EVAL_GENERATOR_MODULE)
    generate_answer(query: str, results: list[<context object>]) -> <answer object>
      Each context object needs `.text` and `.source` attributes (plain
      duck typing -- eval/pipeline.py builds its own simple object with
      those two fields, not the target's own context class).
      The returned answer object needs `.text: str`, `.grounded: bool`,
      `.generation_ms: float`, `.model: str`.

If your project doesn't use an app/ package at all -- say, a flat
main.py at the project root defining embed/embed_one/generate_answer
directly -- point at it with:
    EVAL_EMBEDDER_MODULE=main  EVAL_GENERATOR_MODULE=main
(same module twice is fine if both live in one file). Whatever module
name you give must be importable once the target root is on sys.path,
which load_target() below arranges.

OPTIONAL, with suite-owned fallbacks if absent -- see each module's own
docstring for the exact fallback value and how to override it:

  app.config.GENERATION_BACKEND     -- worker-count safety clamp (see
                                        eval/pipeline.py)
  app.config.LATENCY_BUDGET_MS      -- retrieval latency budget for the
                                        report (see eval/checks/latency.py)
  app.config.GENERATION_MODEL /
  app.config.LOCAL_GENERATION_MODEL -- cosmetic "model" label in the
                                        report only

These optional items DO still come from a fixed location, app.config --
unlike the two required modules above, there's no override for this one,
since it's read defensively with getattr() and a default rather than
required to exist at all (see optional_config() below). If your project
doesn't have an app/config.py, or has one under a different name, these
three items simply fall back to their defaults; nothing breaks.

How the target ROOT DIRECTORY (not the module names within it) is
located, in order:
  1. --rag-root CLI flag (highest priority)
  2. RAG_PROJECT_ROOT environment variable
  3. A sibling directory named "RAG" next to this repo's own folder
     (i.e. ../RAG relative to this file) -- true on the machine this suite
     was built on, and a reasonable default for anyone who clones both
     repos side by side, but not assumed to be true anywhere else.

Run this suite using *the target project's own virtualenv*, not a fresh
one -- it imports and executes that project's real code in-process, which
means it needs that project's exact dependencies. See README.md's Setup
section.
"""
import importlib
import os
import sys
from pathlib import Path
from typing import Any

_THIS_REPO_ROOT = Path(__file__).resolve().parent.parent
_injected = False

EMBEDDER_MODULE = os.environ.get("EVAL_EMBEDDER_MODULE", "app.embedder")
GENERATOR_MODULE = os.environ.get("EVAL_GENERATOR_MODULE", "app.generator")

_REQUIRED_EMBEDDER_ATTRS = ("embed", "embed_one", "get_model")
_REQUIRED_GENERATOR_ATTRS = ("generate_answer",)


class TargetNotFound(RuntimeError):
    pass


def resolve_target_root(cli_arg: str | None = None) -> Path:
    candidate = cli_arg or os.environ.get("RAG_PROJECT_ROOT") or str(_THIS_REPO_ROOT.parent / "RAG")
    path = Path(candidate).resolve()
    if not path.is_dir():
        raise TargetNotFound(
            f"'{path}' is not a directory.\n"
            f"Point at your project's root with --rag-root <path>, or set the RAG_PROJECT_ROOT "
            f"environment variable, e.g.:\n"
            f'  $env:RAG_PROJECT_ROOT = "C:\\path\\to\\your-project"   # PowerShell\n'
            f"  export RAG_PROJECT_ROOT=/path/to/your-project          # bash"
        )
    return path


def load_target(cli_arg: str | None = None) -> Path:
    """Inserts the target project's root at the front of sys.path (once)
    so its modules become importable. Returns the resolved path. Does NOT
    verify the interface -- see verify_target() for that; kept separate
    because some callers (e.g. eval/dataset.py used to, before it stopped
    needing the target at all) only need the path resolved, not verified."""
    global _injected
    root = resolve_target_root(cli_arg)
    if not _injected:
        sys.path.insert(0, str(root))
        _injected = True
    return root


def verify_target(cli_arg: str | None = None) -> Path:
    """Like load_target(), but also actually imports EMBEDDER_MODULE and
    GENERATOR_MODULE and checks every required attribute is present --
    real verification of the real interface, not a proxy check against
    expected file names. Raises TargetNotFound with a specific, actionable
    message (which module failed to import, or which attribute is
    missing) rather than letting a confusing ImportError/AttributeError
    surface three calls later inside eval/pipeline.py."""
    root = load_target(cli_arg)

    def _check(module_name: str, required_attrs: tuple[str, ...], env_var: str) -> None:
        try:
            mod = importlib.import_module(module_name)
        except ImportError as e:
            raise TargetNotFound(
                f"Could not import '{module_name}' from '{root}': {e}\n"
                f"If your project doesn't use this module path, set {env_var} to the right one "
                f"(e.g. {env_var}=main for a flat main.py). See TARGET_INTERFACE.md."
            ) from e
        missing = [a for a in required_attrs if not hasattr(mod, a)]
        if missing:
            raise TargetNotFound(
                f"'{module_name}' (from '{root}') is missing required attribute(s): "
                f"{', '.join(missing)}. See TARGET_INTERFACE.md for the full interface."
            )

    _check(EMBEDDER_MODULE, _REQUIRED_EMBEDDER_ATTRS, "EVAL_EMBEDDER_MODULE")
    _check(GENERATOR_MODULE, _REQUIRED_GENERATOR_ATTRS, "EVAL_GENERATOR_MODULE")
    return root


def get_embedder():
    load_target()
    return importlib.import_module(EMBEDDER_MODULE)


def get_generator():
    load_target()
    return importlib.import_module(GENERATOR_MODULE)


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
