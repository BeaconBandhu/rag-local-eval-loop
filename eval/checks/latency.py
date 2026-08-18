"""Aggregates the timings already collected during eval/pipeline.py's real
retrieval + generation calls -- makes no calls of its own. Retrieval is
graded against the target project's own LATENCY_BUDGET_MS (app/config.py,
50ms by default) exactly like the target's own app/benchmark.py does.
Generation has no such budget in the target project by design (see
app/generator.py's module docstring: generation latency is deliberately
NOT covered by LATENCY_BUDGET_MS). GENERATION_LATENCY_TARGET_MS below is
this eval suite's own reference point, not a constraint the target project
declares anywhere -- override it with the EVAL_GENERATION_LATENCY_TARGET_MS
environment variable if 1500ms isn't the right bar for your use case.
"""
import os
import statistics

from eval import target
from eval.pipeline import ExampleResult

GENERATION_LATENCY_TARGET_MS = float(os.environ.get("EVAL_GENERATION_LATENCY_TARGET_MS", 1500))


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    return values[f] if f == c else values[f] + (k - f) * (values[c] - values[f])


def _block(values: list[float]) -> dict:
    if not values:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    return {
        "avg_ms": round(statistics.mean(values), 2),
        "p50_ms": round(_percentile(values, 50), 2),
        "p95_ms": round(_percentile(values, 95), 2),
        "p99_ms": round(_percentile(values, 99), 2),
    }


def run(results: list[ExampleResult]) -> dict:
    target.load_target()
    from app.config import LATENCY_BUDGET_MS

    usable = [r for r in results if r.error is None]

    embed_ms = [r.embed_ms_en for r in usable] + [r.embed_ms_hi for r in usable]
    search_ms = [r.search_ms_en for r in usable] + [r.search_ms_hi for r in usable]
    retrieval_total_ms = [r.embed_ms_en + r.search_ms_en for r in usable]
    generation_ms = [r.generation_ms for r in usable if r.generation_ms > 0]

    retrieval_p95 = _percentile(retrieval_total_ms, 95)
    generation_p95 = _percentile(generation_ms, 95)

    return {
        "check": "latency",
        "num_evaluated": len(usable),
        "embed": _block(embed_ms),
        "search": _block(search_ms),
        "retrieval_total": _block(retrieval_total_ms),
        "generation": _block(generation_ms),
        "retrieval_latency_budget_ms": LATENCY_BUDGET_MS,
        "retrieval_within_budget": retrieval_p95 <= LATENCY_BUDGET_MS,
        "generation_latency_target_ms": GENERATION_LATENCY_TARGET_MS,
        "generation_within_target": generation_p95 <= GENERATION_LATENCY_TARGET_MS,
    }
