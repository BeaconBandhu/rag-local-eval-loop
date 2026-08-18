# rag-local-eval-loop

An evaluation loop for a local RAG system (FAISS + Sentence-Transformers
retrieval, OpenAI/local-GPU generation) — retrieval quality, hallucination
rate, answer correctness, and a "lying factor" reliability check, all
sampled straight from the [ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)
dataset. No hand-written eval queries anywhere in this repo.

This repo is deliberately separate from the RAG project it evaluates —
see [Setup](#setup) for how the two connect. It tests the *real* target
system in-process (real embedding model, real retriever, real generation
backend), not a reimplementation of its logic.

## Methodology

The check design follows the reference-based vs. reference-free framing
and the LLM-as-a-judge technique from CampusX's *"LLM Eval Methods |
LLM-as-a-Judge | Reference Based Evals Vs Reference Free Evals"*
([video](https://youtu.be/uQFLY8rQVYA)). In honesty: what's confirmed here
is that video's title and the standard industry meaning of those terms —
YouTube's page didn't serve this tool a full transcript, so the five
checks below are this project's own application of that taxonomy to a RAG
system, not a line-by-line reproduction of the video's code. The mapping:

| Check | Kind | Ground truth source |
|---|---|---|
| Retrieval | **Reference-based** | MSMARCO-XI `passages.is_selected` labels |
| Correctness | **Reference-based**, via LLM-as-judge | MSMARCO-XI `Eng_Answer` |
| Faithfulness / hallucination | **Reference-free**, via LLM-as-judge | none shown to the judge — only the retrieved context |
| Reliability / "lying factor" | **Reference-based** (behavioral) | MSMARCO-XI answerable-vs-unanswerable split |
| Latency | Measurement, not a judgment call | — |

`eval/judge.py` implements the two LLM-as-judge calls. They always run
against a fixed OpenAI judge model, independent of whichever generation
backend the target system is using — grading a model with itself is a
known bias risk, so the judge never reuses the exact call that produced
the answer under test.

## Where the data comes from

Every query, candidate passage, and ground-truth answer in this suite
comes from a real MSMARCO-XI validation row — nothing is hand-written.
Verified directly (not assumed) by downloading and inspecting
`validation/hinval.parquet` with `pyarrow`:

- Each row: a query (English + one of 14 Indic languages), 10 candidate
  passages per language, an `is_selected` label per candidate, and
  `Eng_Answer`/`Answer` ground-truth answer text.
- A random 500-row sample split almost exactly 50/50 between rows that
  **have** a relevant candidate (`is_selected` contains a 1, real answer
  text) and rows that **don't** (`is_selected` all zero, `Eng_Answer` is
  the literal string `"No Answer Present."`). This isn't a data quality
  issue — MSMARCO-XI inherits it directly from the original MS MARCO
  dataset, which deliberately includes a large share of unanswerable
  queries.

This suite uses **both** buckets on purpose:

- **Answerable** rows drive the retrieval check (is the right passage
  found?) and the correctness check (does the answer match the reference?).
- **Unanswerable** rows are a built-in negative control: none of their 10
  candidates actually answer the query, so a well-behaved system should
  decline. If it answers anyway, that's a fabrication — see "lying
  factor" below.

`eval/dataset.py` samples a fixed-seed (`--seed`, default 42) N of each
bucket. `eval/index_build.py` then builds a **throwaway, in-memory FAISS
index** from just the sampled examples' candidate passages (mirrors the
target project's own `benchmark/ragbench.py` pattern) — this suite never
touches the target's live `index/` directory.

The index is **mixed-language on purpose**: every candidate passage goes
in in both English and the Indic language, tagged by language. Retrieval
is graded two ways — "cross-lingual" (either language counts as a hit)
and "same-language only" — because the target project's embedding model
was specifically fine-tuned for Hindi+English cross-lingual retrieval on
this exact dataset, and cross-lingual recall is the metric that actually
reflects what that fine-tune was for.

Generation and the LLM-judge checks are **English-only**: the ground-truth
answers and the judge prompts are English, so English is the language
those checks can grade correctly without guessing at a Hindi judge
prompt's accuracy. Retrieval is graded bilingually; generation isn't —
see [Scope and limitations](#scope-and-limitations).

## What's checked

Five independent checks, each in `eval/checks/`:

1. **`retrieval.py`** — Recall@1/3/5 and MRR against `is_selected` ground
   truth, both cross-lingual and same-language variants.
2. **`faithfulness.py`** — reference-free LLM-judge: is every claim in the
   answer supported by the context that was actually retrieved for it?
   This is the hallucination measurement. Also computes **self-report
   precision**: of the answers the target system's own `grounded` flag
   marked as confident, how many did the judge independently confirm as
   faithful? (Restricted to `grounded=True` cases — a refusal is trivially
   "faithful" to any context, so including refusals would make the two
   signals disagree on every correct refusal for reasons unrelated to
   self-report accuracy.)
3. **`correctness.py`** — reference-based LLM-judge: does the answer
   convey MSMARCO-XI's actual `Eng_Answer`, for answerable queries only?
4. **`reliability.py`** — the **"lying factor"**: a 2×2 of ground-truth
   answerable-or-not against system-answered-or-not.
   - **False refusal rate**: answerable per the dataset, but the system
     declined. Lost information, not wrong information.
   - **False confidence rate**: unanswerable per the dataset (no candidate
     is relevant), but the system answered anyway. A fabrication with no
     basis in the retrieved evidence — the sharper failure of the two.
5. **`latency.py`** — embed/search/generation timing percentiles, against
   the target project's real `LATENCY_BUDGET_MS` (retrieval) and this
   suite's own explicitly-labeled `GENERATION_LATENCY_TARGET_MS`
   (generation isn't covered by any budget in the target project by
   design — see its own `app/generator.py` docstring).

## Architecture

```
eval/dataset.py      -- sample N answerable + N unanswerable rows from MSMARCO-XI
eval/index_build.py  -- build a throwaway mixed-language FAISS index from their candidates
        |
        v
eval/pipeline.py (Phase A)
  for each example: real retrieval (target's app.embedder + this index)
                     + real generation (target's app.generator.generate_answer)
  parallelized across examples via a thread pool
        |
        v
eval/checks/*.py (Phase B -- five checks, dispatched concurrently)
  retrieval.py   faithfulness.py   correctness.py   reliability.py   latency.py
        |
        v
eval/report.py  -- terminal gap-to-perfect report + results/<timestamp>.json
```

**Parallelism, honestly described**: Phase B runs all five checks as
concurrent futures — this is the "multiple parallel checking mechanisms"
this repo was built around. Two of them (`faithfulness`, `correctness`)
further parallelize their own judge calls across examples internally, and
*that* part is a real wall-clock win: each judge call is a blocking
network request, and network waits release Python's GIL, so multiple
in-flight requests genuinely overlap. `retrieval`, `reliability`, and
`latency` are pure aggregation over numbers Phase A already collected —
running them concurrently doesn't buy real speed, they're dispatched
through the same pool for architectural consistency, not because they're
a bottleneck.

Phase A's parallelism depends on the target's `GENERATION_BACKEND`
(`app/config.py` in the target project):
- `"openai"` — real speedup, same reasoning as the judge calls above.
- `"local"` — **no speedup, and `eval/pipeline.py` auto-clamps to 1
  worker.** That backend holds one model on one GPU; concurrent threads
  calling `.generate()` on it would contend for the same CUDA device with
  no throughput gain and a real risk of GPU memory pressure from multiple
  simultaneous KV caches. This isn't configurable away — it's a
  correctness choice, not a preference.

## Setup

This suite needs the target RAG project's exact runtime (it imports and
executes that project's `app.*` and `training.*` modules in-process —
same embedding model, same FAISS version, same generation backend, same
`OPENAI_API_KEY` via that project's own `.env`). **Run it with the target
project's own virtualenv Python** rather than creating a new one:

```bash
# from this repo's root
RAG_PROJECT_ROOT=/path/to/RAG  /path/to/RAG/.venv/bin/python -m eval.runner        # macOS/Linux
$env:RAG_PROJECT_ROOT="D:\path\to\RAG"; D:\path\to\RAG\.venv\Scripts\python.exe -m eval.runner   # PowerShell
```

If this repo and the target project are cloned as sibling directories
(`.../RAG` next to `.../rag-local-eval-loop`), `RAG_PROJECT_ROOT` isn't
even required — `eval/target.py` defaults to that sibling path. Override
with `--rag-root <path>` any time.

### CLI options

```
python -m eval.runner
  --num-answerable N     answerable rows to sample (default: 25)
  --num-unanswerable N   unanswerable rows to sample (default: 25)
  --top-k K              results retrieved per query (default: 5)
  --workers N             parallel workers for retrieval+generation (default: 6;
                           auto-clamped to 1 if the target's GENERATION_BACKEND is "local")
  --judge-workers N       parallel workers per judge check (default: 8)
  --seed N                sampling seed (default: 42)
  --language CODE         MSMARCO-XI language code (default: hin)
  --split NAME             MSMARCO-XI split (default: validation)
  --rag-root PATH          path to the target RAG project
```

## A real run

The output below is real — copied verbatim from an actual run against the
target project (`GENERATION_BACKEND="local"`, `Qwen/Qwen3-0.6B`,
`--num-answerable 15 --num-unanswerable 15 --seed 42`), not
hand-constructed:

```
RAG Local Eval Loop -- results
======================================================================
Target project:     C:\Users\User\Pictures\RAG
Generation backend: local (Qwen/Qwen3-0.6B)
Dataset:            ai4bharat/MSMARCO-XI (hin, validation)
Sample:             15 answerable + 15 unanswerable (seed=42)
Index:              671 chunks (EN+HI) from 30 examples' candidates
top_k:              5

RETRIEVAL  (reference-based -- vs. MSMARCO-XI is_selected labels)
-----------------------------------------------------------------
  15 answerable queries evaluated

  cross-lingual (either language is a hit):
  Recall@1                      0.733  [##################......]  ideal 1.000  -26.7pp short
  Recall@3                      1.000  [########################]  ideal 1.000  PERFECT
  Recall@5                      1.000  [########################]  ideal 1.000  PERFECT
  MRR                           0.844  [####################....]  ideal 1.000  -15.6pp short

FAITHFULNESS / HALLUCINATION  (reference-free -- LLM-as-judge, no ground truth shown to judge)
----------------------------------------------------------------------------------------------
  30 answers evaluated
  Faithful rate                 0.633  [###############.........]  ideal 1.000  -36.7pp short
  Hallucination rate            0.367  [#########...............]  ideal 0.000  +36.7pp over
  Self-report precision         0.476  [###########.............]  ideal 1.000  -52.4pp short

CORRECTNESS  (reference-based -- LLM-as-judge vs. MSMARCO-XI Eng_Answer)
------------------------------------------------------------------------
  15 answerable-query answers evaluated
  Correct rate                  0.667  [################........]  ideal 1.000  -33.3pp short

RELIABILITY / "LYING FACTOR"  (should-answer vs. did-answer)
------------------------------------------------------------
  False refusal rate            0.267  [######..................]  ideal 0.000  +26.7pp over
  False confidence rate         0.667  [################........]  ideal 0.000  +66.7pp over

LATENCY
-------
  stage                 avg      p50      p95      p99   (ms)
  retrieval_total       6.84     6.53     8.69     8.72
  generation          1226.70  1208.13  2312.18  2418.05

  Retrieval  p95 8.69ms vs. 50ms budget  -> PASS
  Generation p95 2312.18ms vs. 1500.0ms target  -> OVER TARGET
```

(Recall@3/@5 both hit the retrieval index's ceiling — perfect — while
Recall@1, faithfulness, correctness, and both reliability rates all show
real, non-trivial gaps. That contrast is the report doing its job: it's
telling you retrieval isn't the bottleneck here, generation reliability
is — a conclusion a single blended "accuracy" number would have hidden.)

## This suite already found a real bug

On its first real run against the target project, this suite's own
faithfulness/reliability checks flagged an example where the local model
declined to answer ("The provided documents *do not* contain information
about...") but the target project's own `grounded` flag reported `True`
(confident answer). Tracing it down: the target's refusal detector
(`app/generator.py`'s `_is_grounded()`) only matched the system prompt's
exact literal phrase — `"don't contain information"` — and the smaller
local model doesn't always reproduce that verbatim; it paraphrases
("do not contain information", "doesn't contain information", ...). The
single-string check missed the paraphrase and mislabeled a genuine refusal
as a confident answer — which would have shown up in the target
project's own dashboard, not just this suite's report.

Fixed in the target project by matching a negation + "contain information"
pattern instead of one fixed string (see that project's `app/generator.py`
git history). This is included here as the intended proof of concept: an
eval loop that only ever produces reassuring numbers isn't testing
anything.

## Interpreting results

Every rate metric is reported as **gap-to-perfect**, not gap-to-an-
arbitrary-bar picked by this suite: recall/faithful/correct rates compare
against an ideal of `1.000`; hallucination/false-refusal/false-confidence
rates compare against an ideal of `0.000`. `PERFECT` means the gap rounded
to zero at this sample size — not a guarantee it stays that way at scale.
The one place a real threshold exists is latency: retrieval is checked
against the target project's actual declared budget
(`LATENCY_BUDGET_MS`), and generation against this suite's own explicitly
labeled target (`GENERATION_LATENCY_TARGET_MS`, override via
`EVAL_GENERATION_LATENCY_TARGET_MS`) — those two get PASS/FAIL because
they're the only two with a stated bar to check against.

`results/<timestamp>.json` (gitignored) holds the full report, including
more flagged examples than the terminal shows (5 per check, vs. 3 printed).

## Scope and limitations

- **Generation/judge checks are English-only.** Retrieval is graded
  bilingually (see above), but the system prompt, both judge prompts, and
  MSMARCO-XI's ground-truth answers are all English — extending
  correctness/faithfulness grading to Hindi would need a separately
  designed and verified Hindi judge prompt, not a naive translation of the
  English one. Not implemented here.
- **Judge output isn't perfectly reproducible.** The judge model doesn't
  accept a non-default `temperature` (verified for this project's whole
  generation stack, not assumed), so repeated runs with the same seed can
  see faithfulness/correctness rates shift a few points between runs. The
  dataset sample itself is fully reproducible (fixed-seed); the judge's
  reading of it is not.
- **The local-backend worker clamp is a hard floor, not a suggestion** —
  see [Architecture](#architecture). Don't route around it by editing
  `eval/pipeline.py`'s clamp; it exists because of an actual GPU
  contention/correctness concern, not a conservative default.
- **This suite grades whatever `GENERATION_BACKEND` the target project is
  currently configured with** — it doesn't run both backends and compare
  them in one invocation. Re-run after flipping the target's config to
  compare.

## License

MIT — see [LICENSE](LICENSE).
