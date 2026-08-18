"""LLM-as-a-judge: the technique from the CampusX "LLM Eval Methods" video
this suite follows -- prompting an LLM to score another model's output
against a stated rubric, rather than exact/fuzzy string matching.

Two judge calls, matching that video's reference-based vs. reference-free
split exactly:

  judge_faithfulness()  -- REFERENCE-FREE. No ground-truth answer is given
                            to the judge at all -- only the retrieved
                            context and the generated answer. Scores
                            whether every claim in the answer is actually
                            supported by that context. This is the
                            hallucination check: a reference-free judge is
                            required here specifically because hallucination
                            is a property of the answer's relationship to
                            its *own* context, not to some external ground
                            truth -- an answer can be faithful to bad
                            context, or unfaithful even when the context
                            happens to be the same topic as a correct
                            reference answer.

  judge_correctness()   -- REFERENCE-BASED. Given the MSMARCO-XI ground-
                            truth answer (Eng_Answer) as the reference, and
                            the target system's generated answer, scores
                            whether they convey the same information. This
                            is what "correctness" means here -- e.g. is the
                            model right, not just non-hallucinatory (a
                            model can be faithful to its context and still
                            wrong, if the retrieved context itself doesn't
                            contain the correct answer).

Deliberately a *separate* OpenAI call from whatever GENERATION_BACKEND
produced the answer under test (see eval/target.py) -- judging a model
with itself, using the same call that produced the answer, is a known bias
risk (a model is more likely to rate its own output favorably). The judge
always calls the OpenAI API directly (JUDGE_MODEL below), independent of
whether the system under test is running "openai" or "local" generation.
"""
import json
import os
import time
from dataclasses import dataclass

JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "gpt-5.4-mini")

_client = None


class JudgeNotConfigured(RuntimeError):
    """No OPENAI_API_KEY available for the judge model."""


@dataclass
class JudgeVerdict:
    verdict: bool          # True = faithful / correct, False = hallucinated / incorrect
    reason: str
    judge_ms: float
    raw: str                # raw judge output, kept for debugging/audit


def _get_client():
    global _client
    if _client is None:
        import openai

        if not os.environ.get("OPENAI_API_KEY"):
            raise JudgeNotConfigured(
                "The judge needs OPENAI_API_KEY (loaded via the target project's .env -- "
                "see eval/target.py). Judge-based checks (faithfulness, correctness) can't "
                "run without it; retrieval, reliability, and latency checks don't need it."
            )
        _client = openai.OpenAI()
    return _client


def _call_judge(system_prompt: str, user_content: str) -> JudgeVerdict:
    client = _get_client()
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        max_completion_tokens=200,
        response_format={"type": "json_object"},  # verified working on JUDGE_MODEL before use, not assumed
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    judge_ms = (time.perf_counter() - t0) * 1000
    raw = (response.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(raw)
        verdict = bool(parsed["verdict"])
        reason = str(parsed.get("reason", ""))
    except (json.JSONDecodeError, KeyError, TypeError):
        # Judge didn't follow the JSON contract -- fail closed (treat as a
        # negative verdict) rather than silently dropping the example, and
        # keep the raw text so it's auditable in the saved report.
        verdict = False
        reason = f"[judge output did not parse as expected JSON: {raw[:200]!r}]"
    return JudgeVerdict(verdict=verdict, reason=reason, judge_ms=judge_ms, raw=raw)


_FAITHFULNESS_SYSTEM = """You are a strict fact-checking judge for a retrieval-augmented \
generation system. You will be given CONTEXT (retrieved document chunks) and an ANSWER a \
model produced from that context. Judge ONLY whether every factual claim in the ANSWER is \
directly supported by the CONTEXT -- do not judge whether the answer is true in general, \
only whether the CONTEXT supports it. An answer that correctly says the context doesn't \
cover the question is faithful (verdict: true). An answer that states anything not \
present in or directly implied by the CONTEXT is unfaithful (verdict: false), even if that \
claim happens to be true in reality.

Respond ONLY with a JSON object: {"verdict": true or false, "reason": "one short sentence"}"""


def judge_faithfulness(answer: str, context: str) -> JudgeVerdict:
    user_content = f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
    return _call_judge(_FAITHFULNESS_SYSTEM, user_content)


_CORRECTNESS_SYSTEM = """You are a grading judge comparing a model's ANSWER to a QUESTION \
against a REFERENCE ANSWER known to be correct. Judge whether the ANSWER conveys the same \
core information as the REFERENCE ANSWER -- wording, length, and extra (correct) detail \
don't matter, only whether the key fact(s) match. If the ANSWER says the documents don't \
contain the information, or refuses to answer, that is INCORRECT (verdict: false) -- the \
REFERENCE ANSWER proves the information was answerable.

Respond ONLY with a JSON object: {"verdict": true or false, "reason": "one short sentence"}"""


def judge_correctness(query: str, answer: str, reference_answer: str) -> JudgeVerdict:
    user_content = f"QUESTION:\n{query}\n\nREFERENCE ANSWER:\n{reference_answer}\n\nANSWER:\n{answer}"
    return _call_judge(_CORRECTNESS_SYSTEM, user_content)
