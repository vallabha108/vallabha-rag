"""RAG evaluation pipeline for the Vallabha VLE corpus.

Stage 1 (run_prompts): each prompt in test/prompt.json is answered by
retrieving top-k chunks from the RAG corpus and generating a grounded answer
with Gemini. Raw results land in test/results/rag_results.json.

Stage 2 (evaluate): every answer is scored by two independent LLM-as-judge
libraries — Opik (Comet) and DeepEval (Confident AI) — using Gemini on Vertex
AI as the judge model via ADC. Scores land in test/results/metrics.json.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent / "test"
PROMPTS_PATH = TEST_DIR / "prompt.json"
RESULTS_DIR = TEST_DIR / "results"
RAG_RESULTS_PATH = RESULTS_DIR / "rag_results.json"
METRICS_PATH = RESULTS_DIR / "metrics.json"

EVAL_MODEL = os.environ.get("VALLABHA_RAG_EVAL_MODEL", "gemini-3.5-flash")
# Gemini publisher models are served from the global endpoint, not europe-west4.
GENAI_LOCATION = os.environ.get("VALLABHA_RAG_GENAI_LOCATION", "global")

# Backend for generation + judging: "vertex" (cloud Gemini) or "ollama" (local).
EVAL_BACKEND = os.environ.get("VALLABHA_RAG_EVAL_BACKEND", "vertex")
OLLAMA_MODEL = os.environ.get("VALLABHA_RAG_OLLAMA_MODEL", "gemma4")
OLLAMA_URL = os.environ.get("VALLABHA_RAG_OLLAMA_URL", "http://localhost:11434")


def resolve_model(backend: str, model: str | None) -> str:
    if model:
        return model
    return EVAL_MODEL if backend == "vertex" else OLLAMA_MODEL


def _make_generator(backend: str, model: str, project: str):
    """Return generate(prompt) -> str for the chosen backend."""
    if backend == "vertex":
        from google import genai

        client = genai.Client(vertexai=True, project=project, location=GENAI_LOCATION)

        def generate(prompt: str) -> str:
            r = client.models.generate_content(model=model, contents=prompt)
            return (r.text or "").strip()

        return generate

    import litellm

    def generate(prompt: str) -> str:
        r = litellm.completion(
            model=f"ollama_chat/{model}",
            api_base=OLLAMA_URL,
            messages=[{"role": "user", "content": prompt}],
        )
        return (r.choices[0].message.content or "").strip()

    return generate

LOWER_IS_BETTER = {"hallucination"}

# DeepEval's HallucinationMetric penalizes abstention; on out-of-scope prompts
# abstaining is the desired behavior, so the recommender must be told about it.
CATEGORY_NOTES = {
    "factual": "single-chunk lookup; the answer should be directly retrievable",
    "synthesis": "requires combining several chunks from one document",
    "cross-document": "requires combining chunks from BOTH documents in the corpus",
    "out-of-scope": "the answer is NOT in the corpus - abstaining is the DESIRED behavior",
}

MITIGATION_PROMPT = """\
You are a senior RAG engineer reviewing one failing evaluation case for a
Vertex AI RAG Engine pipeline.

Pipeline configuration:
- Retrieval: RAG Engine default chunking/indexing/embedding, top_k={top_k}
- Generation: {model} with a strict grounding prompt ("answer ONLY from context, \
abstain with a fixed sentence if the context lacks the answer", temperature default)
- Judges: Opik and DeepEval LLM-as-judge metrics (0-1)

Test case (category: {category} - {category_note}):
Question: {question}
Expected answer: {expected}
Actual answer: {answer}
Retrieved chunk sources: {sources}

Low-scoring metrics with the judge's reasoning:
{flagged}

Your task:
1. Diagnose the single most likely root cause: retrieval problem, generation/prompt
   problem, test-design problem, or metric artifact (a judge quirk, e.g. DeepEval's
   hallucination metric penalizing a correct abstention).
2. Give 2-4 concrete, prioritized mitigations with specific parameter values where
   possible (top_k changes, chunk_size/chunk_overlap via TransformationConfig on
   re-upload, grounding-prompt wording changes, prompt.json fixes).
   If the root cause is a metric artifact, say "No action needed" and explain briefly.

Start every mitigation bullet with its scope tag:
- [PROMPT] = fixable for this test prompt alone, without touching the pipeline:
  a per-prompt "top_k" override in test/prompt.json (supported), rewording the
  question, or fixing the expected_output.
- [SYSTEM] = changes the whole pipeline for every prompt: default top_k,
  chunk_size/chunk_overlap (requires re-upload), the shared grounding prompt,
  generation model or temperature.
Prefer [PROMPT] when the issue is isolated to this case; use [SYSTEM] only when
the same fault would affect many prompts.

Format your reply as plain text: first line "Root cause: ..." then a markdown
bullet list of mitigations. Maximum 140 words total.
"""

OVERALL_PROMPT = """\
You are a senior RAG engineer. Below are per-case diagnoses from an evaluation run
of a Vertex AI RAG Engine pipeline (default chunking, top_k={top_k}, strict
grounding prompt, model {model}).

{diagnoses}

Write 3-5 prioritized recommendations for the engineering team to tune this RAG
pipeline. Be concrete (parameter values, prompt wording). Skip anything that is
only a metric artifact. Start every bullet with a scope tag: [SYSTEM] for
pipeline-wide changes (default top_k, chunking via re-upload, shared grounding
prompt, model/temperature) or [PROMPT] for fixes local to individual test prompts
(per-prompt "top_k" override in test/prompt.json, question or expected_output
wording). Plain text, markdown bullets, max 160 words.
"""

ANSWER_PROMPT = """\
You are a question-answering assistant for the Vallabha VLE document corpus.
Answer the question using ONLY the context passages below.
If the context does not contain the information needed to answer, reply exactly:
"The provided documents do not contain this information."
Be concise and factual. Do not use outside knowledge.

Context passages:
{context}

Question: {question}

Answer:"""


def load_prompts(prompts_path: Path = PROMPTS_PATH) -> list[dict]:
    with open(prompts_path) as f:
        return json.load(f)["prompts"]


def run_prompts(
    project: str,
    location: str,
    corpus_display_name: str,
    top_k: int = 5,
    backend: str = EVAL_BACKEND,
    model: str | None = None,
) -> Path:
    """Retrieve + generate an answer for every prompt; save raw results."""
    import vertexai
    from vertexai import rag

    model = resolve_model(backend, model)
    generate = _make_generator(backend, model, project)
    vertexai.init(project=project, location=location)
    corpus_name = None
    for corpus in rag.list_corpora():
        if corpus.display_name == corpus_display_name:
            corpus_name = str(corpus.name)
            break
    if corpus_name is None:
        sys.exit(f"Corpus '{corpus_display_name}' not found. Run: uv run rag upload docs")

    prompts = load_prompts()
    results = []
    for p in prompts:
        # A prompt may carry its own "top_k" in prompt.json, overriding the run default.
        effective_top_k = int(p.get("top_k", top_k))
        print(f"  [{p['id']}] retrieving (top_k={effective_top_k}) ...", flush=True)
        response = rag.retrieval_query(
            rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
            text=p["question"],
            rag_retrieval_config=rag.RagRetrievalConfig(top_k=effective_top_k),
        )
        contexts = [
            {
                "text": ctx.text,
                "source": ctx.source_display_name or ctx.source_uri,
                "score": float(ctx.score) if ctx.score is not None else None,
            }
            for ctx in response.contexts.contexts
        ]
        context_block = "\n\n".join(
            f"[{i}] (source: {c['source']})\n{c['text']}" for i, c in enumerate(contexts, 1)
        )
        print(f"  [{p['id']}] generating answer with {model} ({backend}) ...", flush=True)
        answer = generate(ANSWER_PROMPT.format(context=context_block, question=p["question"]))
        results.append(
            {
                "id": p["id"],
                "top_k": effective_top_k,
                "category": p["category"],
                "reference_docs": p["reference_docs"],
                "question": p["question"],
                "expected_output": p["expected_output"],
                "answer": answer,
                "contexts": contexts,
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "run": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "project": project,
            "location": location,
            "corpus": corpus_display_name,
            "generation_model": model,
            "backend": backend,
            "top_k": top_k,
        },
        "results": results,
    }
    with open(RAG_RESULTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {len(results)} results to {RAG_RESULTS_PATH}")
    return RAG_RESULTS_PATH


def _score_entry(score, reason, extra: dict | None = None) -> dict:
    entry = {"score": round(float(score), 4) if score is not None else None, "reason": reason}
    if extra:
        entry.update(extra)
    return entry


def _opik_scores(case: dict, model: str) -> dict:
    from opik.evaluation.metrics import AnswerRelevance, ContextPrecision, ContextRecall, Hallucination

    context = [c["text"] for c in case["contexts"]]
    out: dict = {}
    metric_calls = {
        "answer_relevance": lambda: AnswerRelevance(model=model, track=False).score(
            input=case["question"], output=case["answer"], context=context
        ),
        "hallucination": lambda: Hallucination(model=model, track=False).score(
            input=case["question"], output=case["answer"], context=context
        ),
        "context_precision": lambda: ContextPrecision(model=model, track=False).score(
            input=case["question"], output=case["answer"],
            expected_output=case["expected_output"], context=context,
        ),
        "context_recall": lambda: ContextRecall(model=model, track=False).score(
            input=case["question"], output=case["answer"],
            expected_output=case["expected_output"], context=context,
        ),
    }
    for name, call in metric_calls.items():
        try:
            res = call()
            out[name] = _score_entry(res.value, res.reason)
        except Exception as exc:  # noqa: BLE001 - one failed metric must not sink the run
            out[name] = {"score": None, "reason": None, "error": str(exc)[:500]}
    return out


def _deepeval_scores(case: dict, judge) -> dict:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
        HallucinationMetric,
    )
    from deepeval.test_case import LLMTestCase

    retrieval_context = [c["text"] for c in case["contexts"]]
    tc = LLMTestCase(
        input=case["question"],
        actual_output=case["answer"],
        expected_output=case["expected_output"],
        retrieval_context=retrieval_context,
        context=retrieval_context,
    )
    metrics = {
        "answer_relevancy": AnswerRelevancyMetric(model=judge, async_mode=False, include_reason=True),
        "faithfulness": FaithfulnessMetric(model=judge, async_mode=False, include_reason=True),
        "contextual_precision": ContextualPrecisionMetric(model=judge, async_mode=False, include_reason=True),
        "contextual_recall": ContextualRecallMetric(model=judge, async_mode=False, include_reason=True),
        "hallucination": HallucinationMetric(model=judge, async_mode=False, include_reason=True),
    }
    out: dict = {}
    for name, metric in metrics.items():
        try:
            metric.measure(tc)
            out[name] = _score_entry(metric.score, metric.reason, {"success": bool(metric.is_successful())})
        except Exception as exc:  # noqa: BLE001
            out[name] = {"score": None, "reason": None, "error": str(exc)[:500]}
    return out


def _make_judges(backend: str, model: str, project: str):
    """Return (deepeval_judge, opik_model) for the chosen backend."""
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    os.environ.setdefault("OPIK_USAGE_REPORT_ENABLED", "false")

    if backend == "vertex":
        # Route Opik's LiteLLM judge and DeepEval's Gemini judge through Vertex AI + ADC.
        os.environ.setdefault("VERTEXAI_PROJECT", project)
        os.environ.setdefault("VERTEXAI_LOCATION", GENAI_LOCATION)
        from deepeval.models import GeminiModel

        judge = GeminiModel(model=model, project=project, location=GENAI_LOCATION, use_vertexai=True)
        return judge, f"vertex_ai/{model}"

    from deepeval.models import OllamaModel
    from opik.evaluation.models import LiteLLMChatModel

    judge = OllamaModel(model=model, base_url=OLLAMA_URL)
    opik_model = LiteLLMChatModel(model_name=f"ollama_chat/{model}", api_base=OLLAMA_URL)
    return judge, opik_model


def evaluate(project: str, backend: str = EVAL_BACKEND, model: str | None = None) -> Path:
    """Score every RAG result with Opik and DeepEval judges."""
    if not RAG_RESULTS_PATH.exists():
        sys.exit(f"{RAG_RESULTS_PATH} not found. Run: uv run rag test-run")
    with open(RAG_RESULTS_PATH) as f:
        payload = json.load(f)

    model = resolve_model(backend, model)
    judge, opik_model = _make_judges(backend, model, project)

    cases = []
    for case in payload["results"]:
        print(f"  [{case['id']}] scoring with Opik ...", flush=True)
        opik_out = _opik_scores(case, opik_model)
        print(f"  [{case['id']}] scoring with DeepEval ...", flush=True)
        deepeval_out = _deepeval_scores(case, judge)
        cases.append({**case, "opik": opik_out, "deepeval": deepeval_out})

    def averages(lib: str) -> dict:
        sums: dict[str, list[float]] = {}
        for c in cases:
            for metric, entry in c[lib].items():
                if entry.get("score") is not None:
                    sums.setdefault(metric, []).append(entry["score"])
        return {m: round(sum(v) / len(v), 4) for m, v in sums.items() if v}

    metrics_payload = {
        "run": {
            **payload["run"],
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "judge_model": model,
            "judge_backend": "Vertex AI (ADC)" if backend == "vertex" else f"Ollama local ({OLLAMA_URL})",
        },
        "summary": {"opik": averages("opik"), "deepeval": averages("deepeval")},
        "cases": cases,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"\nSaved metrics for {len(cases)} cases to {METRICS_PATH}")
    return METRICS_PATH


def _is_low(metric: str, score: float | None) -> bool:
    if score is None:
        return False
    return score > 0.3 if metric in LOWER_IS_BETTER else score < 0.7


def _flagged(case: dict) -> list[dict]:
    flags = []
    for lib in ("opik", "deepeval"):
        for metric, entry in case[lib].items():
            if _is_low(metric, entry.get("score")):
                flags.append(
                    {"library": lib, "metric": metric, "score": entry["score"], "reason": entry.get("reason") or ""}
                )
    return flags


def recommend(project: str, backend: str = EVAL_BACKEND, model: str | None = None) -> Path:
    """Add per-case mitigations + run-level recommendations to metrics.json."""
    if not METRICS_PATH.exists():
        sys.exit(f"{METRICS_PATH} not found. Run: uv run rag test-eval")
    with open(METRICS_PATH) as f:
        data = json.load(f)
    run = data["run"]

    model = resolve_model(backend, model)
    generate = _make_generator(backend, model, project)
    diagnoses = []
    flagged_cases = 0
    for case in data["cases"]:
        flags = _flagged(case)
        if not flags:
            case.pop("mitigation", None)
            continue
        flagged_cases += 1
        flagged_text = "\n".join(
            f"- {f['library']}.{f['metric']} = {f['score']:.2f}"
            + (" (lower is better)" if f["metric"] in LOWER_IS_BETTER else "")
            + (f"\n  judge: {f['reason'][:400]}" if f["reason"] else "")
            for f in flags
        )
        print(f"  [{case['id']}] {len(flags)} low metric(s) -> generating mitigation ...", flush=True)
        advice = generate(
            MITIGATION_PROMPT.format(
                top_k=case.get("top_k", run.get("top_k")),
                model=run.get("generation_model"),
                category=case["category"],
                category_note=CATEGORY_NOTES.get(case["category"], ""),
                question=case["question"],
                expected=case["expected_output"],
                answer=case["answer"],
                sources=", ".join(sorted({c["source"] for c in case["contexts"]})) or "none",
                flagged=flagged_text,
            ),
        )
        case["mitigation"] = {
            "flags": [f"{f['library']}.{f['metric']}={f['score']:.2f}" for f in flags],
            "advice": advice,
        }
        diagnoses.append(f"Case {case['id']} ({case['category']}):\n{advice}")

    if diagnoses:
        print(f"  generating run-level recommendations from {flagged_cases} flagged case(s) ...", flush=True)
        data["recommendations"] = generate(
            OVERALL_PROMPT.format(
                top_k=run.get("top_k"), model=run.get("generation_model"), diagnoses="\n\n".join(diagnoses)
            )
        )
    else:
        data["recommendations"] = "All metrics are in the healthy band - no tuning actions required for this run."
        print("  no flagged cases - nothing to mitigate.")

    with open(METRICS_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved mitigations for {flagged_cases} case(s) to {METRICS_PATH}")
    return METRICS_PATH
