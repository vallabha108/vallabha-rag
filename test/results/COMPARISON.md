# Backend comparison: Vertex `gemini-3.5-flash` vs local Ollama `gemma4`

Same corpus (`vallabha-vle-docs`), same 9 prompts (`test/prompt.json`), same `top_k=5`.
Each backend did its own **generation AND judging** (Opik + DeepEval), so differences
combine generator quality and judge quality. Raw artifacts: `vertex/` and `ollama/`.

## Summary averages

| Metric | Vertex | Ollama | Note |
|---|---|---|---|
| Opik answer relevance | 1.00 | 1.00 | identical |
| Opik hallucination ↓ | 0.00 | 0.00 | identical — no fabrication either way |
| Opik context precision | 0.88 | 0.79 | small drop |
| Opik context recall | 0.87 | 0.88 | equivalent |
| DeepEval answer relevancy | 0.96 | 1.00 | equivalent |
| DeepEval faithfulness | 1.00 | 1.00 | identical |
| DeepEval contextual precision | 0.70 | 0.76 | equivalent |
| DeepEval contextual recall | 0.78 | 0.67 | small drop |
| DeepEval hallucination ↓ | 0.33 | 0.60 | noisier local judge + known abstention artifact |
| Metric errors | 0 | 1 | local judge timed out on the longest case |
| Flagged cases (mitigations) | 5 | 8 | local judges trip thresholds more often |

## Key findings

1. **Answer quality is comparable on factual prompts.** All factual/synthesis answers
   from `gemma4` were correct and grounded; both out-of-scope prompts correctly
   abstained on both backends.
2. **`gemma4` attempted the cross-document comparison that Gemini abstained on.**
   The strict grounding prompt made cloud Gemini reply "not in the documents" for the
   India-vs-Bhutan comparison, while the local model synthesized an answer from the
   retrieved chunks. Looser instruction-following can look like a feature here, but it
   is the same trait that risks hallucination on harder cases.
3. **Local judges are noisier.** DeepEval hallucination averaged 0.60 vs 0.33 (on top
   of the abstention artifact), 8 of 9 cases got flagged for mitigation vs 5, and one
   metric (`deepeval.hallucination` on the longest case) hit a local timeout
   (`RetryError/TimeoutError`) — a small model generating long judge rationales can
   exceed DeepEval's per-metric timeout.
4. **Wall-clock:** the Ollama run took ~50 minutes end-to-end on an Apple-silicon Mac
   (vs ~15 minutes for Vertex), dominated by ~80 sequential local judge calls.

## Recommendation

Use `--backend ollama` for **cheap, private iteration** on prompts and retrieval
settings; use `--backend vertex` for **scores you compare over time or act on**.
Do not mix backends when tracking a metric trend — judge severity differs.

Reproduce: `uv run rag test-all --backend ollama` / `uv run rag test-all`.
