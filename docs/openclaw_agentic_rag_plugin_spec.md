# OpenClaw Agentic RAG Plugin Spec (v0.1)

## Intent

Provide a model-independent retrieval layer with confidence gating, so local agents can abstain when evidence is weak instead of hallucinating.

## Scope

- Input: natural-language `query`
- Output: structured decision with `mode = answer|abstain`
- Retrieval source: local indexed corpus (`lexical` or `hybrid`)
- Integration target: OpenClaw tool-call path

## Contract

Request:

```json
{
  "tool": "agentic_rag",
  "query": "What does meaning_trace_v1 capture?"
}
```

Response:

```json
{
  "tool": "agentic_rag",
  "query": "What does meaning_trace_v1 capture?",
  "mode": "answer",
  "confidence": 0.72,
  "answer": "Grounded answer from retrieved evidence: ...",
  "rationale": "Grounded by retrieval hits above thresholds ...",
  "context": "[doc_1] ...",
  "hits": [
    {"doc_id": "doc_1", "source": "contracts", "score": 0.8, "text": "..."}
  ],
  "metrics": {
    "top_score": 0.8,
    "mean_top2": 0.73,
    "evidence_chars": 640,
    "hits": 3,
    "retrieval_mode_effective": "hybrid",
    "embedding_ready": true,
    "hybrid_min_lexical_score": 0.1
  }
}
```

## Guardrails

- Never force an answer below confidence threshold.
- Keep abstain explicit and non-empty.
- Keep evidence snippets traceable (`doc_id`, `source`).
- Keep retrieval and generation separable.
- Keep lexical fallback active if semantic stream is unavailable.
- In hybrid mode, optionally require lexical anchor (`hybrid_min_lexical_score > 0`) to block semantic-only false positives.

## A/B Metrics

- `grounded_answer_rate_on_answerable`
- `abstain_on_unanswerable_rate`
- `abstain_on_answerable_rate`
- `hallucination_rate_on_unanswerable`

## Promotion Criteria (initial)

- Improve `abstain_on_unanswerable_rate` by at least `+0.25` vs baseline.
- Keep `abstain_on_answerable_rate <= 0.15`.
- No regression in grounded answer quality on answerable prompts > `0.10`.
