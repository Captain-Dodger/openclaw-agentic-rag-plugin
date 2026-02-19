# Agentic RAG Plugin (Scaffold)

This module provides a model-agnostic retrieval tool that can be attached to an OpenClaw-style agent pipeline.

## What it does

- Retrieves relevant local documents for a query.
- Computes confidence from retrieval quality.
- Returns either:
  - `mode="answer"` with grounded snippets, or
  - `mode="abstain"` if evidence is too weak.

## Why this matters

It gives local agents a safer operating mode:

- fewer ungrounded answers on unanswerable prompts,
- explicit abstention path instead of forcing a guess,
- measurable with A/B metrics.

## Entry point

- `handle_tool_call(payload, state=None)` in `plugin.py`
- Manifest: `manifest.json`

## Quick test

Run the included A/B harness:

```bash
python3 tools/run_openclaw_agentic_rag_ab.py \
  --suite assays/openclaw_agentic_rag_ab_suite_v1.json \
  --corpus docs/skills/openclaw_agentic_rag/corpus_demo.json
```

Outputs are written to:

- `docs/skills/openclaw_agentic_rag_runs/<run_id>/summary.json`
- `docs/skills/openclaw_agentic_rag_runs/<run_id>/report.md`
- `docs/skills/openclaw_agentic_rag_runs/<run_id>/per_case.csv`

