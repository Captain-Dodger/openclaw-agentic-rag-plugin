# OpenClaw Agentic RAG Plugin

Model-agnostic retrieval plugin scaffold with confidence-gated abstention.

This repository focuses on one practical goal: **reduce hallucinations on unknown prompts without tanking answer quality on known prompts**.

## Why this exists

Agent tool chains often force a response even when evidence is weak.  
This plugin separates two paths:

- `answer` when retrieval evidence is strong enough
- `abstain` when evidence is weak

That behavior is measurable via A/B.

## Core behavior

For each query:

1. retrieve top-k local hits
2. compute a confidence score from retrieval quality
3. apply thresholds:
   - `min_retrieval_score`
   - `min_confidence`
4. return structured output with `mode=answer|abstain`

Response includes:

- `answer`
- `rationale`
- `hits` with `doc_id`, `source`, `score`
- `metrics` (`top_score`, `mean_top2`, `evidence_chars`, `hits`)

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python tools/run_openclaw_agentic_rag_ab.py
```

### Tune thresholds (recommended demo profile)

```bash
python tools/run_openclaw_agentic_rag_ab.py \
  --suite assays/openclaw_agentic_rag_ab_suite_v1.json \
  --corpus data/corpus_demo.json \
  --min-retrieval-score 0.12 \
  --min-confidence 0.12
```

Outputs:

- `results/openclaw_agentic_rag_runs/<run_id>/summary.json`
- `results/openclaw_agentic_rag_runs/<run_id>/report.md`
- `results/openclaw_agentic_rag_runs/<run_id>/per_case.csv`

## Current benchmark snapshot

From `results/openclaw_agentic_rag_runs/run_20260219_231559/summary.json`:

- baseline `abstain_on_unanswerable_rate`: `0.0`
- plugin `abstain_on_unanswerable_rate`: `1.0`
- baseline `hallucination_rate_on_unanswerable`: `1.0`
- plugin `hallucination_rate_on_unanswerable`: `0.0`
- `grounded_answer_rate_on_answerable`: unchanged (`1.0 -> 1.0`) for this suite

Note: this is a small demo suite, not a production claim.

## OpenClaw integration

Entrypoint:

- `agentic_rag_plugin.plugin:handle_tool_call`

Manifest:

- `src/agentic_rag_plugin/manifest.json`

Minimal router sketch:

```python
from agentic_rag_plugin.plugin import handle_tool_call

def route_tool_call(tool_name: str, payload: dict, state: dict):
    if tool_name == "agentic_rag":
        return handle_tool_call(payload, state)
    raise ValueError(f"unknown tool: {tool_name}")
```

## Repository layout

- `src/agentic_rag_plugin/` plugin package
- `tools/run_openclaw_agentic_rag_ab.py` A/B runner
- `assays/openclaw_agentic_rag_ab_suite_v1.json` sample suite
- `data/corpus_demo.json` sample corpus
- `docs/openclaw_agentic_rag_plugin_spec.md` protocol/spec
- `results/` run artifacts

## Limitations

- Retrieval is lexical/scaffold-level (not embedding search yet).
- Baseline in A/B is intentionally naive.
- No network retrieval in this repo.
- Thresholds are corpus-dependent and must be tuned.

## License

MIT (see `LICENSE`).
