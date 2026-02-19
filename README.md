# OpenClaw Agentic RAG Plugin (MIT)

Model-agnostic retrieval plugin scaffold with confidence-gated abstention.

This repo is a standalone extraction of the plugin/harness work so you can upload it directly to GitHub.

## Highlights

- Retrieval-first answer path (`answer` mode).
- Explicit abstention path (`abstain` mode) when grounding is weak.
- Structured output contract (`hits`, `metrics`, `confidence`).
- Included A/B harness with measurable outcomes.

## Repo Layout

- `src/agentic_rag_plugin/` plugin code + manifest
- `tools/run_openclaw_agentic_rag_ab.py` A/B evaluation runner
- `assays/openclaw_agentic_rag_ab_suite_v1.json` sample evaluation suite
- `data/corpus_demo.json` sample retrieval corpus
- `docs/openclaw_agentic_rag_plugin_spec.md` spec
- `results/` generated run artifacts

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

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

## OpenClaw-style Entrypoint

- Function: `agentic_rag_plugin.plugin:handle_tool_call`
- Manifest: `src/agentic_rag_plugin/manifest.json`

## License

MIT (see `LICENSE`).

