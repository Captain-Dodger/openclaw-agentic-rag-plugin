# OpenClaw Agentic RAG Plugin (MIT)

Model-agnostic retrieval plugin scaffold with confidence-gated abstention.

This repo now ships a **clean adapter layer**:

- OpenClaw-native plugin surface (`openclaw.plugin.json`, `index.ts`)
- Python core preserved as-is (`src/agentic_rag_plugin/*`)
- Thin bridge between both (`bridge/run_agentic_rag_tool.py`)

Goal: keep your own logic style while docking cleanly into OpenClaw.

## Why this exists

Agent tool chains often force a response even when evidence is weak.  
This plugin separates two paths:

- `answer` when retrieval evidence is strong enough
- `abstain` when evidence is weak

That behavior is measurable via A/B.

## Core behavior

For each query:

1. retrieve top-k local hits
   - lexical mode: token overlap + phrase boost
   - hybrid mode: lexical + embedding cosine fusion
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

## OpenClaw adapter layer

### Files

- `openclaw.plugin.json`: plugin manifest + config schema
- `package.json`: declares OpenClaw extension entrypoint
- `index.ts`: registers optional tool `agentic_rag`
- `bridge/run_agentic_rag_tool.py`: executes Python RAG core and returns JSON

### OpenClaw config example

```json5
{
  plugins: {
    load: {
      paths: ["D:/path/to/openclaw-agentic-rag-plugin"]
    },
    entries: {
      "agentic-rag": {
        enabled: true,
        config: {
          pythonBin: "python",
          retrievalMode: "hybrid",
          embeddingEnabled: true,
          embeddingBaseUrl: "http://127.0.0.1:1234/v1",
          embeddingModel: "text-embedding-nomic-embed-text-v1.5",
          hybridLexicalWeight: 0.35,
          hybridMinLexicalScore: 0.10,
          corpusPath: "data/corpus_demo.json",
          minRetrievalScore: 0.12,
          minConfidence: 0.12,
          topK: 4,
          maxContextChars: 1400,
          timeoutMs: 15000
        }
      }
    }
  },
  tools: {
    allow: ["group:core", "agentic_rag"]
  }
}
```

Notes:

- `agentic_rag` is registered as an **optional** plugin tool.
- Relative paths in plugin config are resolved against plugin root.
- Ready-to-copy local config preset:
  - `openclaw.plugin.local.example.json`
- Endpoint-level mapping and ready-to-run examples:
  - `docs/openclaw_endpoint_mapping.md`

## Quick start (local A/B)

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

### Hybrid mode (embeddings enabled)

```bash
python tools/run_openclaw_agentic_rag_ab.py \
  --suite assays/openclaw_agentic_rag_ab_suite_v1.json \
  --corpus data/corpus_demo.json \
  --plugin-retrieval-mode hybrid \
  --embedding-enabled \
  --embedding-base-url http://127.0.0.1:1234/v1 \
  --embedding-model text-embedding-nomic-embed-text-v1.5 \
  --hybrid-lexical-weight 0.35 \
  --hybrid-min-lexical-score 0.10 \
  --min-retrieval-score 0.12 \
  --min-confidence 0.12
```

`--hybrid-min-lexical-score` is the anti-overreach knob for hybrid mode:
it drops semantic-only hits that have weak lexical anchor.

### Lexical vs hybrid compare

```bash
python tools/run_openclaw_agentic_rag_retrieval_compare.py \
  --suite assays/openclaw_agentic_rag_ab_suite_v1.json \
  --corpus data/corpus_demo.json \
  --embedding-base-url http://127.0.0.1:1234/v1 \
  --embedding-model text-embedding-nomic-embed-text-v1.5 \
  --hybrid-lexical-weight 0.35 \
  --hybrid-min-lexical-score 0.10
```

Outputs:

- `results/openclaw_agentic_rag_runs/<run_id>/summary.json`
- `results/openclaw_agentic_rag_runs/<run_id>/report.md`
- `results/openclaw_agentic_rag_runs/<run_id>/per_case.csv`

## Benchmark snapshot (demo suite)

From `results/openclaw_agentic_rag_runs/run_20260219_231559/summary.json`:

- baseline `abstain_on_unanswerable_rate`: `0.0`
- plugin `abstain_on_unanswerable_rate`: `1.0`
- baseline `hallucination_rate_on_unanswerable`: `1.0`
- plugin `hallucination_rate_on_unanswerable`: `0.0`
- `grounded_answer_rate_on_answerable`: unchanged (`1.0 -> 1.0`)

This is a small demo suite, not a production claim.

## Repository layout

- `src/agentic_rag_plugin/` Python core
- `index.ts` OpenClaw tool adapter
- `bridge/run_agentic_rag_tool.py` Python bridge entrypoint
- `tools/run_openclaw_agentic_rag_ab.py` A/B runner
- `tools/run_openclaw_agentic_rag_retrieval_compare.py` lexical vs hybrid compare
- `assays/openclaw_agentic_rag_ab_suite_v1.json` sample suite
- `data/corpus_demo.json` sample corpus
- `docs/openclaw_agentic_rag_plugin_spec.md` protocol/spec
- `results/` run artifacts

## Limitations

- Hybrid mode requires an OpenAI-compatible embedding endpoint.
- With hybrid mode, semantic-only matches can over-answer if no lexical anchor is enforced.
- Large corpora should use a persistent vector index (current v0 computes in-memory vectors).
- Baseline in A/B is intentionally naive.
- No network retrieval in this repo.
- Thresholds are corpus-dependent and must be tuned.

## License

MIT (see `LICENSE`).
