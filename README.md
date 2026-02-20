# OpenClaw Agentic RAG Plugin (MIT)

OpenClaw plugin scaffold for retrieval-gated answering.

The repository contains:

- OpenClaw-native plugin surface (`openclaw.plugin.json`, `index.ts`)
- Python retrieval core (`src/agentic_rag_plugin/*`)
- Bridge process (`bridge/run_agentic_rag_tool.py`)

## What it does

The plugin separates two paths:

- `answer` when retrieval evidence is strong enough
- `abstain` when evidence is weak

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

## Optional: arbiter_v1 (experimental)

You can enable an optional 3-role arbitration pass:

- `evidence` role
- `action` role
- `policy` role

This layer stays interpreted and emits diagnostics in `metrics.arbiter_v1`.
Default is `off` (no behavior change from baseline).

Config keys:

- `arbiterMode` (`off|shadow|enforce`)
- `arbiterEnabled`
- `arbiterSharedLabel`
- `arbiterMinEvidenceChars`
- `arbiterHighImpactMargin`
- `arbiterAllowRefine`
- `arbiterFailClosedOnConflict`

Details:

- `docs/arbiter_v1.md`

## Corpus source modes

`corpusPath` can point to:

- a JSON corpus file (`[{id, source, text}, ...]`)
- a folder (recursive ingest) with supported files:
  - `.txt`, `.md`, `.markdown`, `.rst`, `.log`, `.pdf`, `.odt`

PDF extraction is optional and requires `pypdf` to be installed in your Python environment.
ODT extraction uses built-in XML parsing.

## OpenClaw adapter layer

### Files

- `openclaw.plugin.json`: plugin manifest + config schema
- `package.json`: declares OpenClaw extension entrypoint
- `index.ts`: registers optional tool `agentic_rag`
- `bridge/run_agentic_rag_tool.py`: executes Python RAG core and returns JSON

## Installation (OpenClaw standard setup)

1. Clone this repository.
2. Install Python package (recommended in virtualenv):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

3. Add plugin path to your OpenClaw config:

```json5
{
  plugins: {
    load: {
      paths: ["D:/path/to/openclaw-agentic-rag-plugin"]
    }
  }
}
```

4. Add plugin entry config:
- Start from one of:
  - `configs/plugin.entry.lexical.example.json`
  - `configs/plugin.entry.hybrid.lmstudio.example.json`
  - `configs/plugin.entry.hybrid.openai-compatible.example.json`

5. Allow tool `agentic_rag` in your tool policy.

6. Restart OpenClaw gateway.

### Gateway token (for authenticated `/tools/invoke` tests)

Set a gateway token in your OpenClaw environment (example):

```bash
mkdir -p ~/.openclaw
cat >> ~/.openclaw/.env <<'EOF'
OPENCLAW_GATEWAY_TOKEN=replace-with-a-long-random-token
EOF
```

Then restart the OpenClaw gateway process so it picks up the new env value.

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
          arbiterMode: "off",
          arbiterEnabled: false,
          arbiterSharedLabel: "contracts_v1",
          arbiterMinEvidenceChars: 120,
          arbiterHighImpactMargin: 0.1,
          arbiterAllowRefine: true,
          arbiterFailClosedOnConflict: true,
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
- Endpoint/profile presets (easy to edit):
  - `configs/plugin.entry.lexical.example.json`
  - `configs/plugin.entry.hybrid.lmstudio.example.json`
  - `configs/plugin.entry.hybrid.openai-compatible.example.json`
- Folder-to-JSON helper:
  - `tools/build_corpus_from_folder.py`
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

### Build corpus from folder (RAG-style)

```bash
python tools/build_corpus_from_folder.py \
  --input /path/to/knowledge_folder \
  --output data/corpus_from_folder.json \
  --chunk-chars 1200 \
  --overlap-chars 120
```

Then set plugin config:

```json5
{
  corpusPath: "data/corpus_from_folder.json"
}
```

Or point directly to a folder:

```json5
{
  corpusPath: "D:/path/to/knowledge_folder"
}
```

### Smoke test against OpenClaw gateway

```bash
curl -sS http://127.0.0.1:18789/tools/invoke \
  -H "Authorization: Bearer YOUR_GATEWAY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "agentic_rag",
    "args": {"query": "Which retrieval modes are supported?"},
    "sessionKey": "main"
  }'
```

If this returns `{"error":{"message":"Unauthorized"...}}`, the gateway is reachable and you only need a valid token.

### Bridge smoke (no gateway auth required)

```bash
printf '%s' '{
  "query":"Which retrieval modes are supported?",
  "pluginConfig":{
    "corpusPath":"data/corpus_demo.json",
    "minRetrievalScore":0.12,
    "minConfidence":0.12,
    "topK":4
  }
}' | python bridge/run_agentic_rag_tool.py
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

## Evaluation output

Run artifacts are written to:

- `results/openclaw_agentic_rag_runs/<run_id>/summary.json`
- `results/openclaw_agentic_rag_runs/<run_id>/report.md`
- `results/openclaw_agentic_rag_runs/<run_id>/per_case.csv`

Treat demo-suite results as local verification, not as universal performance claims.

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
