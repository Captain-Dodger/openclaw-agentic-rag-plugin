# OpenClaw Endpoint Mapping for `agentic_rag`

This document maps OpenClaw HTTP endpoints to the `agentic_rag` plugin adapter.

## 1) Endpoint choice

| Endpoint | Default | Best use for `agentic_rag` | Notes |
| --- | --- | --- | --- |
| `POST /tools/invoke` | Enabled | Direct deterministic tool invocation | Best for testing + automation. |
| `POST /v1/chat/completions` | Disabled | Agent-run flow (model decides tool use) | Requires enabling in gateway config. |
| `POST /v1/responses` | Disabled | Agent-run flow + Responses API clients | Requires enabling in gateway config. |

Recommendation:

- Use `/tools/invoke` for plugin validation and CI checks.
- Use `/v1/chat/completions` or `/v1/responses` for normal conversational agent loops.

## 2) Request mapping (`/tools/invoke` -> plugin -> Python core)

### HTTP request body

```json
{
  "tool": "agentic_rag",
  "args": {
    "query": "What does abstain_on_unanswerable_rate measure?",
    "state": {
      "tick": 42,
      "agent": "alpha"
    }
  },
  "sessionKey": "main"
}
```

### Mapping path

1. OpenClaw gateway receives `tool=agentic_rag`.
2. OpenClaw tool router calls `index.ts` tool `execute(...)`.
3. `index.ts` forwards payload to Python bridge:
   - `args.query` -> `payload.query`
   - `args.state` -> `payload.state`
   - plugin config -> `payload.pluginConfig`
4. Python bridge calls `AgenticRagPlugin.decide(query)` and returns structured result envelope.
5. Tool returns:
   - `content`: text summary for LLM/use
   - `details`: full structured result (`mode`, `confidence`, `metrics`, `hits`, etc.)

## 3) Plugin config mapping

Configured under `plugins.entries["agentic-rag"].config`:

| Plugin config key | Bridge key | Python config key |
| --- | --- | --- |
| `retrievalMode` | `pluginConfig.retrievalMode` | `retrieval_mode` |
| `embeddingEnabled` | `pluginConfig.embeddingEnabled` | `embedding_enabled` |
| `embeddingBaseUrl` | `pluginConfig.embeddingBaseUrl` | `embedding_base_url` |
| `embeddingModel` | `pluginConfig.embeddingModel` | `embedding_model` |
| `embeddingTimeoutMs` | `pluginConfig.embeddingTimeoutMs` | `embedding_timeout_ms` |
| `hybridLexicalWeight` | `pluginConfig.hybridLexicalWeight` | `hybrid_lexical_weight` |
| `hybridMinLexicalScore` | `pluginConfig.hybridMinLexicalScore` | `hybrid_min_lexical_score` |
| `corpusPath` | `pluginConfig.corpusPath` | corpus source path passed to `from_path(...)` (JSON file or folder) |
| `minRetrievalScore` | `pluginConfig.minRetrievalScore` | `min_retrieval_score` |
| `minConfidence` | `pluginConfig.minConfidence` | `min_confidence` |
| `topK` | `pluginConfig.topK` | `top_k` |
| `maxContextChars` | `pluginConfig.maxContextChars` | `max_context_chars` |
| `abstainMessage` | `pluginConfig.abstainMessage` | `abstain_message` |
| `arbiterMode` | `pluginConfig.arbiterMode` | `arbiter_mode` |
| `arbiterEnabled` | `pluginConfig.arbiterEnabled` | `arbiter_enabled` |
| `arbiterSharedLabel` | `pluginConfig.arbiterSharedLabel` | `arbiter_shared_label` |
| `arbiterMinEvidenceChars` | `pluginConfig.arbiterMinEvidenceChars` | `arbiter_min_evidence_chars` |
| `arbiterHighImpactMargin` | `pluginConfig.arbiterHighImpactMargin` | `arbiter_high_impact_margin` |
| `arbiterAllowRefine` | `pluginConfig.arbiterAllowRefine` | `arbiter_allow_refine` |
| `arbiterFailClosedOnConflict` | `pluginConfig.arbiterFailClosedOnConflict` | `arbiter_fail_closed_on_conflict` |
| `pythonBin` | adapter runtime only | bridge interpreter |
| `bridgeScript` | adapter runtime only | bridge file path |
| `timeoutMs` | adapter runtime only | bridge timeout |

## 4) Response mapping

OpenClaw `/tools/invoke` success shape:

```json
{
  "ok": true,
  "result": {
    "content": [{ "type": "text", "text": "..." }],
    "details": {
      "mode": "answer",
      "confidence": 0.1917,
      "answer": "...",
      "rationale": "...",
      "context": "...",
      "hits": [],
      "metrics": {}
    }
  }
}
```

### `details.mode` semantics

- `answer`: grounded evidence cleared thresholds
- `abstain`: insufficient evidence (safe refusal)
- `bridge_error` (adapter-side): Python bridge failed or timed out

## 5) Error behavior

Gateway-level:

- `400` invalid request or tool input error
- `401` auth failure
- `404` tool not found / denied by policy
- `500` tool execution failed

Adapter-level (inside `result.details.mode` when tool itself runs):

- `bridge_error`: subprocess spawn/timeout/JSON envelope failure
- `input_error`: missing/empty `query`
- `disabled`: plugin disabled by config

## 6) Copy/paste test commands

### Direct tool invocation (recommended first test)

```bash
curl -sS http://127.0.0.1:18789/tools/invoke \
  -H "Authorization: Bearer YOUR_GATEWAY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "agentic_rag",
    "args": {
      "query": "What does abstain_on_unanswerable_rate measure?"
    },
    "sessionKey": "main"
  }'
```

### OpenAI-compatible endpoint (agent decides tool usage)

```bash
curl -sS http://127.0.0.1:18789/v1/chat/completions \
  -H "Authorization: Bearer YOUR_GATEWAY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openclaw:main",
    "messages": [
      {"role": "user", "content": "Use agentic_rag to explain abstain_on_unanswerable_rate."}
    ]
  }'
```

### OpenResponses endpoint (agent decides tool usage)

```bash
curl -sS http://127.0.0.1:18789/v1/responses \
  -H "Authorization: Bearer YOUR_GATEWAY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openclaw:main",
    "input": "Use agentic_rag to explain abstain_on_unanswerable_rate."
  }'
```

## 7) OpenClaw references used

- `src/gateway/tools-invoke-http.ts`
- `src/gateway/server-http.ts`
- `docs/zh-CN/gateway/tools-invoke-http-api.md`
- `docs/zh-CN/gateway/openai-http-api.md`
- `docs/zh-CN/gateway/openresponses-http-api.md`
- `docs/plugins/agent-tools.md`
- `docs/plugins/manifest.md`
