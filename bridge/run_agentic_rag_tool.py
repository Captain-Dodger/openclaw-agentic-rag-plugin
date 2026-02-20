#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_rag_plugin.config import AgenticRagConfig  # noqa: E402
from agentic_rag_plugin.plugin import AgenticRagPlugin  # noqa: E402


def _read_payload() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("payload must be a JSON object")
    return obj


def _sanitize_number(value: Any, default: float | int, lo: float | int, hi: float | int) -> float | int:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        num = value
    elif isinstance(value, str):
        try:
            num = float(value.strip())
        except ValueError:
            return default
    else:
        return default
    if num < lo:
        return lo
    if num > hi:
        return hi
    if isinstance(default, int):
        return int(num)
    return float(num)


def _sanitize_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "on"}:
            return True
        if v in {"0", "false", "no", "off"}:
            return False
    return default


def _config_from_plugin_config(cfg: dict[str, Any]) -> AgenticRagConfig:
    retrieval_mode = str(cfg.get("retrievalMode") or "lexical").strip().lower()
    if retrieval_mode not in {"lexical", "hybrid"}:
        retrieval_mode = "lexical"
    return AgenticRagConfig(
        top_k=int(_sanitize_number(cfg.get("topK"), 4, 1, 32)),
        min_retrieval_score=float(_sanitize_number(cfg.get("minRetrievalScore"), 0.18, 0.0, 1.0)),
        min_confidence=float(_sanitize_number(cfg.get("minConfidence"), 0.45, 0.0, 1.0)),
        max_context_chars=int(_sanitize_number(cfg.get("maxContextChars"), 1400, 64, 50000)),
        abstain_message=str(
            cfg.get("abstainMessage")
            or "I do not have enough grounded evidence in indexed sources to answer safely."
        ),
        retrieval_mode=retrieval_mode,
        embedding_enabled=_sanitize_bool(cfg.get("embeddingEnabled", False), False),
        embedding_base_url=str(cfg.get("embeddingBaseUrl") or "http://127.0.0.1:1234/v1"),
        embedding_model=str(cfg.get("embeddingModel") or "text-embedding-nomic-embed-text-v1.5"),
        embedding_timeout_ms=int(_sanitize_number(cfg.get("embeddingTimeoutMs"), 10000, 500, 120000)),
        hybrid_lexical_weight=float(_sanitize_number(cfg.get("hybridLexicalWeight"), 0.35, 0.0, 1.0)),
        hybrid_min_lexical_score=float(_sanitize_number(cfg.get("hybridMinLexicalScore"), 0.0, 0.0, 1.0)),
        arbiter_enabled=_sanitize_bool(cfg.get("arbiterEnabled", False), False),
        arbiter_shared_label=str(cfg.get("arbiterSharedLabel") or "contracts_v1"),
        arbiter_min_evidence_chars=int(
            _sanitize_number(cfg.get("arbiterMinEvidenceChars"), 120, 16, 20000)
        ),
        arbiter_high_impact_margin=float(
            _sanitize_number(cfg.get("arbiterHighImpactMargin"), 0.10, 0.0, 1.0)
        ),
        arbiter_allow_refine=_sanitize_bool(cfg.get("arbiterAllowRefine", True), True),
        arbiter_fail_closed_on_conflict=_sanitize_bool(
            cfg.get("arbiterFailClosedOnConflict", True),
            True,
        ),
    )


def _corpus_path_from_plugin_config(cfg: dict[str, Any]) -> Path:
    candidate = cfg.get("corpusPath")
    if isinstance(candidate, str) and candidate.strip():
        path = Path(candidate.strip())
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        return path
    return ROOT / "data" / "corpus_demo.json"


def _ok(result: dict[str, Any]) -> None:
    print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))


def _err(message: str, *, tb: str | None = None) -> None:
    out: dict[str, Any] = {"ok": False, "error": message}
    if tb:
        out["traceback"] = tb
    print(json.dumps(out, ensure_ascii=False))


def main() -> int:
    try:
        payload = _read_payload()
        query = str(payload.get("query", "")).strip()
        if not query:
            _err("query required")
            return 2

        plugin_cfg = payload.get("pluginConfig")
        if not isinstance(plugin_cfg, dict):
            plugin_cfg = {}

        corpus_path = _corpus_path_from_plugin_config(plugin_cfg)
        if not corpus_path.exists():
            _err(f"corpus not found: {corpus_path}")
            return 3

        rag_cfg = _config_from_plugin_config(plugin_cfg)
        plugin = AgenticRagPlugin.from_path(corpus_path, config=rag_cfg)
        decision = plugin.decide(query)
        result = decision.as_dict()
        result["tool"] = "agentic_rag"
        result["query"] = query
        state = payload.get("state")
        if isinstance(state, dict):
            result["state_hint"] = {"keys": sorted([str(k) for k in state.keys()])[:8]}
        _ok(result)
        return 0
    except Exception as exc:  # noqa: BLE001
        _err(str(exc), tb=traceback.format_exc(limit=8))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
