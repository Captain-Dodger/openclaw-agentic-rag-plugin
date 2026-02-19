from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .config import AgenticRagConfig
from .types import Document, RagDecision, RetrievalHit


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)


def _normalize(text: str) -> str:
    return " ".join((text or "").casefold().split())


def _tokens(text: str) -> set[str]:
    return {t for t in TOKEN_RE.findall(_normalize(text)) if t}


def _score(query: str, doc: str) -> float:
    q_tokens = _tokens(query)
    d_tokens = _tokens(doc)
    if not q_tokens or not d_tokens:
        return 0.0
    overlap = len(q_tokens & d_tokens)
    union = len(q_tokens | d_tokens)
    token_overlap = overlap / max(1, len(q_tokens))
    jaccard = overlap / max(1, union)
    qn = _normalize(query)
    dn = _normalize(doc)
    phrase_boost = 0.15 if qn and qn in dn else 0.0
    return round(min(1.0, (0.65 * token_overlap) + (0.35 * jaccard) + phrase_boost), 4)


class AgenticRagPlugin:
    """Model-agnostic retrieval + confidence gating for tool-use pipelines."""

    def __init__(self, docs: list[Document], config: AgenticRagConfig | None = None) -> None:
        self.docs = docs
        self.config = config or AgenticRagConfig()

    @classmethod
    def from_json_path(
        cls, path: str | Path, config: AgenticRagConfig | None = None
    ) -> "AgenticRagPlugin":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        docs = [
            Document(
                id=str(item.get("id", "")),
                text=str(item.get("text", "")),
                source=str(item.get("source", "unknown")),
            )
            for item in raw
            if isinstance(item, dict)
        ]
        return cls(docs, config=config)

    def retrieve(self, query: str) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        for doc in self.docs:
            score = _score(query, doc.text)
            if score <= 0:
                continue
            hits.append(
                RetrievalHit(
                    doc_id=doc.id,
                    source=doc.source,
                    score=score,
                    text=doc.text,
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[: self.config.top_k]

    def decide(self, query: str) -> RagDecision:
        hits = self.retrieve(query)
        top = hits[0].score if hits else 0.0
        mean_top2 = sum(h.score for h in hits[:2]) / max(1, min(2, len(hits)))
        confidence = round(min(1.0, (0.75 * top) + (0.25 * mean_top2)), 4)

        grounded = bool(
            hits
            and top >= self.config.min_retrieval_score
            and confidence >= self.config.min_confidence
        )
        evidence_chars = 0
        context_parts: list[str] = []
        answer_lines: list[str] = []
        for hit in hits:
            snippet = hit.text.strip()
            evidence_chars += len(snippet)
            context_parts.append(f"[{hit.doc_id}] ({hit.source}) {snippet}")
            answer_lines.append(f"- [{hit.source}] {snippet}")
            if evidence_chars >= self.config.max_context_chars:
                break

        context = "\n".join(context_parts)[: self.config.max_context_chars]
        if not grounded:
            return RagDecision(
                mode="abstain",
                confidence=confidence,
                answer=self.config.abstain_message,
                rationale=(
                    "No sufficient grounded evidence from retrieval "
                    f"(top_score={top}, confidence={confidence})."
                ),
                context=context,
                hits=hits,
                metrics={
                    "top_score": top,
                    "mean_top2": round(mean_top2, 4),
                    "evidence_chars": len(context),
                    "hits": len(hits),
                },
            )

        answer = "Grounded answer from retrieved evidence:\n" + "\n".join(answer_lines[:3])
        return RagDecision(
            mode="answer",
            confidence=confidence,
            answer=answer,
            rationale=(
                "Grounded by retrieval hits above thresholds "
                f"(top_score={top}, confidence={confidence})."
            ),
            context=context,
            hits=hits,
            metrics={
                "top_score": top,
                "mean_top2": round(mean_top2, 4),
                "evidence_chars": len(context),
                "hits": len(hits),
            },
        )


_DEFAULT_PLUGIN: AgenticRagPlugin | None = None


def _default_plugin() -> AgenticRagPlugin:
    global _DEFAULT_PLUGIN
    if _DEFAULT_PLUGIN is not None:
        return _DEFAULT_PLUGIN
    corpus_path = os.getenv(
        "OPENCLAW_AGENTIC_RAG_CORPUS",
        str(Path(__file__).resolve().parents[2] / "data" / "corpus_demo.json"),
    )
    _DEFAULT_PLUGIN = AgenticRagPlugin.from_json_path(corpus_path)
    return _DEFAULT_PLUGIN


def handle_tool_call(payload: dict[str, Any], state: dict[str, Any] | None = None) -> dict[str, Any]:
    """OpenClaw-style tool handler entrypoint.

    Expected payload:
    - query: str
    """
    query = str(payload.get("query", "")).strip()
    plugin = _default_plugin()
    decision = plugin.decide(query)
    out = decision.as_dict()
    out["tool"] = "agentic_rag"
    out["query"] = query
    if state:
        out["state_hint"] = {"keys": sorted(state.keys())[:8]}
    return out
