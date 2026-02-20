from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from .config import AgenticRagConfig
from .types import Document, RagDecision, RetrievalHit


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


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


def _vector_norm(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    a_norm = _vector_norm(a)
    b_norm = _vector_norm(b)
    if a_norm <= 0 or b_norm <= 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (a_norm * b_norm)


class EmbeddingHttpClient:
    """Small OpenAI-compatible embeddings client (LM Studio compatible)."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_ms: int,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_ms = max(500, timeout_ms)
        self.api_key = api_key

    def _endpoint(self) -> str:
        if self.base_url.endswith("/embeddings"):
            return self.base_url
        return f"{self.base_url}/embeddings"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self.model, "input": texts}
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urlrequest.Request(self._endpoint(), data=body, headers=headers, method="POST")
        try:
            with urlrequest.urlopen(req, timeout=self.timeout_ms / 1000.0) as resp:
                raw = resp.read().decode("utf-8")
        except urlerror.HTTPError as exc:
            raise RuntimeError(f"embedding http error: {exc.code} {exc.reason}") from exc
        except urlerror.URLError as exc:
            raise RuntimeError(f"embedding url error: {exc.reason}") from exc
        parsed = json.loads(raw)
        data = parsed.get("data")
        if not isinstance(data, list):
            raise RuntimeError("embedding response missing data[]")

        vectors: list[list[float] | None] = [None] * len(texts)
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            index = item.get("index", i)
            if not isinstance(index, int) or index < 0 or index >= len(texts):
                continue
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                continue
            vector = [float(v) for v in embedding if isinstance(v, (int, float))]
            if vector:
                vectors[index] = vector

        if any(v is None for v in vectors):
            raise RuntimeError("embedding response incomplete for requested inputs")
        return [v for v in vectors if v is not None]


class AgenticRagPlugin:
    """Model-agnostic retrieval + confidence gating for tool-use pipelines."""

    def __init__(self, docs: list[Document], config: AgenticRagConfig | None = None) -> None:
        self.docs = docs
        self.config = config or AgenticRagConfig()
        self._embedding_client: EmbeddingHttpClient | None = None
        self._doc_embeddings: list[list[float]] | None = None
        self._embedding_error: str | None = None
        self._embedding_failed: bool = False
        self._last_retrieval_meta: dict[str, Any] = {}

        if self.config.embedding_enabled and self.config.retrieval_mode.casefold() == "hybrid":
            self._embedding_client = EmbeddingHttpClient(
                base_url=self.config.embedding_base_url,
                model=self.config.embedding_model,
                timeout_ms=self.config.embedding_timeout_ms,
                api_key=os.getenv("OPENCLAW_AGENTIC_RAG_EMBEDDING_API_KEY"),
            )

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

    def _set_retrieval_meta(self, **kwargs: Any) -> None:
        self._last_retrieval_meta = kwargs

    def _retrieve_lexical(self, query: str, *, effective_mode: str, note: str | None = None) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        for doc in self.docs:
            lexical = _score(query, doc.text)
            if lexical <= 0:
                continue
            hits.append(
                RetrievalHit(
                    doc_id=doc.id,
                    source=doc.source,
                    score=lexical,
                    text=doc.text,
                    lexical_score=lexical,
                    semantic_score=None,
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        self._set_retrieval_meta(
            retrieval_mode_requested=self.config.retrieval_mode,
            retrieval_mode_effective=effective_mode,
            embedding_enabled=self.config.embedding_enabled,
            embedding_ready=self._doc_embeddings is not None,
            embedding_error=self._embedding_error,
            note=note,
        )
        return hits[: self.config.top_k]

    def _ensure_doc_embeddings(self) -> bool:
        if self._embedding_client is None:
            return False
        if self._embedding_failed:
            return False
        if self._doc_embeddings is not None:
            return True
        if not self.docs:
            self._doc_embeddings = []
            self._embedding_error = None
            return True
        try:
            self._doc_embeddings = self._embedding_client.embed_texts([d.text for d in self.docs])
            self._embedding_error = None
            return True
        except Exception as exc:  # noqa: BLE001
            self._embedding_error = str(exc)
            self._doc_embeddings = None
            self._embedding_failed = True
            return False

    def _semantic_scores(self, query: str) -> dict[str, float]:
        if self._embedding_client is None:
            return {}
        if not self._ensure_doc_embeddings():
            return {}
        assert self._doc_embeddings is not None
        if len(self._doc_embeddings) != len(self.docs):
            self._embedding_error = "doc embedding cardinality mismatch"
            return {}
        try:
            q_vec = self._embedding_client.embed_texts([query])[0]
        except Exception as exc:  # noqa: BLE001
            self._embedding_error = str(exc)
            self._embedding_failed = True
            return {}
        out: dict[str, float] = {}
        for idx, doc in enumerate(self.docs):
            sem = max(0.0, _cosine_similarity(q_vec, self._doc_embeddings[idx]))
            if sem > 0:
                out[doc.id] = round(sem, 4)
        return out

    def _retrieve_hybrid(self, query: str) -> list[RetrievalHit]:
        semantic_by_id = self._semantic_scores(query)
        if not semantic_by_id:
            return self._retrieve_lexical(
                query,
                effective_mode="lexical_fallback",
                note="hybrid requested but semantic stream unavailable",
            )

        lexical_weight = _clamp(float(self.config.hybrid_lexical_weight), 0.0, 1.0)
        semantic_weight = 1.0 - lexical_weight
        lexical_floor = _clamp(float(self.config.hybrid_min_lexical_score), 0.0, 1.0)
        hits: list[RetrievalHit] = []
        for doc in self.docs:
            lexical = _score(query, doc.text)
            semantic = float(semantic_by_id.get(doc.id, 0.0))
            if lexical < lexical_floor:
                continue
            combined = round((lexical_weight * lexical) + (semantic_weight * semantic), 4)
            if combined <= 0:
                continue
            hits.append(
                RetrievalHit(
                    doc_id=doc.id,
                    source=doc.source,
                    score=combined,
                    text=doc.text,
                    lexical_score=round(lexical, 4),
                    semantic_score=round(semantic, 4),
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        self._set_retrieval_meta(
            retrieval_mode_requested=self.config.retrieval_mode,
            retrieval_mode_effective="hybrid",
            embedding_enabled=self.config.embedding_enabled,
            embedding_ready=True,
            embedding_error=self._embedding_error,
            hybrid_lexical_weight=lexical_weight,
            hybrid_semantic_weight=semantic_weight,
            hybrid_min_lexical_score=lexical_floor,
            note=None,
        )
        return hits[: self.config.top_k]

    def retrieve(self, query: str) -> list[RetrievalHit]:
        mode = (self.config.retrieval_mode or "lexical").casefold()
        if mode == "hybrid" and self.config.embedding_enabled:
            return self._retrieve_hybrid(query)
        return self._retrieve_lexical(query, effective_mode="lexical")

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
            if hit.semantic_score is None:
                answer_lines.append(f"- [{hit.source}] {snippet}")
            else:
                answer_lines.append(
                    f"- [{hit.source}] (combined={hit.score}, lex={hit.lexical_score}, "
                    f"sem={hit.semantic_score}) {snippet}"
                )
            if evidence_chars >= self.config.max_context_chars:
                break

        context = "\n".join(context_parts)[: self.config.max_context_chars]
        metrics = {
            "top_score": top,
            "mean_top2": round(mean_top2, 4),
            "evidence_chars": len(context),
            "hits": len(hits),
            "retrieval_mode_requested": self._last_retrieval_meta.get("retrieval_mode_requested"),
            "retrieval_mode_effective": self._last_retrieval_meta.get("retrieval_mode_effective"),
            "embedding_enabled": self._last_retrieval_meta.get("embedding_enabled"),
            "embedding_ready": self._last_retrieval_meta.get("embedding_ready"),
        }
        if self._last_retrieval_meta.get("embedding_error"):
            metrics["embedding_error"] = self._last_retrieval_meta["embedding_error"]
        if self._last_retrieval_meta.get("hybrid_lexical_weight") is not None:
            metrics["hybrid_lexical_weight"] = self._last_retrieval_meta["hybrid_lexical_weight"]
            metrics["hybrid_semantic_weight"] = self._last_retrieval_meta["hybrid_semantic_weight"]
            metrics["hybrid_min_lexical_score"] = self._last_retrieval_meta["hybrid_min_lexical_score"]
        if self._last_retrieval_meta.get("note"):
            metrics["retrieval_note"] = self._last_retrieval_meta["note"]

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
                metrics=metrics,
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
            metrics=metrics,
        )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


_DEFAULT_PLUGIN: AgenticRagPlugin | None = None


def _default_plugin() -> AgenticRagPlugin:
    global _DEFAULT_PLUGIN
    if _DEFAULT_PLUGIN is not None:
        return _DEFAULT_PLUGIN
    corpus_path = os.getenv(
        "OPENCLAW_AGENTIC_RAG_CORPUS",
        str(Path(__file__).resolve().parents[2] / "data" / "corpus_demo.json"),
    )
    config = AgenticRagConfig(
        retrieval_mode=os.getenv("OPENCLAW_AGENTIC_RAG_RETRIEVAL_MODE", "lexical"),
        embedding_enabled=_env_bool("OPENCLAW_AGENTIC_RAG_EMBEDDING_ENABLED", False),
        embedding_base_url=os.getenv("OPENCLAW_AGENTIC_RAG_EMBEDDING_BASE_URL", "http://127.0.0.1:1234/v1"),
        embedding_model=os.getenv(
            "OPENCLAW_AGENTIC_RAG_EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5"
        ),
        embedding_timeout_ms=_env_int("OPENCLAW_AGENTIC_RAG_EMBEDDING_TIMEOUT_MS", 10000),
        hybrid_lexical_weight=_env_float("OPENCLAW_AGENTIC_RAG_HYBRID_LEXICAL_WEIGHT", 0.35),
        hybrid_min_lexical_score=_env_float("OPENCLAW_AGENTIC_RAG_HYBRID_MIN_LEXICAL_SCORE", 0.0),
    )
    _DEFAULT_PLUGIN = AgenticRagPlugin.from_json_path(corpus_path, config=config)
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
