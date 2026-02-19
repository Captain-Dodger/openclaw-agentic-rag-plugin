from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    source: str


@dataclass(frozen=True)
class RetrievalHit:
    doc_id: str
    source: str
    score: float
    text: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RagDecision:
    mode: str
    confidence: float
    answer: str
    rationale: str
    context: str
    hits: list[RetrievalHit]
    metrics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["hits"] = [h.as_dict() for h in self.hits]
        return out

