from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgenticRagConfig:
    """Control plane for retrieval + abstention behavior."""

    top_k: int = 4
    min_retrieval_score: float = 0.18
    min_confidence: float = 0.45
    max_context_chars: int = 1400
    abstain_message: str = (
        "I do not have enough grounded evidence in indexed sources to answer safely."
    )

