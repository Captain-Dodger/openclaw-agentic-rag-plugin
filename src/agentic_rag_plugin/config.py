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
    retrieval_mode: str = "lexical"  # lexical | hybrid
    embedding_enabled: bool = False
    embedding_base_url: str = "http://127.0.0.1:1234/v1"
    embedding_model: str = "text-embedding-nomic-embed-text-v1.5"
    embedding_timeout_ms: int = 10000
    hybrid_lexical_weight: float = 0.35
    hybrid_min_lexical_score: float = 0.0
