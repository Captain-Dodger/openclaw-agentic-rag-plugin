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
    # Optional multi-role arbitration layer (off by default).
    arbiter_mode: str = "off"  # off | shadow | enforce
    arbiter_enabled: bool = False
    arbiter_shared_label: str = "contracts_v1"
    arbiter_evidence_skills: tuple[str, str] = ("retrieval_quality", "source_attribution")
    arbiter_action_skills: tuple[str, str] = ("tool_selection", "query_refinement")
    arbiter_policy_skills: tuple[str, str] = ("risk_scoring", "policy_gate")
    arbiter_min_evidence_chars: int = 120
    arbiter_high_impact_margin: float = 0.10
    arbiter_allow_refine: bool = True
    arbiter_fail_closed_on_conflict: bool = True
