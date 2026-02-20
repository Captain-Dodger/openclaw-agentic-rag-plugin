from __future__ import annotations

import re
from typing import Any

from .config import AgenticRagConfig
from .types import ArbiterPacket, RetrievalHit


HIGH_IMPACT_RE = re.compile(
    r"(rm\s+-rf|drop\s+table|delete\s+all|shutdown|credential|password|api[_ -]?key|token|exploit)",
    re.IGNORECASE,
)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _high_impact_query(query: str) -> bool:
    return bool(HIGH_IMPACT_RE.search(query or ""))


def _arbiter_packet(
    *,
    role: str,
    claim: str,
    evidence_refs: list[str],
    confidence: float,
    risk: float,
    proposed_action: str,
    missing_info: list[str] | None = None,
) -> ArbiterPacket:
    return ArbiterPacket(
        role=role,
        claim=claim,
        evidence_refs=evidence_refs,
        confidence=round(_clamp(confidence, 0.0, 1.0), 4),
        risk=round(_clamp(risk, 0.0, 1.0), 4),
        proposed_action=proposed_action,
        missing_info=missing_info or [],
    )


def run_arbiter_v1(
    *,
    query: str,
    hits: list[RetrievalHit],
    top_score: float,
    mean_top2: float,
    confidence: float,
    evidence_chars: int,
    grounded_candidate: bool,
    config: AgenticRagConfig,
) -> dict[str, Any]:
    """Three-role arbitration pass for answer/abstain decisions.

    This keeps existing single-step behavior intact and only adds an optional
    interpreted decision layer that can be toggled via config.
    """
    evidence_refs = [h.doc_id for h in hits[:3]]
    missing: list[str] = []
    if not hits:
        missing.append("retrieval_hits")
    if evidence_chars < config.arbiter_min_evidence_chars:
        missing.append("evidence_chars")

    # 1) Evidence role.
    grounding_strength = 0.60 * top_score + 0.40 * mean_top2
    grounding_ok = grounded_candidate and evidence_chars >= config.arbiter_min_evidence_chars
    grounding_action = "answer" if grounding_ok else "abstain"
    grounding = _arbiter_packet(
        role="evidence",
        claim=(
            "Evidence is sufficient and source-grounded."
            if grounding_ok
            else "Evidence is weak or sparse for a grounded answer."
        ),
        evidence_refs=evidence_refs,
        confidence=grounding_strength,
        risk=1.0 - grounding_strength,
        proposed_action=grounding_action,
        missing_info=missing,
    )

    # 2) Action role.
    near_threshold = bool(hits) and not grounding_ok and confidence >= (config.min_confidence * 0.75)
    action_mode = "answer" if grounding_ok else "abstain"
    refine_query: str | None = None
    if near_threshold and config.arbiter_allow_refine:
        action_mode = "refine_query"
        refine_query = f"{query.strip()} (use exact terms from indexed sources)"
    action_conf = 0.50 * confidence + 0.50 * grounding_strength
    action = _arbiter_packet(
        role="action",
        claim=(
            "Proceed with grounded response."
            if action_mode == "answer"
            else ("Refine query and retry retrieval." if action_mode == "refine_query" else "Do not answer yet.")
        ),
        evidence_refs=evidence_refs[:2],
        confidence=action_conf,
        risk=1.0 - action_conf,
        proposed_action=action_mode,
        missing_info=missing,
    )

    # 3) Policy role.
    high_impact = _high_impact_query(query)
    gov_risk = 0.20
    if not grounding_ok:
        gov_risk += 0.30
    if high_impact:
        gov_risk += 0.35
    gov_risk = _clamp(gov_risk, 0.0, 1.0)
    gov_action = "abstain" if (gov_risk >= 0.55 or (high_impact and confidence < (config.min_confidence + config.arbiter_high_impact_margin))) else "answer"
    governance = _arbiter_packet(
        role="policy",
        claim=(
            "Policy/risk gates allow response."
            if gov_action == "answer"
            else "Policy/risk gates require abstention or clarification."
        ),
        evidence_refs=evidence_refs[:1],
        confidence=1.0 - gov_risk,
        risk=gov_risk,
        proposed_action=gov_action,
        missing_info=(["policy_clearance"] if gov_action == "abstain" else []),
    )

    packets = [grounding, action, governance]

    # Debate + decide.
    proposed = [p.proposed_action for p in packets]
    conflict = len(set(proposed)) > 1
    decision_mode = "answer"
    decision_action = "answer"
    reasons: list[str] = []

    if high_impact and confidence < (config.min_confidence + config.arbiter_high_impact_margin):
        decision_mode = "abstain"
        decision_action = "abstain"
        reasons.append("high_impact_margin_guard")
    elif conflict and config.arbiter_fail_closed_on_conflict:
        decision_mode = "abstain"
        decision_action = "abstain"
        reasons.append("arbiter_conflict_guard")
    elif action.proposed_action == "refine_query":
        decision_mode = "abstain"
        decision_action = "refine_query"
        reasons.append("refine_before_answer")
    elif grounding.proposed_action != "answer" or governance.proposed_action != "answer":
        decision_mode = "abstain"
        decision_action = "abstain"
        reasons.append("consensus_not_reached")
    else:
        reasons.append("consensus_answer")

    return {
        "mode": decision_mode,
        "action": decision_action,
        "rationale": "; ".join(reasons),
        "refine_query": refine_query,
        "packets": [p.as_dict() for p in packets],
        "metrics": {
            "state_machine": ["observe", "orbit", "debate", "decide"],
            "conflict": conflict,
            "high_impact_query": high_impact,
            "grounded_candidate": grounded_candidate,
            "decision_action": decision_action,
            "shared_label": config.arbiter_shared_label,
            "skills_by_role": {
                "evidence": list(config.arbiter_evidence_skills) + [config.arbiter_shared_label],
                "action": list(config.arbiter_action_skills) + [config.arbiter_shared_label],
                "policy": list(config.arbiter_policy_skills) + [config.arbiter_shared_label],
            },
            "observed": {
                "top_score": round(top_score, 4),
                "mean_top2": round(mean_top2, 4),
                "confidence": round(confidence, 4),
                "evidence_chars": int(evidence_chars),
                "hit_count": len(hits),
            },
        },
    }
