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


def _calculate_tension(grounding_ok: bool, high_impact: bool, confidence: float, conflict: bool) -> float:
    """Berechnet den epistemischen Metabolismus (Heat)."""
    tension = 0.1 # Base heat
    if not grounding_ok:
        tension += 0.3 # Uncertainty creates tension
    if high_impact:
        tension += 0.4 # High risk creates tension
    if conflict:
        tension += 0.2 # Role conflict creates tension
    
    # Mathematical Singularity: Extreme low confidence creates intense heat (Dissonance)
    if confidence < 0.3:
        tension += 0.2 
        
    return _clamp(tension, 0.0, 1.0)

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
    """Thermodynamic Arbitration pass for answer/abstain decisions.
    
    Replaces static thresholds with emergent tension modulation.
    """
    evidence_refs = [h.doc_id for h in hits[:3]]
    missing: list[str] = []
    if not hits:
        missing.append("retrieval_hits")
    if evidence_chars < config.arbiter_min_evidence_chars:
        missing.append("evidence_chars")

    high_impact = _high_impact_query(query)
    grounding_strength = 0.60 * top_score + 0.40 * mean_top2
    grounding_ok = grounded_candidate and evidence_chars >= config.arbiter_min_evidence_chars

    # 1) Calculate Thermodynamic Tension
    # Simulate a potential role conflict before computing the formal packets
    simulated_action_mode = "answer" if grounding_ok else "abstain"
    simulated_gov_risk = 0.20 + (0.30 if not grounding_ok else 0.0) + (0.35 if high_impact else 0.0)
    simulated_conflict = (simulated_action_mode == "answer" and simulated_gov_risk >= 0.55)
    
    tension = _calculate_tension(grounding_ok, high_impact, confidence, simulated_conflict)
    
    # Dynamic Theta shifts based on tension. High tension LOWERS the risk threshold, meaning the system
    # abstains earlier. Standard theta was 0.55.
    dynamic_theta_risk = _clamp(0.65 - (tension * 0.40), 0.20, 0.90)

    # 1) Evidence role.
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
    # At high tension, the arbiter refines queries more aggressively
    dynamic_refine_margin = config.min_confidence * (1.0 - (tension * 0.3))
    near_threshold = bool(hits) and not grounding_ok and confidence >= dynamic_refine_margin
    
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

    # 3) Policy role (Thermodynamically Modulated).
    gov_risk = 0.20
    if not grounding_ok:
        gov_risk += 0.30
    if high_impact:
        gov_risk += 0.35
    gov_risk = _clamp(gov_risk, 0.0, 1.0)
    
    # Use dynamic_theta_risk for strictness
    gov_action = "abstain" if (gov_risk >= dynamic_theta_risk or (high_impact and confidence < (config.min_confidence + (config.arbiter_high_impact_margin * (1.0 + tension))))) else "answer"
    
    governance = _arbiter_packet(
        role="policy",
        claim=(
            "Policy/risk gates allow response."
            if gov_action == "answer"
            else "Tension/risk gates require abstention or clarification."
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

    if high_impact and confidence < (config.min_confidence + (config.arbiter_high_impact_margin * (1.0 + tension))):
        decision_mode = "abstain"
        decision_action = "abstain"
        reasons.append("thermodynamic_high_impact_guard")
    elif conflict and config.arbiter_fail_closed_on_conflict:
        decision_mode = "abstain"
        decision_action = "abstain"
        reasons.append("thermodynamic_conflict_guard")
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
            "thermodynamics": {
                "tension": round(tension, 4),
                "dynamic_theta": round(dynamic_theta_risk, 4)
            },
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
