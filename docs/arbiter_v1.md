# arbiter_v1 (Experimental)

`arbiter_v1` adds an optional multi-step arbitration layer on top of retrieval gating.

It is disabled by default and does not change baseline behavior unless explicitly enabled.

## The 6 inserted points

1. **Three roles with fixed responsibilities**
   - `evidence`: source sufficiency and grounding quality
   - `action`: answer vs refine vs abstain
   - `policy`: risk and safety guard

2. **Skillgraph-style routing metadata**
   - Each role has 2 unique skill labels.
   - All roles share one label (`arbiterSharedLabel`).
   - Labels are emitted in diagnostics (`metrics.arbiter_v1.skills_by_role`).

3. **State-machine path per decision**
   - `observe -> orbit -> debate -> decide`
   - Emitted in `metrics.arbiter_v1.state_machine`.

4. **Packet contract**
   - `role`
   - `claim`
   - `evidence_refs`
   - `confidence`
   - `risk`
   - `proposed_action`
   - `missing_info`

5. **Hard guardrails**
   - extra confidence margin for high-impact queries
   - fail-closed on role conflict (`arbiterFailClosedOnConflict`)
   - minimum evidence chars before answer (`arbiterMinEvidenceChars`)

6. **Operational metrics**
   - conflict flag
   - high-impact flag
   - observed retrieval stats (`top_score`, `mean_top2`, `confidence`, `evidence_chars`)
   - decision action (`answer|abstain|refine_query`)

## Config keys

- `arbiterEnabled` (bool)
- `arbiterSharedLabel` (string)
- `arbiterMinEvidenceChars` (int)
- `arbiterHighImpactMargin` (float)
- `arbiterAllowRefine` (bool)
- `arbiterFailClosedOnConflict` (bool)

## What this would do in Gridworld

If you apply the same pattern in Gridworld, it typically changes behavior like this:

- More explicit separation between perception, intention, and safety.
- Fewer impulsive actions during ambiguous ticks (more controlled abstention/refinement).
- Better post-hoc diagnostics: you can see which role vetoed action and why.
- Cleaner drift triage: conflicts become a signal (`conflict=true`) rather than hidden instability.
- Better iteration loop: tune one gate at a time (evidence floor, conflict policy, high-impact margin).

In short: less hidden coupling, more measurable agency.
