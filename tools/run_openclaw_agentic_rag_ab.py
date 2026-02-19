#!/usr/bin/env python3
"""Run A/B evaluation for Agentic RAG plugin scaffold.

A (baseline): naive retrieval answer mode (weak/no abstain behavior).
B (plugin): confidence-gated answer/abstain from AgenticRagPlugin.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_rag_plugin import AgenticRagConfig, AgenticRagPlugin


DEFAULT_OUT_ROOT = ROOT / "results" / "openclaw_agentic_rag_runs"
DEFAULT_SUITE = ROOT / "assays" / "openclaw_agentic_rag_ab_suite_v1.json"
DEFAULT_CORPUS = ROOT / "data" / "corpus_demo.json"


@dataclass(frozen=True)
class EvalRow:
    item_id: str
    answerable: bool
    query: str
    mode_name: str
    outcome_mode: str
    confidence: float
    top_score: float
    answered: bool
    abstained: bool
    expected_hit: bool


def _contains_any(text: str, needles: list[str]) -> bool:
    if not needles:
        return False
    blob = (text or "").casefold()
    return any((needle or "").casefold() in blob for needle in needles)


def _run_baseline(plugin: AgenticRagPlugin, query: str) -> dict[str, Any]:
    # Deliberately naive baseline: answer if any hit, no confidence abstain gate.
    hits = plugin.retrieve(query)
    top = hits[0].score if hits else 0.0
    if not hits:
        return {
            "mode": "answer",
            "confidence": 0.2,
            "answer": "Best guess: unavailable source, but likely true.",
            "metrics": {"top_score": 0.0, "hits": 0},
        }
    short = "; ".join(f"[{h.source}] {h.text}" for h in hits[:2])
    return {
        "mode": "answer",
        "confidence": max(0.35, min(1.0, top)),
        "answer": f"Baseline retrieved: {short}",
        "metrics": {"top_score": top, "hits": len(hits)},
    }


def _metrics(rows: list[EvalRow]) -> dict[str, Any]:
    answerable = [r for r in rows if r.answerable]
    unanswerable = [r for r in rows if not r.answerable]

    def rate(n: int, d: int) -> float:
        return round((n / d), 4) if d else 0.0

    return {
        "rows": len(rows),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "grounded_answer_rate_on_answerable": rate(
            sum(1 for r in answerable if r.answered and r.expected_hit), len(answerable)
        ),
        "abstain_on_answerable_rate": rate(sum(1 for r in answerable if r.abstained), len(answerable)),
        "abstain_on_unanswerable_rate": rate(
            sum(1 for r in unanswerable if r.abstained), len(unanswerable)
        ),
        "hallucination_rate_on_unanswerable": rate(
            sum(1 for r in unanswerable if r.answered), len(unanswerable)
        ),
        "mean_confidence": round(sum(r.confidence for r in rows) / max(1, len(rows)), 4),
    }


def _render_report(summary: dict[str, Any]) -> str:
    a = summary["baseline"]
    b = summary["plugin"]
    d = summary["delta_plugin_minus_baseline"]
    lines = [
        "# OpenClaw Agentic RAG A/B",
        "",
        f"- generated_utc: `{summary['generated_utc']}`",
        f"- suite: `{summary['suite']}`",
        f"- corpus: `{summary['corpus']}`",
        "",
        "## Baseline (A)",
        "",
        f"- grounded_answer_rate_on_answerable: `{a['grounded_answer_rate_on_answerable']}`",
        f"- abstain_on_answerable_rate: `{a['abstain_on_answerable_rate']}`",
        f"- abstain_on_unanswerable_rate: `{a['abstain_on_unanswerable_rate']}`",
        f"- hallucination_rate_on_unanswerable: `{a['hallucination_rate_on_unanswerable']}`",
        "",
        "## Plugin (B)",
        "",
        f"- grounded_answer_rate_on_answerable: `{b['grounded_answer_rate_on_answerable']}`",
        f"- abstain_on_answerable_rate: `{b['abstain_on_answerable_rate']}`",
        f"- abstain_on_unanswerable_rate: `{b['abstain_on_unanswerable_rate']}`",
        f"- hallucination_rate_on_unanswerable: `{b['hallucination_rate_on_unanswerable']}`",
        "",
        "## Delta (B - A)",
        "",
        f"- grounded_answer_rate_on_answerable: `{d['grounded_answer_rate_on_answerable']}`",
        f"- abstain_on_answerable_rate: `{d['abstain_on_answerable_rate']}`",
        f"- abstain_on_unanswerable_rate: `{d['abstain_on_unanswerable_rate']}`",
        f"- hallucination_rate_on_unanswerable: `{d['hallucination_rate_on_unanswerable']}`",
        "",
    ]
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A/B for Agentic RAG plugin.")
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--min-retrieval-score", type=float, default=0.18)
    parser.add_argument("--min-confidence", type=float, default=0.45)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = (args.out_dir or (DEFAULT_OUT_ROOT / f"run_{stamp}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    suite_raw = json.loads(args.suite.read_text(encoding="utf-8"))
    items = suite_raw.get("items", [])
    plugin = AgenticRagPlugin.from_json_path(
        args.corpus,
        config=AgenticRagConfig(
            min_retrieval_score=float(args.min_retrieval_score),
            min_confidence=float(args.min_confidence),
        ),
    )

    rows_a: list[EvalRow] = []
    rows_b: list[EvalRow] = []
    all_rows: list[dict[str, Any]] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id", ""))
        query = str(item.get("query", ""))
        answerable = bool(item.get("answerable", False))
        expected = [str(k) for k in (item.get("expected_keywords") or [])]

        baseline = _run_baseline(plugin, query)
        expected_hit_a = _contains_any(str(baseline.get("answer", "")), expected)
        row_a = EvalRow(
            item_id=item_id,
            answerable=answerable,
            query=query,
            mode_name="baseline",
            outcome_mode=str(baseline.get("mode", "")),
            confidence=float(baseline.get("confidence", 0.0)),
            top_score=float((baseline.get("metrics") or {}).get("top_score", 0.0)),
            answered=str(baseline.get("mode", "")) == "answer",
            abstained=str(baseline.get("mode", "")) == "abstain",
            expected_hit=expected_hit_a,
        )
        rows_a.append(row_a)

        decision = plugin.decide(query)
        expected_hit_b = _contains_any(decision.answer, expected)
        row_b = EvalRow(
            item_id=item_id,
            answerable=answerable,
            query=query,
            mode_name="plugin",
            outcome_mode=decision.mode,
            confidence=decision.confidence,
            top_score=float(decision.metrics.get("top_score", 0.0)),
            answered=decision.mode == "answer",
            abstained=decision.mode == "abstain",
            expected_hit=expected_hit_b,
        )
        rows_b.append(row_b)

        all_rows.append(
            {
                "id": item_id,
                "answerable": answerable,
                "query": query,
                "baseline_mode": row_a.outcome_mode,
                "baseline_confidence": row_a.confidence,
                "baseline_top_score": row_a.top_score,
                "baseline_expected_hit": row_a.expected_hit,
                "plugin_mode": row_b.outcome_mode,
                "plugin_confidence": row_b.confidence,
                "plugin_top_score": row_b.top_score,
                "plugin_expected_hit": row_b.expected_hit,
            }
        )

    a = _metrics(rows_a)
    b = _metrics(rows_b)
    delta = {
        key: round(float(b.get(key, 0.0)) - float(a.get(key, 0.0)), 4)
        for key in (
            "grounded_answer_rate_on_answerable",
            "abstain_on_answerable_rate",
            "abstain_on_unanswerable_rate",
            "hallucination_rate_on_unanswerable",
        )
    }

    summary = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "suite": args.suite.as_posix(),
        "corpus": args.corpus.as_posix(),
        "plugin_config": {
            "min_retrieval_score": float(args.min_retrieval_score),
            "min_confidence": float(args.min_confidence),
        },
        "rows": len(all_rows),
        "baseline": a,
        "plugin": b,
        "delta_plugin_minus_baseline": delta,
    }

    with (out_dir / "per_case.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()) if all_rows else [])
        if all_rows:
            writer.writeheader()
            writer.writerows(all_rows)

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(_render_report(summary), encoding="utf-8")

    print(f"Run dir: {out_dir.as_posix()}")
    print(f"Rows: {summary['rows']}")
    print(
        "Delta abstain_on_unanswerable_rate:",
        summary["delta_plugin_minus_baseline"]["abstain_on_unanswerable_rate"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
