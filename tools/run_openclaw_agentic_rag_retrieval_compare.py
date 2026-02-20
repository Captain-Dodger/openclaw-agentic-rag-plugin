#!/usr/bin/env python3
"""Compare lexical vs hybrid retrieval modes for Agentic RAG plugin."""

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
    retrieval_mode_effective: str
    embedding_ready: bool


def _contains_any(text: str, needles: list[str]) -> bool:
    if not needles:
        return False
    blob = (text or "").casefold()
    return any((needle or "").casefold() in blob for needle in needles)


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
        "embedding_ready_rate": rate(sum(1 for r in rows if r.embedding_ready), len(rows)),
    }


def _render_report(summary: dict[str, Any]) -> str:
    l = summary["lexical"]
    h = summary["hybrid"]
    d = summary["delta_hybrid_minus_lexical"]
    lines = [
        "# Agentic RAG Retrieval Compare",
        "",
        f"- generated_utc: `{summary['generated_utc']}`",
        f"- suite: `{summary['suite']}`",
        f"- corpus: `{summary['corpus']}`",
        "",
        "## Lexical",
        "",
        f"- grounded_answer_rate_on_answerable: `{l['grounded_answer_rate_on_answerable']}`",
        f"- abstain_on_answerable_rate: `{l['abstain_on_answerable_rate']}`",
        f"- abstain_on_unanswerable_rate: `{l['abstain_on_unanswerable_rate']}`",
        f"- hallucination_rate_on_unanswerable: `{l['hallucination_rate_on_unanswerable']}`",
        "",
        "## Hybrid",
        "",
        f"- grounded_answer_rate_on_answerable: `{h['grounded_answer_rate_on_answerable']}`",
        f"- abstain_on_answerable_rate: `{h['abstain_on_answerable_rate']}`",
        f"- abstain_on_unanswerable_rate: `{h['abstain_on_unanswerable_rate']}`",
        f"- hallucination_rate_on_unanswerable: `{h['hallucination_rate_on_unanswerable']}`",
        "",
        "## Delta (hybrid - lexical)",
        "",
        f"- grounded_answer_rate_on_answerable: `{d['grounded_answer_rate_on_answerable']}`",
        f"- abstain_on_answerable_rate: `{d['abstain_on_answerable_rate']}`",
        f"- abstain_on_unanswerable_rate: `{d['abstain_on_unanswerable_rate']}`",
        f"- hallucination_rate_on_unanswerable: `{d['hallucination_rate_on_unanswerable']}`",
        "",
    ]
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare lexical and hybrid retrieval modes.")
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--min-retrieval-score", type=float, default=0.18)
    parser.add_argument("--min-confidence", type=float, default=0.45)
    parser.add_argument("--embedding-base-url", type=str, default="http://127.0.0.1:1234/v1")
    parser.add_argument("--embedding-model", type=str, default="text-embedding-nomic-embed-text-v1.5")
    parser.add_argument("--embedding-timeout-ms", type=int, default=10000)
    parser.add_argument("--hybrid-lexical-weight", type=float, default=0.35)
    parser.add_argument("--hybrid-min-lexical-score", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def _evaluate_mode(
    plugin: AgenticRagPlugin, items: list[dict[str, Any]], mode_name: str
) -> tuple[list[EvalRow], list[dict[str, Any]]]:
    rows: list[EvalRow] = []
    per_case: list[dict[str, Any]] = []
    for item in items:
        item_id = str(item.get("id", ""))
        query = str(item.get("query", ""))
        answerable = bool(item.get("answerable", False))
        expected = [str(k) for k in (item.get("expected_keywords") or [])]

        decision = plugin.decide(query)
        expected_hit = _contains_any(decision.answer, expected)
        row = EvalRow(
            item_id=item_id,
            answerable=answerable,
            query=query,
            mode_name=mode_name,
            outcome_mode=decision.mode,
            confidence=decision.confidence,
            top_score=float(decision.metrics.get("top_score", 0.0)),
            answered=decision.mode == "answer",
            abstained=decision.mode == "abstain",
            expected_hit=expected_hit,
            retrieval_mode_effective=str(decision.metrics.get("retrieval_mode_effective", "")),
            embedding_ready=bool(decision.metrics.get("embedding_ready", False)),
        )
        rows.append(row)
        per_case.append(
            {
                "id": item_id,
                "answerable": answerable,
                "query": query,
                f"{mode_name}_mode": row.outcome_mode,
                f"{mode_name}_confidence": row.confidence,
                f"{mode_name}_top_score": row.top_score,
                f"{mode_name}_expected_hit": row.expected_hit,
                f"{mode_name}_retrieval_mode_effective": row.retrieval_mode_effective,
                f"{mode_name}_embedding_ready": row.embedding_ready,
            }
        )
    return rows, per_case


def main() -> int:
    args = _parse_args()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = (args.out_dir or (DEFAULT_OUT_ROOT / f"compare_{stamp}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    suite_raw = json.loads(args.suite.read_text(encoding="utf-8"))
    items = [x for x in suite_raw.get("items", []) if isinstance(x, dict)]

    lexical = AgenticRagPlugin.from_path(
        args.corpus,
        config=AgenticRagConfig(
            min_retrieval_score=float(args.min_retrieval_score),
            min_confidence=float(args.min_confidence),
            retrieval_mode="lexical",
            embedding_enabled=False,
        ),
    )
    hybrid = AgenticRagPlugin.from_path(
        args.corpus,
        config=AgenticRagConfig(
            min_retrieval_score=float(args.min_retrieval_score),
            min_confidence=float(args.min_confidence),
            retrieval_mode="hybrid",
            embedding_enabled=True,
            embedding_base_url=str(args.embedding_base_url),
            embedding_model=str(args.embedding_model),
            embedding_timeout_ms=int(args.embedding_timeout_ms),
            hybrid_lexical_weight=float(args.hybrid_lexical_weight),
            hybrid_min_lexical_score=float(args.hybrid_min_lexical_score),
        ),
    )

    rows_lex, per_case_lex = _evaluate_mode(lexical, items, "lexical")
    rows_hyb, per_case_hyb = _evaluate_mode(hybrid, items, "hybrid")

    merged_rows: list[dict[str, Any]] = []
    for i in range(min(len(per_case_lex), len(per_case_hyb))):
        merged = dict(per_case_lex[i])
        merged.update(per_case_hyb[i])
        merged_rows.append(merged)

    lex_metrics = _metrics(rows_lex)
    hyb_metrics = _metrics(rows_hyb)
    delta = {
        key: round(float(hyb_metrics.get(key, 0.0)) - float(lex_metrics.get(key, 0.0)), 4)
        for key in (
            "grounded_answer_rate_on_answerable",
            "abstain_on_answerable_rate",
            "abstain_on_unanswerable_rate",
            "hallucination_rate_on_unanswerable",
            "mean_confidence",
        )
    }

    summary = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "suite": args.suite.as_posix(),
        "corpus": args.corpus.as_posix(),
        "compare_config": {
            "min_retrieval_score": float(args.min_retrieval_score),
            "min_confidence": float(args.min_confidence),
            "embedding_base_url": str(args.embedding_base_url),
            "embedding_model": str(args.embedding_model),
            "embedding_timeout_ms": int(args.embedding_timeout_ms),
            "hybrid_lexical_weight": float(args.hybrid_lexical_weight),
            "hybrid_min_lexical_score": float(args.hybrid_min_lexical_score),
        },
        "rows": len(merged_rows),
        "lexical": lex_metrics,
        "hybrid": hyb_metrics,
        "delta_hybrid_minus_lexical": delta,
    }

    with (out_dir / "per_case.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(merged_rows[0].keys()) if merged_rows else [])
        if merged_rows:
            writer.writeheader()
            writer.writerows(merged_rows)

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(_render_report(summary), encoding="utf-8")

    print(f"Run dir: {out_dir.as_posix()}")
    print(f"Rows: {summary['rows']}")
    print("Delta grounded_answer_rate_on_answerable:", summary["delta_hybrid_minus_lexical"]["grounded_answer_rate_on_answerable"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
