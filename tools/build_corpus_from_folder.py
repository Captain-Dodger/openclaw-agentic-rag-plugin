#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_rag_plugin import load_documents_from_source, write_documents_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build corpus JSON from a folder (supports txt/md/pdf/odt)."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input folder or file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "corpus_from_folder.json",
        help="Output corpus json path.",
    )
    parser.add_argument("--chunk-chars", type=int, default=1200)
    parser.add_argument("--overlap-chars", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    docs = load_documents_from_source(
        args.input,
        chunk_chars=args.chunk_chars,
        overlap_chars=args.overlap_chars,
    )
    out = write_documents_json(docs, args.output)
    print(f"input: {args.input.resolve().as_posix()}")
    print(f"output: {out.as_posix()}")
    print(f"documents: {len(docs)}")
    print(f"chunk_chars: {args.chunk_chars}")
    print(f"overlap_chars: {args.overlap_chars}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
