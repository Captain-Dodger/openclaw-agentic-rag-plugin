from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from .types import Document


SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".log", ".pdf", ".odt"}


def _normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split())


def _read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_pdf_text(path: Path) -> str:
    try:
        import pypdf  # type: ignore[import-not-found]
    except Exception:
        return ""
    try:
        reader = pypdf.PdfReader(str(path))
        parts: list[str] = []
        for page in reader.pages:
            raw = page.extract_text() or ""
            if raw.strip():
                parts.append(raw)
        return "\n".join(parts)
    except Exception:
        return ""


def _extract_odt_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            raw = zf.read("content.xml")
    except Exception:
        return ""
    try:
        root = ET.fromstring(raw)
    except Exception:
        return ""
    parts = [x.strip() for x in root.itertext() if x and x.strip()]
    return " ".join(parts)


def _load_text_from_file(path: Path) -> str:
    ext = path.suffix.casefold()
    if ext in {".txt", ".md", ".markdown", ".rst", ".log"}:
        return _read_text_file(path)
    if ext == ".pdf":
        return _extract_pdf_text(path)
    if ext == ".odt":
        return _extract_odt_text(path)
    return ""


def _chunk_text(text: str, *, chunk_chars: int, overlap_chars: int) -> list[str]:
    blob = _normalize_whitespace(text)
    if not blob:
        return []
    chunk_chars = max(200, int(chunk_chars))
    overlap_chars = max(0, int(overlap_chars))
    if overlap_chars >= chunk_chars:
        overlap_chars = chunk_chars // 4
    if len(blob) <= chunk_chars:
        return [blob]
    out: list[str] = []
    start = 0
    step = chunk_chars - overlap_chars
    while start < len(blob):
        end = min(len(blob), start + chunk_chars)
        chunk = blob[start:end].strip()
        if chunk:
            out.append(chunk)
        if end >= len(blob):
            break
        start += step
    return out


def _slug_path(path: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", path.casefold()).strip("_")
    return value or "doc"


def _load_documents_from_json(path: Path) -> list[Document]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    docs: list[Document] = []
    if not isinstance(raw, list):
        return docs
    for item in raw:
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("id", "")).strip()
        text = str(item.get("text", "")).strip()
        source = str(item.get("source", "unknown")).strip() or "unknown"
        if not doc_id or not text:
            continue
        docs.append(Document(id=doc_id, text=text, source=source))
    return docs


def _load_documents_from_dir(
    root: Path,
    *,
    chunk_chars: int,
    overlap_chars: int,
    supported_extensions: set[str] | None = None,
) -> list[Document]:
    exts = supported_extensions or SUPPORTED_EXTENSIONS
    docs: list[Document] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.suffix.casefold() not in exts:
            continue
        raw = _load_text_from_file(path)
        chunks = _chunk_text(raw, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
        if not chunks:
            continue
        rel = path.relative_to(root).as_posix()
        slug = _slug_path(rel)
        for idx, chunk in enumerate(chunks, start=1):
            docs.append(
                Document(
                    id=f"{slug}__c{idx:03d}",
                    text=chunk,
                    source=rel,
                )
            )
    return docs


def load_documents_from_source(
    source_path: str | Path,
    *,
    chunk_chars: int = 1200,
    overlap_chars: int = 120,
    supported_extensions: set[str] | None = None,
) -> list[Document]:
    path = Path(source_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"corpus source not found: {path}")
    if path.is_file():
        if path.suffix.casefold() == ".json":
            docs = _load_documents_from_json(path)
        else:
            docs = _load_documents_from_dir(
                path.parent,
                chunk_chars=chunk_chars,
                overlap_chars=overlap_chars,
                supported_extensions={path.suffix.casefold()},
            )
            docs = [d for d in docs if d.source == path.name]
    else:
        docs = _load_documents_from_dir(
            path,
            chunk_chars=chunk_chars,
            overlap_chars=overlap_chars,
            supported_extensions=supported_extensions,
        )
    if not docs:
        raise ValueError(
            "no usable documents found in corpus source "
            f"{path} (supported: {sorted(supported_extensions or SUPPORTED_EXTENSIONS)})"
        )
    return docs


def write_documents_json(documents: list[Document], output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"id": d.id, "source": d.source, "text": d.text} for d in documents]
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
