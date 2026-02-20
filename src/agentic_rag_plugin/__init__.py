"""Agentic RAG plugin scaffold for OpenClaw-style tool pipelines."""

from .corpus_loader import load_documents_from_source, write_documents_json
from .config import AgenticRagConfig
from .plugin import AgenticRagPlugin, handle_tool_call
from .types import Document, RagDecision, RetrievalHit

__all__ = [
    "AgenticRagConfig",
    "AgenticRagPlugin",
    "Document",
    "RagDecision",
    "RetrievalHit",
    "load_documents_from_source",
    "write_documents_json",
    "handle_tool_call",
]
