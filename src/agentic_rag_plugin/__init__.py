"""Agentic RAG plugin scaffold for OpenClaw-style tool pipelines."""

from .arbiter import run_arbiter_v1
from .corpus_loader import load_documents_from_source, write_documents_json
from .config import AgenticRagConfig
from .plugin import AgenticRagPlugin, handle_tool_call
from .types import ArbiterPacket, Document, RagDecision, RetrievalHit

__all__ = [
    "AgenticRagConfig",
    "AgenticRagPlugin",
    "ArbiterPacket",
    "Document",
    "RagDecision",
    "RetrievalHit",
    "run_arbiter_v1",
    "load_documents_from_source",
    "write_documents_json",
    "handle_tool_call",
]
