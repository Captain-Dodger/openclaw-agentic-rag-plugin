"""Agentic RAG plugin scaffold for OpenClaw-style tool pipelines."""

from .config import AgenticRagConfig
from .plugin import AgenticRagPlugin, handle_tool_call
from .types import Document, RagDecision, RetrievalHit

__all__ = [
    "AgenticRagConfig",
    "AgenticRagPlugin",
    "Document",
    "RagDecision",
    "RetrievalHit",
    "handle_tool_call",
]

