"""Resolve dense embed + rerank backends (local weights vs OpenRouter API)."""

from __future__ import annotations

import os


def get_embed_backend() -> str:
    """Resolve dense embed backend: ``local`` or ``openrouter``.

    Precedence mirrors reranker:
    1. Explicit ``KB_EMBED_BACKEND``
    2. ``LLM_UPSTREAM_MODE=openrouter`` → openrouter (laptop)
    3. Otherwise ``local``
    """
    explicit = (os.getenv("KB_EMBED_BACKEND") or "").strip().lower()
    if explicit:
        return explicit
    if (os.getenv("LLM_UPSTREAM_MODE") or "").strip().lower() == "openrouter":
        return "openrouter"
    return "local"
