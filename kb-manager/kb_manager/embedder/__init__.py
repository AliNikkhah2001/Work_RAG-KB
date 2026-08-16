"""Text embedding interfaces and implementations."""

from kb_manager.embedder.base import BaseEmbedder
from kb_manager.embedder.registry import get_embedder
from kb_manager.embedder.sentence_transformer import SentenceTransformerEmbedder

__all__ = [
    "BaseEmbedder",
    "SentenceTransformerEmbedder",
    "get_embedder",
]
