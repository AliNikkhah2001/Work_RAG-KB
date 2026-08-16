"""Chunking strategies for the KB management system."""

from .base import BaseChunker, Chunk
from .fixed import FixedChunker
from .registry import get_chunker, register_chunker
from .semantic import SemanticChunker

Chunker = BaseChunker

__all__ = [
    "BaseChunker",
    "Chunk",
    "Chunker",
    "FixedChunker",
    "SemanticChunker",
    "get_chunker",
    "register_chunker",
]
