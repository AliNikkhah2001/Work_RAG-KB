from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A single chunk of text extracted from a document."""

    content: str
    ordinal: int
    chunk_type: str  # semantic, qa_pair, reason_detail, body, header
    heading_path: str = ""
    keywords: list[str] = field(default_factory=list)
    token_count: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.token_count == 0:
            self.token_count = _estimate_tokens(self.content)


def _estimate_tokens(text: str) -> int:
    """Approximate token count as word count * 2."""
    if not text.strip():
        return 0
    return len(text.split()) * 2


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """Split *text* into a list of chunks.

        Args:
            text: Raw document text.
            metadata: Optional metadata forwarded to every chunk.

        Returns:
            Ordered list of :class:`Chunk` instances.
        """
        ...
