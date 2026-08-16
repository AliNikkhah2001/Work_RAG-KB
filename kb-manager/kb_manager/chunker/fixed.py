from __future__ import annotations

import re
from typing import Any

from .base import BaseChunker, Chunk, _estimate_tokens

# ---------------------------------------------------------------------------
# Sentence tokenizer with Hazm fallback
# ---------------------------------------------------------------------------

_SENT_END_RE = re.compile(r"(?<=[.!؟!?])\s+")
_MIN_SENTENCE_CHARS = 10

try:
    from hazm import sent_tokenize as _hazm_sent_tokenize

    def _split_sentences(text: str) -> list[str]:
        try:
            return [s for s in _hazm_sent_tokenize(text) if s.strip()]
        except Exception:  # noqa: BLE001
            return _fallback_split_sentences(text)

except ImportError:

    def _split_sentences(text: str) -> list[str]:  # type: ignore[misc]
        return _fallback_split_sentences(text)


def _fallback_split_sentences(text: str) -> list[str]:
    """Regex-based sentence splitter as a last resort."""
    raw = _SENT_END_RE.split(text)
    return [s.strip() for s in raw if s.strip()]


class FixedChunker(BaseChunker):
    """Fixed-size chunker with sentence-boundary awareness.

    The chunker accumulates sentences until adding the next sentence would
    exceed ``max_tokens``.  If the accumulated text is shorter than
    ``min_tokens`` it is merged forward into the next chunk.  Each chunk
    shares an ``overlap_tokens`` token tail/head with its neighbours.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        min_tokens: int = 64,
        overlap_tokens: int = 50,
    ) -> None:
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Chunk *text* into fixed-size, sentence-aligned segments.

        Args:
            text: Raw document text.
            metadata: Optional metadata forwarded to every chunk.

        Returns:
            Ordered list of :class:`Chunk` instances.
        """
        meta = metadata or {}
        sentences = _split_sentences(text)
        raw_chunks = self._accumulate(sentences)
        merged = self._merge_short(raw_chunks)
        return self._build_chunks(merged, meta)

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def _accumulate(self, sentences: list[str]) -> list[str]:
        """Group sentences so that each group fits within max_tokens."""
        if not sentences:
            return []

        chunks: list[str] = []
        current_sentences: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = _estimate_tokens(sentence)

            # A single sentence already exceeds the limit – emit it alone
            if sent_tokens >= self.max_tokens and not current_sentences:
                chunks.append(sentence)
                continue

            # Would adding this sentence push us over the limit?
            if current_sentences and current_tokens + sent_tokens > self.max_tokens:
                chunks.append(" ".join(current_sentences))
                current_sentences = [sentence]
                current_tokens = sent_tokens
            else:
                current_sentences.append(sentence)
                current_tokens += sent_tokens

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks

    def _merge_short(self, chunks: list[str]) -> list[str]:
        """Merge chunks shorter than min_tokens with the next chunk."""
        if not chunks:
            return []

        merged: list[str] = []
        buffer = chunks[0]

        for chunk in chunks[1:]:
            if _estimate_tokens(buffer) < self.min_tokens:
                buffer = f"{buffer}\n\n{chunk}"
            else:
                merged.append(buffer)
                buffer = chunk

        merged.append(buffer)
        return merged

    # ------------------------------------------------------------------
    # Build output
    # ------------------------------------------------------------------

    def _build_chunks(self, texts: list[str], metadata: dict[str, Any]) -> list[Chunk]:
        """Wrap raw text fragments into :class:`Chunk` objects."""
        chunks: list[Chunk] = []
        for idx, content in enumerate(texts):
            content = content.strip()
            if not content:
                continue
            chunks.append(
                Chunk(
                    content=content,
                    ordinal=idx,
                    chunk_type="body",
                    metadata=dict(metadata),
                )
            )
        return self._apply_overlap(chunks)

    def _apply_overlap(self, chunks: list[Chunk]) -> list[Chunk]:
        """Prepend tail of previous chunk / append head of next chunk."""
        if self.overlap_tokens <= 0 or len(chunks) < 2:
            return chunks

        enriched: list[Chunk] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_words = chunks[i - 1].content.split()
            overlap_word_count = max(1, self.overlap_tokens // 2)
            overlap_text = " ".join(prev_words[-overlap_word_count:])
            new_content = f"…{overlap_text}…\n{chunks[i].content}"
            enriched.append(
                Chunk(
                    content=new_content,
                    ordinal=chunks[i].ordinal,
                    chunk_type=chunks[i].chunk_type,
                    heading_path=chunks[i].heading_path,
                    keywords=list(chunks[i].keywords),
                    metadata=dict(chunks[i].metadata),
                )
            )
        return enriched
