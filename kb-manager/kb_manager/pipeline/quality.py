"""Quality gates and validation for KB content."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import func, select

from kb_manager.models.database import Document

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from kb_manager.chunker.base import Chunk

logger = logging.getLogger(__name__)

_PERSIAN_RE = re.compile(r"[\u0600-\u06ff\u0750-\u077f]")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True)
class QualityThresholds:
    """Configurable quality thresholds for validation."""

    min_chunk_length: int = 20
    max_chunk_length: int = 10_000
    min_persian_ratio: float = 0.1
    min_encoding_quality: float = 0.8
    min_doc_chunk_count: int = 1
    max_doc_chunk_count: int = 500
    stale_days: int = 90


class QualityGate:
    """Validates chunks, documents, and checks for staleness."""

    def __init__(self, thresholds: QualityThresholds | None = None) -> None:
        self._thresholds = thresholds or QualityThresholds()

    @property
    def thresholds(self) -> QualityThresholds:
        return self._thresholds

    # ------------------------------------------------------------------
    # Chunk validation
    # ------------------------------------------------------------------

    def validate_chunk(self, chunk: Chunk) -> list[str]:
        """Check chunk quality and return a list of error messages.

        An empty list means the chunk passed all checks.

        Checks performed:
        - Minimum / maximum content length
        - No control characters or replacement characters
        - Sufficient Persian character ratio
        - Reasonable encoding quality
        """
        errors: list[str] = []
        content = chunk.content

        if not content or not content.strip():
            errors.append("Chunk content is empty")
            return errors

        length = len(content)
        if length < self._thresholds.min_chunk_length:
            errors.append(
                f"Content too short: {length} chars (minimum {self._thresholds.min_chunk_length})"
            )

        if length > self._thresholds.max_chunk_length:
            errors.append(
                f"Content too long: {length} chars (maximum {self._thresholds.max_chunk_length})"
            )

        encoding_q = self._encoding_quality(content)
        if encoding_q < self._thresholds.min_encoding_quality:
            errors.append(
                f"Low encoding quality: {encoding_q:.2%} "
                f"(minimum {self._thresholds.min_encoding_quality:.0%})"
            )

        persian_ratio = self._persian_ratio(content)
        if persian_ratio < self._thresholds.min_persian_ratio:
            errors.append(
                f"Low Persian ratio: {persian_ratio:.2%} "
                f"(minimum {self._thresholds.min_persian_ratio:.0%})"
            )

        return errors

    # ------------------------------------------------------------------
    # Document validation
    # ------------------------------------------------------------------

    def validate_document(
        self,
        document: Document,
        chunks: list[Chunk],
    ) -> dict[str, object]:
        """Validate a document and its chunks before indexing.

        Returns a dict with keys:
        - ``valid`` (bool): overall pass / fail
        - ``errors`` (list[str]): blocking errors
        - ``warnings`` (list[str]): non-blocking warnings
        - ``chunk_errors`` (dict[int, list[str]]): per-chunk errors (ordinal → msgs)
        """
        errors: list[str] = []
        warnings: list[str] = []
        chunk_errors: dict[int, list[str]] = {}

        if not document.title or not document.title.strip():
            errors.append("Document title is empty")

        if not document.source_path:
            errors.append("Document source_path is empty")

        if len(chunks) < self._thresholds.min_doc_chunk_count:
            errors.append(
                f"Too few chunks: {len(chunks)} (minimum {self._thresholds.min_doc_chunk_count})"
            )

        if len(chunks) > self._thresholds.max_doc_chunk_count:
            warnings.append(
                f"Large chunk count: {len(chunks)} "
                f"(recommended maximum {self._thresholds.max_doc_chunk_count})"
            )

        for chunk in chunks:
            errs = self.validate_chunk(chunk)
            if errs:
                chunk_errors[chunk.ordinal] = errs

        if chunk_errors:
            failed = len(chunk_errors)
            errors.append(f"{failed}/{len(chunks)} chunks failed validation")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "chunk_errors": chunk_errors,
        }

    # ------------------------------------------------------------------
    # Staleness check
    # ------------------------------------------------------------------

    async def check_staleness(self, session: AsyncSession) -> dict[str, object]:
        """Check for stale documents (not updated within the threshold period).

        Returns a dict with keys:
        - ``total_active`` (int)
        - ``stale_count`` (int)
        - ``stale_percentage`` (float)
        - ``by_category`` (list[dict]): breakdown per category
        """
        cutoff = datetime.now(UTC) - timedelta(days=self._thresholds.stale_days)

        total_q = await session.execute(
            select(func.count(Document.id)).where(Document.status == "active")
        )
        total_active = total_q.scalar_one()

        stale_q = await session.execute(
            select(func.count(Document.id)).where(
                Document.status == "active",
                Document.updated_at < cutoff,
            )
        )
        stale_count = stale_q.scalar_one()

        stale_pct = (stale_count / total_active * 100) if total_active else 0.0

        cat_q = await session.execute(
            select(
                Document.category,
                func.count(Document.id).label("total"),
                func.count(Document.id).where(Document.updated_at < cutoff).label("stale"),
            )
            .where(Document.status == "active")
            .group_by(Document.category)
        )
        by_category = [
            {
                "category": row.category,
                "total": row.total,
                "stale": row.stale,
            }
            for row in cat_q.all()
        ]

        return {
            "total_active": total_active,
            "stale_count": stale_count,
            "stale_percentage": round(stale_pct, 2),
            "by_category": by_category,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encoding_quality(text: str) -> float:
        """Fraction of characters that are well-encoded (0.0 – 1.0)."""
        if not text:
            return 0.0
        bad = 0
        for ch in text:
            cp = ord(ch)
            if cp == 0xFFFD or 0xD800 <= cp <= 0xDFFF or _CONTROL_CHAR_RE.match(ch):
                bad += 1
        return 1.0 - (bad / len(text))

    @staticmethod
    def _persian_ratio(text: str) -> float:
        """Fraction of alphabetic characters that are Persian/Arabic."""
        alpha = [ch for ch in text if ch.isalpha()]
        if not alpha:
            return 0.0
        persian = sum(1 for ch in alpha if _PERSIAN_RE.match(ch))
        return persian / len(alpha)
