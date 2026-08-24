"""Core logic for cleaning up incomplete QA chunks."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import select

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

from kb_manager.models.database import Chunk, Document
from kb_manager.pipeline.versioning import VersionManager

logger = logging.getLogger(__name__)


@dataclass
class IncompleteChunkInfo:
    """Information about an incomplete QA chunk."""
    chunk_id: str
    document_id: str
    document_title: str
    ordinal: int
    missing_fields: list[str]
    fields: dict


@dataclass
class DocumentCleanupPreview:
    """Preview of cleanup for a single document."""
    document_id: str
    document_title: str
    total_qa_chunks: int
    incomplete_chunks: list[IncompleteChunkInfo]
    complete_chunks: int

    @property
    def incomplete_count(self) -> int:
        return len(self.incomplete_chunks)

    @property
    def has_incomplete(self) -> bool:
        return self.incomplete_count > 0


@dataclass
class CleanupResult:
    """Result of a cleanup operation."""
    document_id: str
    document_title: str
    deleted_count: int
    kept_count: int
    version_created: bool = False
    error: str | None = None


class QACleanup:
    """Handles detection and cleanup of incomplete QA chunks."""

    def __init__(self, db) -> None:
        self._db = db
        self._version_mgr = VersionManager()

    def _is_qa_chunk_incomplete(self, chunk: Chunk) -> tuple[bool, list[str]]:
        """Check if a qa_pair chunk is incomplete.

        Returns:
            (is_incomplete, list_of_missing_field_names)
        """
        if chunk.chunk_type != "qa_pair":
            return False, []

        fields = chunk.doc_metadata.get("fields", {}) if chunk.doc_metadata else {}

        missing = []

        # Check for question
        if not fields.get("question"):
            missing.append("question")

        # Check for answer (answer or briefanswer)
        has_answer = bool(fields.get("answer") or fields.get("briefanswer"))
        if not has_answer:
            missing.append("answer/briefanswer")

        return len(missing) > 0, missing

    async def find_incomplete_chunks(
        self,
        session: AsyncSession,
        document_id: str | None = None,
    ) -> list[IncompleteChunkInfo]:
        """Find all incomplete QA chunks, optionally filtered by document."""
        query = select(Chunk, Document).join(
            Document, Chunk.document_id == Document.id
        ).where(Chunk.chunk_type == "qa_pair")

        if document_id:
            query = query.where(Chunk.document_id == document_id)

        result = await session.execute(query)
        rows = result.all()

        incomplete = []
        for chunk, document in rows:
            is_incomplete, missing = self._is_qa_chunk_incomplete(chunk)
            if is_incomplete:
                incomplete.append(IncompleteChunkInfo(
                    chunk_id=chunk.id,
                    document_id=document.id,
                    document_title=document.title,
                    ordinal=chunk.ordinal,
                    missing_fields=missing,
                    fields=chunk.doc_metadata.get("fields", {}) if chunk.doc_metadata else {},
                ))

        return incomplete

    async def get_document_preview(
        self,
        session: AsyncSession,
        document_id: str,
    ) -> DocumentCleanupPreview | None:
        """Get a preview of cleanup for a single document."""
        # Get document
        document = await session.get(Document, document_id)
        if not document:
            return None

        # Get all qa_pair chunks for this document
        result = await session.execute(
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .where(Chunk.chunk_type == "qa_pair")
            .order_by(Chunk.ordinal)
        )
        chunks = result.scalars().all()

        total_qa = len(chunks)
        incomplete_chunks = []
        complete_count = 0

        for chunk in chunks:
            is_incomplete, missing = self._is_qa_chunk_incomplete(chunk)
            if is_incomplete:
                incomplete_chunks.append(IncompleteChunkInfo(
                    chunk_id=chunk.id,
                    document_id=document.id,
                    document_title=document.title,
                    ordinal=chunk.ordinal,
                    missing_fields=missing,
                    fields=chunk.doc_metadata.get("fields", {}) if chunk.doc_metadata else {},
                ))
            else:
                complete_count += 1

        return DocumentCleanupPreview(
            document_id=document.id,
            document_title=document.title,
            total_qa_chunks=total_qa,
            incomplete_chunks=incomplete_chunks,
            complete_chunks=complete_count,
        )

    async def get_all_documents_preview(
        self,
        session: AsyncSession,
    ) -> list[DocumentCleanupPreview]:
        """Get preview for all documents that have incomplete QA chunks."""
        # Find all documents with incomplete QA chunks
        incomplete_infos = await self.find_incomplete_chunks(session)

        # Group by document
        doc_map: dict[str, DocumentCleanupPreview] = {}
        for info in incomplete_infos:
            if info.document_id not in doc_map:
                # Get complete count for this document
                result = await session.execute(
                    select(Chunk)
                    .where(Chunk.document_id == info.document_id)
                    .where(Chunk.chunk_type == "qa_pair")
                )
                all_chunks = result.scalars().all()
                complete = sum(
                    1 for c in all_chunks
                    if not self._is_qa_chunk_incomplete(c)[0]
                )
                doc_map[info.document_id] = DocumentCleanupPreview(
                    document_id=info.document_id,
                    document_title=info.document_title,
                    total_qa_chunks=len(all_chunks),
                    incomplete_chunks=[],
                    complete_chunks=complete,
                )
            doc_map[info.document_id].incomplete_chunks.append(info)

        return list(doc_map.values())

    async def cleanup_document(
        self,
        session: AsyncSession,
        document_id: str,
    ) -> CleanupResult:
        """Delete incomplete QA chunks for a document and create a version snapshot."""
        preview = await self.get_document_preview(session, document_id)
        if not preview:
            return CleanupResult(
                document_id=document_id,
                document_title="Unknown",
                deleted_count=0,
                kept_count=0,
                error="Document not found",
            )

        if not preview.has_incomplete:
            return CleanupResult(
                document_id=document_id,
                document_title=preview.document_title,
                deleted_count=0,
                kept_count=preview.complete_chunks,
            )

        # Create version snapshot BEFORE deleting chunks
        try:
            await self._version_mgr.create_snapshot(
                document_id,
                session,
                change_summary="Cleanup: removed incomplete QA chunks",
                changed_by="cleanup_incomplete_qa",
            )
            version_created = True
        except Exception as e:
            logger.exception("Failed to create version snapshot for %s", document_id)
            return CleanupResult(
                document_id=document_id,
                document_title=preview.document_title,
                deleted_count=0,
                kept_count=preview.complete_chunks,
                error=f"Failed to create version snapshot: {e}",
            )

        # Delete incomplete chunks
        deleted_count = 0
        for info in preview.incomplete_chunks:
            chunk = await session.get(Chunk, info.chunk_id)
            if chunk:
                await session.delete(chunk)
                deleted_count += 1

        await session.flush()

        # Update document chunk_count
        document = await session.get(Document, document_id)
        if document:
            remaining = await session.execute(
                select(Chunk).where(Chunk.document_id == document_id)
            )
            document.chunk_count = len(remaining.scalars().all())

        return CleanupResult(
            document_id=document_id,
            document_title=preview.document_title,
            deleted_count=deleted_count,
            kept_count=preview.complete_chunks,
            version_created=True,
        )

    async def cleanup_all_documents(
        self,
        session: AsyncSession,
    ) -> list[CleanupResult]:
        """Clean up incomplete QA chunks for all documents."""
        previews = await self.get_all_documents_preview(session)
        results = []

        for preview in previews:
            if preview.has_incomplete:
                result = await self.cleanup_document(session, preview.document_id)
                results.append(result)

        return results