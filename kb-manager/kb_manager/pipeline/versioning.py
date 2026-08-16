"""Version tracking for documents."""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from sqlalchemy import select

from kb_manager.models.database import Chunk, Document, DocumentVersion

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash of *text* for change detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class VersionManager:
    """Manages document version snapshots, history, and rollback."""

    async def create_snapshot(
        self,
        document_id: str,
        session: AsyncSession,
        *,
        change_summary: str | None = None,
        changed_by: str | None = None,
    ) -> DocumentVersion:
        """Save the current state of a document as a version snapshot.

        Args:
            document_id: ID of the document to snapshot.
            session: Active async DB session.
            change_summary: Optional human-readable description of the change.
            changed_by: Optional identifier of the user or process.

        Returns:
            The newly created :class:`DocumentVersion`.

        Raises:
            ValueError: If the document does not exist.
        """
        doc = await session.get(Document, document_id)
        if doc is None:
            raise ValueError(f"Document not found: {document_id}")

        chunks_result = await session.execute(
            select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.ordinal)
        )
        chunks = chunks_result.scalars().all()

        snapshot_data = {
            "title": doc.title,
            "domain": doc.domain,
            "category": doc.category,
            "content_hash": doc.content_hash,
            "chunk_count": len(chunks),
            "chunks": [
                {
                    "ordinal": c.ordinal,
                    "chunk_type": c.chunk_type,
                    "content": c.content,
                    "heading_path": c.heading_path,
                    "keywords": c.keywords,
                    "token_count": c.token_count,
                }
                for c in chunks
            ],
        }

        version = DocumentVersion(
            document_id=document_id,
            version=doc.version,
            content_hash=doc.content_hash,
            change_summary=change_summary,
            changed_by=changed_by,
            chunk_count=len(chunks),
            status="snapshot",
            snapshot_data=snapshot_data,
        )
        session.add(version)
        await session.flush()
        logger.info(
            "Created snapshot v%d for document %s (%d chunks)",
            doc.version,
            document_id,
            len(chunks),
        )
        return version

    async def get_version_history(
        self,
        document_id: str,
        session: AsyncSession,
    ) -> list[DocumentVersion]:
        """Return all versions for a document, newest first.

        Args:
            document_id: ID of the document.
            session: Active async DB session.

        Returns:
            List of :class:`DocumentVersion` ordered by version descending.

        Raises:
            ValueError: If the document does not exist.
        """
        doc = await session.get(Document, document_id)
        if doc is None:
            raise ValueError(f"Document not found: {document_id}")

        result = await session.execute(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .order_by(DocumentVersion.version.desc())
        )
        return list(result.scalars().all())

    async def rollback_to_version(
        self,
        document_id: str,
        version_id: str,
        session: AsyncSession,
    ) -> Document:
        """Restore a document to a previous version.

        This replaces the document's current content and chunks with the
        snapshot data stored in the target version, and creates a new
        version entry recording the rollback.

        Args:
            document_id: ID of the document to restore.
            version_id: ID of the version to restore to.
            session: Active async DB session.

        Returns:
            The updated :class:`Document`.

        Raises:
            ValueError: If the document or version does not exist.
        """
        doc = await session.get(Document, document_id)
        if doc is None:
            raise ValueError(f"Document not found: {document_id}")

        version = await session.get(DocumentVersion, version_id)
        if version is None or version.document_id != document_id:
            raise ValueError(f"Version {version_id} does not belong to document {document_id}")

        snapshot = version.snapshot_data
        if not snapshot:
            raise ValueError(f"Version {version_id} has no snapshot data")

        # Delete existing chunks
        existing_chunks = await session.execute(
            select(Chunk).where(Chunk.document_id == document_id)
        )
        for chunk in existing_chunks.scalars().all():
            await session.delete(chunk)
        await session.flush()

        # Restore chunks from snapshot
        for chunk_data in snapshot.get("chunks", []):
            chunk = Chunk(
                document_id=document_id,
                ordinal=chunk_data["ordinal"],
                chunk_type=chunk_data["chunk_type"],
                content=chunk_data["content"],
                heading_path=chunk_data.get("heading_path", ""),
                keywords=chunk_data.get("keywords", []),
                token_count=chunk_data.get("token_count", 0),
            )
            session.add(chunk)

        # Update document metadata
        doc.version += 1
        doc.content_hash = snapshot.get("content_hash", doc.content_hash)
        doc.chunk_count = len(snapshot.get("chunks", []))

        # Create a rollback version entry
        rollback_version = DocumentVersion(
            document_id=document_id,
            version=doc.version,
            content_hash=doc.content_hash,
            change_summary=f"Rollback to version {version.version} (version_id={version_id})",
            changed_by="system:rollback",
            chunk_count=doc.chunk_count,
            status="rollback",
            snapshot_data=snapshot,
        )
        session.add(rollback_version)
        await session.flush()

        logger.info(
            "Rolled back document %s to version %s (new version: %d)",
            document_id,
            version_id,
            doc.version,
        )
        return doc
