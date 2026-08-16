"""Version management routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select

from kb_manager.models.database import Document, DocumentVersion
from kb_manager.web.app import db, templates

router = APIRouter()


@router.get("")
async def list_versions(request: Request):
    """Show all versioned documents."""
    async with db.session() as session:
        result = await session.execute(
            select(Document).where(Document.version > 1).order_by(Document.updated_at.desc())
        )
        documents = result.scalars().all()

    return templates.TemplateResponse(
        request,
        "versions.html",
        {"documents": documents},
    )


@router.get("/{doc_id}")
async def version_history(request: Request, doc_id: str):
    """Show version history for a document."""
    async with db.session() as session:
        document = await session.get(Document, doc_id)
        if not document:
            return RedirectResponse("/versions", status_code=302)

        result = await session.execute(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == doc_id)
            .order_by(DocumentVersion.version.desc())
        )
        versions = result.scalars().all()

    return templates.TemplateResponse(
        request,
        "version_history.html",
        {"document": document, "versions": versions},
    )


@router.post("/{doc_id}/rollback/{version_id}")
async def rollback_version(doc_id: str, version_id: str):
    """Rollback to a specific version."""
    async with db.session() as session:
        version = await session.get(DocumentVersion, version_id)
        document = await session.get(Document, doc_id)

        if version and document:
            # Restore document from version snapshot
            if version.snapshot_data:
                document.title = version.snapshot_data.get("title", document.title)
                document.domain = version.snapshot_data.get("domain", document.domain)
                document.category = version.snapshot_data.get("category", document.category)

            document.content_hash = version.content_hash
            document.version += 1

    return RedirectResponse(f"/versions/{doc_id}", status_code=302)
