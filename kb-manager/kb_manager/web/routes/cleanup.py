"""Cleanup routes for incomplete QA chunks."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import func, select

from kb_manager.cleanup.qa_cleanup import QACleanup
from kb_manager.models.database import Chunk, Document
from kb_manager.web.deps import db, templates

router = APIRouter(tags=["cleanup"])


@router.get("/qa", response_class=HTMLResponse)
async def cleanup_qa_dashboard(request: Request):
    """Show dashboard of documents with incomplete QA chunks."""
    cleanup = QACleanup(db)

    async with db.session() as session:
        previews = await cleanup.get_all_documents_preview(session)

        # Get total stats from previews
        total_docs = len(previews)
        total_incomplete = sum(p.incomplete_count for p in previews)
        # total_qa should be total QA in DB, not just from previews
        total_qa_result = await session.execute(select(func.count(Chunk.id)).where(Chunk.chunk_type == "qa_pair"))
        total_qa = total_qa_result.scalar() or 0
        # also get complete count
        complete_qa = total_qa - total_incomplete

    return templates.TemplateResponse(
        request,
        "cleanup_qa.html",
        {
            "previews": previews,
            "total_docs": total_docs,
            "total_incomplete": total_incomplete,
            "total_qa": total_qa,
            "complete_qa": complete_qa,
        },
    )


@router.get("/qa/{doc_id}/preview", response_class=HTMLResponse)
async def cleanup_qa_preview(request: Request, doc_id: str):
    """Show detailed preview of incomplete chunks for a document."""
    cleanup = QACleanup(db)

    async with db.session() as session:
        preview = await cleanup.get_document_preview(session, doc_id)

        if not preview:
            return RedirectResponse("/cleanup/qa", status_code=302)

        # Also get version history for context
        from kb_manager.pipeline.versioning import VersionManager
        version_mgr = VersionManager()
        versions = await version_mgr.get_version_history(doc_id, session)

    return templates.TemplateResponse(
        request,
        "cleanup_qa_preview.html",
        {
            "preview": preview,
            "versions": versions,
        },
    )


@router.post("/qa/{doc_id}/execute")
async def cleanup_qa_execute(doc_id: str):
    """Execute cleanup for a single document."""
    cleanup = QACleanup(db)

    async with db.session() as session:
        result = await cleanup.cleanup_document(session, doc_id)

    # Redirect back to preview with message
    return RedirectResponse(
        f"/cleanup/qa/{doc_id}/preview?cleaned={result.deleted_count}&kept={result.kept_count}",
        status_code=302,
    )


@router.post("/qa/execute-all")
async def cleanup_qa_execute_all():
    """Execute cleanup for all documents with incomplete chunks."""
    cleanup = QACleanup(db)

    async with db.session() as session:
        results = await cleanup.cleanup_all_documents(session)

    total_deleted = sum(r.deleted_count for r in results)
    total_kept = sum(r.kept_count for r in results)

    return RedirectResponse(
        f"/cleanup/qa?cleaned_all={total_deleted}&kept_all={total_kept}",
        status_code=302,
    )