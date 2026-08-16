"""Document management routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import RedirectResponse
from sqlalchemy import func, select

from kb_manager.models.database import Chunk, Document
from kb_manager.web.app import db, templates

router = APIRouter()


@router.get("")
async def list_documents(
    request: Request,
    domain: str = "",
    category: str = "",
    status: str = "",
    page: int = 1,
    per_page: int = 20,
):
    """List all documents with filtering and pagination."""
    async with db.session() as session:
        query = select(Document)
        count_query = select(func.count(Document.id))

        if domain:
            query = query.where(Document.domain == domain)
            count_query = count_query.where(Document.domain == domain)
        if category:
            query = query.where(Document.category == category)
            count_query = count_query.where(Document.category == category)
        if status:
            query = query.where(Document.status == status)
            count_query = count_query.where(Document.status == status)

        total = (await session.execute(count_query)).scalar() or 0
        total_pages = max(1, (total + per_page - 1) // per_page)
        offset = (page - 1) * per_page

        query = query.order_by(Document.updated_at.desc()).offset(offset).limit(per_page)
        result = await session.execute(query)
        documents = result.scalars().all()

        # Get distinct values for filters
        domains = (await session.execute(select(Document.domain).distinct())).scalars().all()
        categories = (await session.execute(select(Document.category).distinct())).scalars().all()
        statuses = (await session.execute(select(Document.status).distinct())).scalars().all()

    return templates.TemplateResponse(
        request,
        "documents.html",
        {
            "documents": documents,
            "domains": domains,
            "categories": categories,
            "statuses": statuses,
            "current_domain": domain,
            "current_category": category,
            "current_status": status,
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
        },
    )


@router.get("/{doc_id}")
async def document_detail(request: Request, doc_id: str):
    """Show document detail with chunks."""
    async with db.session() as session:
        document = await session.get(Document, doc_id)
        if not document:
            return RedirectResponse("/documents", status_code=302)

        result = await session.execute(
            select(Chunk).where(Chunk.document_id == doc_id).order_by(Chunk.ordinal)
        )
        chunks = result.scalars().all()

    return templates.TemplateResponse(
        request,
        "document_detail.html",
        {"document": document, "chunks": chunks},
    )


@router.post("/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    domain: str = Form("general"),
    category: str = Form("article"),
):
    """Upload a new document file."""
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    async with db.session() as session:
        document = Document(
            source_path=str(file_path),
            title=file.filename,
            domain=domain,
            category=category,
            status="draft",
        )
        session.add(document)
        await session.flush()
        doc_id = document.id

    return RedirectResponse(f"/documents/{doc_id}", status_code=302)


@router.post("/{doc_id}/delete")
async def delete_document(doc_id: str):
    """Soft-delete a document (set status=archived)."""
    async with db.session() as session:
        document = await session.get(Document, doc_id)
        if document:
            document.status = "archived"

    return RedirectResponse("/documents", status_code=302)
