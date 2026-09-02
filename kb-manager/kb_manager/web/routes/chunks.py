"""Chunk management routes."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import func, select

from kb_manager.models.database import Chunk, Document
from kb_manager.web.deps import db, templates

router = APIRouter()


@router.get("")
async def list_chunks(request: Request, page: int = 1, per_page: int = 50):
    """List all chunks with pagination."""
    from sqlalchemy import func, select
    from sqlalchemy.orm import load_only
    
    async with db.session() as session:
        total = (await session.execute(select(func.count(Chunk.id)))).scalar() or 0
        total_pages = max(1, (total + per_page - 1) // per_page)
        offset = (page - 1) * per_page

        result = await session.execute(
            select(Chunk)
            .options(load_only(
                Chunk.id, Chunk.document_id, Chunk.ordinal,
                Chunk.chunk_type, Chunk.heading_path,
                Chunk.content,
                Chunk.token_count, Chunk.quality_score, Chunk.is_verified,
                Chunk.created_at
            ))
            .order_by(Chunk.created_at.desc())
            .offset(offset)
            .limit(per_page)
        )
        chunks = result.scalars().all()

    return templates.TemplateResponse(
        request,
        "chunks.html",
        {
            "chunks": chunks,
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
        },
    )


@router.get("/{chunk_id}")
async def chunk_detail(request: Request, chunk_id: str):
    """Show chunk detail."""
    async with db.session() as session:
        chunk = await session.get(Chunk, chunk_id)
        if not chunk:
            return RedirectResponse("/documents", status_code=302)

    return templates.TemplateResponse(
        request,
        "chunk_detail.html",
        {"chunk": chunk},
    )


@router.post("/{chunk_id}/verify")
async def verify_chunk(chunk_id: str):
    """Mark chunk as verified."""
    async with db.session() as session:
        chunk = await session.get(Chunk, chunk_id)
        if chunk:
            chunk.is_verified = True
            doc_id = chunk.document_id
        else:
            doc_id = ""

    return RedirectResponse(f"/documents/{doc_id}", status_code=302)


@router.post("/{chunk_id}/edit")
async def edit_chunk(chunk_id: str, content: str = Form(...)):
    """Edit chunk content."""
    async with db.session() as session:
        chunk = await session.get(Chunk, chunk_id)
        if chunk:
            chunk.content = content
            doc_id = chunk.document_id
        else:
            doc_id = ""

    return RedirectResponse(f"/documents/{doc_id}", status_code=302)
