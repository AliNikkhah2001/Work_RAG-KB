"""Monitoring routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from sqlalchemy import func, select

from kb_manager.models.database import Chunk, Document, IngestionJob, RetrievalLog
from kb_manager.web.app import db, templates

router = APIRouter()


@router.get("")
async def monitoring_dashboard(request: Request):
    """Show monitoring dashboard."""
    async with db.session() as session:
        doc_count = (await session.execute(select(func.count(Document.id)))).scalar() or 0
        chunk_count = (await session.execute(select(func.count(Chunk.id)))).scalar() or 0
        job_count = (await session.execute(select(func.count(IngestionJob.id)))).scalar() or 0
        retrieval_count = (await session.execute(select(func.count(RetrievalLog.id)))).scalar() or 0

        # Documents by status
        status_counts = {}
        result = await session.execute(
            select(Document.status, func.count(Document.id)).group_by(Document.status)
        )
        for row in result.fetchall():
            status_counts[row[0]] = row[1]

        # Chunks by verification status
        verified_count = (
            await session.execute(select(func.count(Chunk.id)).where(Chunk.is_verified))
        ).scalar() or 0
        unverified_count = chunk_count - verified_count

    return templates.TemplateResponse(
        request,
        "monitoring.html",
        {
            "doc_count": doc_count,
            "chunk_count": chunk_count,
            "job_count": job_count,
            "retrieval_count": retrieval_count,
            "status_counts": status_counts,
            "verified_count": verified_count,
            "unverified_count": unverified_count,
        },
    )


@router.get("/staleness")
async def staleness_report(request: Request):
    """Show staleness report."""
    async with db.session() as session:
        # Find documents that haven't been updated recently
        from datetime import timedelta

        from sqlalchemy import or_

        from kb_manager.models.database import _utcnow

        threshold = _utcnow() - timedelta(days=30)

        result = await session.execute(
            select(Document)
            .where(or_(Document.updated_at < threshold, Document.content_hash == ""))
            .order_by(Document.updated_at.asc())
        )
        stale_documents = result.scalars().all()

        # Calculate staleness stats
        total_docs = (await session.execute(select(func.count(Document.id)))).scalar() or 0
        stale_count = len(stale_documents)
        fresh_count = total_docs - stale_count

    return templates.TemplateResponse(
        request,
        "staleness.html",
        {
            "stale_documents": stale_documents,
            "total_docs": total_docs,
            "stale_count": stale_count,
            "fresh_count": fresh_count,
        },
    )


@router.get("/metrics")
async def metrics_summary(request: Request):
    """Show metrics summary."""
    async with db.session() as session:
        # Average chunks per document
        avg_chunks = (await session.execute(select(func.avg(Document.chunk_count)))).scalar() or 0

        # Total tokens
        total_tokens = (await session.execute(select(func.sum(Chunk.token_count)))).scalar() or 0

        # Average quality score
        avg_quality = (await session.execute(select(func.avg(Chunk.quality_score)))).scalar() or 0

        # Documents by domain
        domain_counts = {}
        result = await session.execute(
            select(Document.domain, func.count(Document.id)).group_by(Document.domain)
        )
        for row in result.fetchall():
            domain_counts[row[0]] = row[1]

        # Documents by category
        category_counts = {}
        result = await session.execute(
            select(Document.category, func.count(Document.id)).group_by(Document.category)
        )
        for row in result.fetchall():
            category_counts[row[0]] = row[1]

        # Job success rate
        total_jobs = (await session.execute(select(func.count(IngestionJob.id)))).scalar() or 0
        successful_jobs = (
            await session.execute(
                select(func.count(IngestionJob.id)).where(IngestionJob.status == "completed")
            )
        ).scalar() or 0
        success_rate = (successful_jobs / total_jobs * 100) if total_jobs > 0 else 0

    return templates.TemplateResponse(
        request,
        "metrics.html",
        {
            "avg_chunks": round(avg_chunks, 1),
            "total_tokens": total_tokens,
            "avg_quality": round(avg_quality, 2),
            "domain_counts": domain_counts,
            "category_counts": category_counts,
            "total_jobs": total_jobs,
            "successful_jobs": successful_jobs,
            "success_rate": round(success_rate, 1),
        },
    )
