"""Pipeline management routes."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select

from kb_manager.models.database import IngestionJob
from kb_manager.web.deps import db, templates

router = APIRouter()

# Active background tasks
_running_tasks: dict[str, asyncio.Task] = {}


@router.get("")
async def pipeline_status(request: Request):
    """Show pipeline status page with run history."""
    from kb_manager.config import load_config

    config = load_config()
    async with db.session() as session:
        result = await session.execute(
            select(IngestionJob).order_by(IngestionJob.created_at.desc()).limit(50)
        )
        jobs = result.scalars().all()

        # Get current running job
        current_result = await session.execute(
            select(IngestionJob)
            .where(IngestionJob.status.in_(["pending", "running"]))
            .order_by(IngestionJob.created_at.desc())
            .limit(1)
        )
        current_job = current_result.scalar_one_or_none()

    return templates.TemplateResponse(
        request,
        "pipeline.html",
        {
            "jobs": jobs,
            "current_job": current_job,
            "chunking_config": config.chunking,
            "source_dir": config.source_dir,
        },
    )


@router.post("/run")
async def run_pipeline(
    request: Request,
    job_type: str = Form("incremental"),
    parent_scope: str = Form("sheet"),
    source_dir: str = Form(""),
):
    """Trigger a pipeline run as a background task."""
    from kb_manager.config import load_config
    from kb_manager.models.database import Database
    from kb_manager.pipeline.orchestrator import PipelineOrchestrator

    config = load_config()
    if not source_dir.strip():
        source_dir = config.source_dir

    # Create job record
    async with db.session() as session:
        job = IngestionJob(
            job_type=job_type,
            status="running",
            source_dir=source_dir,
            started_at=datetime.now(UTC),
            error_log=f"parent_scope={parent_scope}",
        )
        session.add(job)
        await session.flush()
        job_id = job.id

    # Run pipeline in background
    async def _run():
        try:
            async with db.session() as session:
                orchestrator = PipelineOrchestrator(database=db)
                if job_type == "full_rebuild":
                    summary = await orchestrator.run_full_rebuild(source_dir)
                else:
                    summary = await orchestrator.run_incremental(source_dir)
                # Update job record
                job_rec = await session.get(IngestionJob, job_id)
                if job_rec:
                    job_rec.status = "completed"
                    job_rec.documents_total = summary.documents_processed
                    job_rec.documents_ok = summary.documents_created + summary.documents_updated
                    job_rec.documents_failed = summary.documents_failed
                    job_rec.chunks_total = summary.chunks_created
                    job_rec.completed_at = datetime.now(UTC)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).exception("Pipeline job %s failed", job_id)
            async with db.session() as session:
                job_rec = await session.get(IngestionJob, job_id)
                if job_rec:
                    job_rec.status = "failed"
                    job_rec.error_log = str(exc)[:2000]
                    job_rec.completed_at = datetime.now(UTC)

    task = asyncio.create_task(_run())
    _running_tasks[job_id] = task

    return RedirectResponse("/pipeline", status_code=302)


@router.get("/status/{job_id}")
async def job_status(request: Request, job_id: str):
    """Show job status."""
    async with db.session() as session:
        job = await session.get(IngestionJob, job_id)
        if not job:
            return RedirectResponse("/pipeline", status_code=302)

    return templates.TemplateResponse(
        request,
        "job_status.html",
        {"job": job},
    )
