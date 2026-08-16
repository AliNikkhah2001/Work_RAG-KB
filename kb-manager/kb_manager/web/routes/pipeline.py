"""Pipeline management routes."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select

from kb_manager.models.database import IngestionJob
from kb_manager.web.app import db, templates

router = APIRouter()


@router.get("")
async def pipeline_status(request: Request):
    """Show pipeline status page with run history."""
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
        },
    )


@router.post("/run")
async def run_pipeline(request: Request, job_type: str = Form("incremental")):
    """Trigger a pipeline run."""
    async with db.session() as session:
        job = IngestionJob(
            job_type=job_type,
            status="pending",
            started_at=datetime.now(UTC),
        )
        session.add(job)

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
