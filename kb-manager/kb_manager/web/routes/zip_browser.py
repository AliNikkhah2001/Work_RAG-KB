"""Zip browser + selective pipeline ingest.

Upload a zip (Persian filenames UTF-8 safe), browse its tree with checkboxes,
select folders/files, and run the ingestion pipeline only on selected items.

Flow:
  GET  /transparency/zip          → upload form
  POST /transparency/zip/preview  → save zip, list contents, show tree
  POST /transparency/zip/ingest   → extract selected → run pipeline (background)
No PowerShell needed — all via zipfile + orchestrator.
"""

from __future__ import annotations

import asyncio
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from kb_manager.models.database import IngestionJob
from kb_manager.web.deps import db, templates

router = APIRouter()

STAGE_ROOT = Path("data/uploads/zip_stage")
STAGE_ROOT.mkdir(parents=True, exist_ok=True)

_running: dict[str, asyncio.Task] = {}


def _safe_join(base: Path, rel: str) -> Path | None:
    """Prevent zip-slip: ensure rel stays inside base."""
    try:
        p = (base / rel).resolve()
        base_r = base.resolve()
        if str(p).startswith(str(base_r)):
            return p
    except Exception:
        return None
    return None


def _build_tree(namelist: list[str], infos: dict[str, zipfile.ZipInfo]) -> dict:
    """Build nested dict tree for display."""
    tree: dict[str, Any] = {}
    for name in sorted(namelist):
        parts = [p for p in name.split("/") if p]
        is_dir = name.endswith("/")
        cur = tree
        for i, part in enumerate(parts):
            last = i == len(parts) - 1
            if last and not is_dir:
                cur.setdefault(part, {"__file": True, "__path": name, "__size": infos[name].file_size})
            else:
                cur.setdefault(part, {})
                cur = cur[part]
    return tree


def _list_entries(zip_path: Path) -> tuple[list[str], dict[str, zipfile.ZipInfo]]:
    with zipfile.ZipFile(zip_path, "r") as z:
        infos = {i.filename: i for i in z.infolist()}
        return z.namelist(), infos


@router.get("/zip", response_class=HTMLResponse)
async def zip_upload_page(request: Request):
    return templates.TemplateResponse(request, "zip_browser.html", {})


@router.post("/zip/preview", response_class=HTMLResponse)
async def zip_preview(request: Request, file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".zip"):
        return templates.TemplateResponse(request, "zip_browser.html", {"error": "Please upload a .zip file"}, status_code=400)
    content = await file.read()
    if len(content) > 200 * 1024 * 1024:
        return templates.TemplateResponse(request, "zip_browser.html", {"error": "Zip too large (>200MB)"}, status_code=400)
    zid = uuid.uuid4().hex[:12]
    zip_path = STAGE_ROOT / f"{zid}.zip"
    zip_path.write_bytes(content)
    try:
        namelist, infos = _list_entries(zip_path)
    except zipfile.BadZipFile as e:
        zip_path.unlink(missing_ok=True)
        return templates.TemplateResponse(request, "zip_browser.html", {"error": f"Bad zip: {e}"}, status_code=400)

    # Build flat list for template with dir/file classification
    entries: list[dict] = []
    for name in sorted(namelist):
        info = infos[name]
        is_dir = name.endswith("/")
        # skip empty dir entries that are just prefixes? keep for tree
        entries.append({
            "path": name,
            "is_dir": is_dir,
            "size": info.file_size if not is_dir else 0,
            "depth": name.rstrip("/").count("/"),
            "basename": name.rstrip("/").split("/")[-1] if name.rstrip("/") else name,
        })

    return templates.TemplateResponse(request, "zip_preview.html", {
        "zip_id": zid,
        "zip_name": file.filename,
        "zip_path": str(zip_path),
        "entries": entries,
        "total_files": sum(1 for e in entries if not e["is_dir"]),
        "total_dirs": sum(1 for e in entries if e["is_dir"]),
    })


@router.post("/zip/ingest", response_class=HTMLResponse)
async def zip_ingest(
    request: Request,
    zip_id: str = Form(...),
    selected: list[str] = Form([]),
    job_type: str = Form("full_rebuild"),
    parent_scope: str = Form("sheet"),
):
    zip_path = STAGE_ROOT / f"{zip_id}.zip"
    if not zip_path.exists():
        return templates.TemplateResponse(request, "zip_browser.html", {"error": "Zip expired or not found — re-upload"}, status_code=400)
    if not selected:
        return templates.TemplateResponse(request, "zip_browser.html", {"error": "Select at least one file/folder"}, status_code=400)

    # Resolve selected to explicit file list
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            all_names = set(z.namelist())
            # Expand folder selections to all files under that folder
            files_to_extract: set[str] = set()
            for sel in selected:
                sel = sel.strip()
                if not sel:
                    continue
                if sel in all_names and not sel.endswith("/"):
                    files_to_extract.add(sel)
                else:
                    # folder prefix
                    prefix = sel.rstrip("/") + "/"
                    for n in all_names:
                        if n.startswith(prefix) and not n.endswith("/"):
                            files_to_extract.add(n)
            if not files_to_extract:
                return templates.TemplateResponse(request, "zip_browser.html", {"error": "No files matched selection"}, status_code=400)

            # Extract only selected files to stage dir
            extract_root = STAGE_ROOT / f"{zip_id}_extracted"
            extract_root.mkdir(parents=True, exist_ok=True)
            # clean previous
            for p in extract_root.rglob("*"):
                if p.is_file():
                    p.unlink()
            for name in files_to_extract:
                target = _safe_join(extract_root, name)
                if target is None:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as src, open(target, "wb") as dst:
                    dst.write(src.read())
    except Exception as e:
        return templates.TemplateResponse(request, "zip_browser.html", {"error": f"Extract failed: {e}"}, status_code=400)

    extract_root_str = str(extract_root.resolve())

    # Create job and run pipeline in background (reuse pipeline.py logic)
    async with db.session() as session:
        job = IngestionJob(
            job_type=job_type,
            status="running",
            source_dir=extract_root_str,
            started_at=datetime.now(UTC),
            error_log=None,
        )
        session.add(job)
        await session.flush()
        job_id = job.id

    async def _run():
        try:
            async with db.session() as session:
                from kb_manager.chunker.semantic import SemanticChunker
                from kb_manager.pipeline.orchestrator import PipelineOrchestrator

                scope = parent_scope if parent_scope in ("sheet", "document") else "sheet"
                chunker = SemanticChunker(parent_scope=scope)
                orch = PipelineOrchestrator(database=db, chunker=chunker)
                if job_type == "full_rebuild":
                    summary = await orch.run_full_rebuild(extract_root_str)
                else:
                    summary = await orch.run_incremental(extract_root_str)
                rec = await session.get(IngestionJob, job_id)
                if rec:
                    rec.status = "completed"
                    rec.documents_total = summary.documents_processed
                    rec.documents_ok = summary.documents_created + summary.documents_updated
                    rec.documents_failed = summary.documents_failed
                    rec.chunks_total = summary.chunks_created
                    rec.completed_at = datetime.now(UTC)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).exception("Zip ingest job %s failed", job_id)
            async with db.session() as session:
                rec = await session.get(IngestionJob, job_id)
                if rec:
                    rec.status = "failed"
                    rec.error_log = str(exc)[:3000]
                    rec.completed_at = datetime.now(UTC)

    task = asyncio.create_task(_run())
    _running[job_id] = task
    task.add_done_callback(lambda t, jid=job_id: _running.pop(jid, None))

    return RedirectResponse("/pipeline", status_code=302)
