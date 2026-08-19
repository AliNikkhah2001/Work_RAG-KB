"""Benchmarks, performance plots, and KB version snapshots.

Provides the web UI for:
  * running a retrieval benchmark over a test dataset (multi-format),
  * rendering performance plots (Hit@K, MRR, latency) as PNGs,
  * creating / listing immutable version snapshots of the KB.

Benchmark jobs run as background asyncio tasks on the app event loop so the
aiosqlite engine / BM25 index cache are reused correctly.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from kb_manager.config import PROJECT_ROOT
from kb_manager.web.app import templates

try:
    from kb_manager.evaluation.benchmark import summarize_ir_metrics
    from kb_manager.evaluation.plots import (
        plot_duplicate_stats,
        render_benchmark_plots,
    )
    from kb_manager.evaluation.query_formats import format_list
    from kb_manager.versioning.snapshot import (
        create_snapshot,
        list_snapshots,
    )
except Exception as exc:  # pragma: no cover - import-time safety
    raise RuntimeError(f"benchmark imports failed: {exc}") from exc

logger = logging.getLogger(__name__)

router = APIRouter()

DATA_DIR = PROJECT_ROOT / "data"
PLOTS_DIR = DATA_DIR / "plots"
RESULTS_JSON = DATA_DIR / "benchmark_results.json"
IR_METRICS_JSON = DATA_DIR / "ir_metrics.json"
DUP_STATS_JSON = DATA_DIR / "qa_duplication.json"

_JOBS: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Benchmark execution helpers
# ---------------------------------------------------------------------------

def _adapter(dataset_path: str) -> list[dict]:
    with open(dataset_path, encoding="utf-8") as f:
        return json.load(f)


def _list_datasets() -> list[str]:
    if not DATA_DIR.exists():
        return []
    excluded = {"benchmark_results.json", "ir_metrics.json", "qa_duplication.json"}
    out = []
    for p in DATA_DIR.glob("*.json"):
        if p.name in excluded:
            continue
        if "_questions" in p.name or p.name in ("eval_clean.json",):
            out.append(p.name)
    return sorted(out)


async def _run_benchmark(
    job_id: str,
    dataset_name: str,
    top_k: int,
    sample_size: int,
) -> None:
    """Search every dataset query against the live KB and record results."""
    from kb_manager.evaluation.benchmark import AsyncBenchmarkRunner
    from kb_manager.web.routes.search import search_knowledge_base

    job = _JOBS[job_id]
    dataset_path = DATA_DIR / dataset_name
    dataset = _adapter(str(dataset_path))

    if sample_size and sample_size > 0 and sample_size < len(dataset):
        dataset = dataset[:sample_size]

    async def _search(query: str, k: int):
        # Search results are SearchResult models with .chunk_id / .hybrid_score
        steps = await search_knowledge_base(query, k)
        return [(r.chunk_id, r.hybrid_score) for r in steps.final_results]

    runner = AsyncBenchmarkRunner(_search, top_k=top_k, version="live-kb")
    result = await runner._run_async(
        dataset,
        progress=lambda done, total: job.update({"progress": done, "total": total}),
    )

    job["progress"] = len(dataset)
    job["total"] = len(dataset)

    # IR metrics (ranx when available, pure-Python fallback).
    try:
        ir = summarize_ir_metrics(result)
    except Exception:
        ir = {}
    result_to_dict = result.to_dict()

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(result_to_dict, f, ensure_ascii=False, indent=2)
    with open(IR_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump({"top_k": top_k, "metrics": ir}, f, ensure_ascii=False, indent=2)

    # QA duplication stats from the current KB.
    from kb_manager.versioning.snapshot import export_compact

    export = export_compact(str(PROJECT_ROOT / "data" / "kb_test.db"))
    dups = export["counts"]
    with open(DUP_STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(dups, f, ensure_ascii=False, indent=2)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        render_benchmark_plots(str(RESULTS_JSON), str(PLOTS_DIR))
        plot_duplicate_stats(DUP_STATS_JSON, str(PLOTS_DIR / "qa_duplication.png"))
    except Exception as exc:
        logger.warning("plot generation failed: %s", exc)

    job["status"] = "done"
    job["finished_at"] = datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("")
async def benchmarks_page(request: Request):
    """Benchmarks + performance plots + snapshot dashboard."""
    snapshots = list_snapshots()

    latest_results = None
    if RESULTS_JSON.exists():
        try:
            latest_results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        except Exception:
            latest_results = None

    ir_metrics = None
    if IR_METRICS_JSON.exists():
        with contextlib.suppress(Exception):
            ir_metrics = json.loads(IR_METRICS_JSON.read_text(encoding="utf-8"))

    plots = []
    if PLOTS_DIR.exists():
        plots = sorted(p.name for p in PLOTS_DIR.glob("*.png"))

    return templates.TemplateResponse(
        request,
        "benchmarks.html",
        {
            "datasets": _list_datasets(),
            "formats": format_list(),
            "latest_results": latest_results,
            "ir_metrics": ir_metrics,
            "plots": plots,
            "snapshots": snapshots,
        },
    )


@router.post("/run")
async def run_benchmark(
    dataset: str = Form("test_questions.json"),
    top_k: int = Form(5),
    sample_size: int = Form(0),
):
    """Start a background benchmark job and redirect to its status page."""
    dataset = dataset.strip()
    if not (DATA_DIR / dataset).exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset}")
    if top_k < 1 or top_k > 50:
        raise HTTPException(status_code=400, detail="top_k must be 1..50")

    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "status": "running",
        "progress": 0,
        "total": 1,
        "dataset": dataset,
        "top_k": top_k,
        "sample_size": sample_size,
        "started_at": datetime.now(UTC).isoformat(),
        "finished_at": None,
        "error": "",
    }
    _JOBS[job_id] = job
    _JOBS[job_id]["_task"] = asyncio.create_task(
        _run_benchmark(job_id, dataset, top_k, sample_size)
    )
    return RedirectResponse(f"/benchmarks?job={job_id}", status_code=303)


@router.get("/status/{job_id}")
async def benchmark_status(job_id: str):
    """Poll a benchmark job's progress (JSON)."""
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job")
    return JSONResponse(job)


@router.get("/result")
async def latest_result():
    """Latest benchmark results as JSON."""
    if not RESULTS_JSON.exists():
        raise HTTPException(status_code=404, detail="No benchmark results yet")
    return FileResponse(RESULTS_JSON, media_type="application/json")


@router.get("/plot/{name}")
async def serve_plot(name: str):
    """Serve a generated performance plot PNG."""
    safe = (PLOTS_DIR / name).resolve()
    if not safe.exists() or safe.parent != PLOTS_DIR.resolve():
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(safe, media_type="image/png")


# ---------------------------------------------------------------------------
# Version snapshots
# ---------------------------------------------------------------------------

@router.post("/snapshot/create")
async def create_kb_snapshot(
    label: str = Form(...),
    notes: str = Form(""),
):
    """Create a new immutable version snapshot of the current KB state."""
    label = label.strip().replace(" ", "_")
    if not label:
        raise HTTPException(status_code=400, detail="Label required")

    def _work() -> str:
        create_snapshot(label, notes=notes)
        return label

    created_label = await asyncio.to_thread(_work)
    return RedirectResponse(f"/benchmarks?snapshot={created_label}", status_code=303)


@router.get("/comparison")
async def comparison_page(request: Request):
    """Version comparison page with interactive charts."""
    # Load latest benchmark results
    latest_results = None
    if RESULTS_JSON.exists():
        try:
            latest_results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        except Exception:
            latest_results = None

    # Load version comparison data (if available)
    comparison_data = None
    comparison_path = DATA_DIR / "benchmark_comparison.json"
    if comparison_path.exists():
        try:
            comparison_data = json.loads(comparison_path.read_text(encoding="utf-8"))
        except Exception:
            comparison_data = None

    # Get list of available plots
    plots = []
    if PLOTS_DIR.exists():
        plots = sorted(p.name for p in PLOTS_DIR.glob("*.png"))

    return templates.TemplateResponse(
        request,
        "comparison.html",
        {
            "latest_results": latest_results,
            "comparison_data": comparison_data,
            "plots": plots,
        },
    )


@router.get("/snapshots")
async def snapshots_page(request: Request):
    """List all version snapshots."""
    return templates.TemplateResponse(
        request,
        "snapshots.html",
        {"snapshots": list_snapshots()},
    )


@router.get("/snapshots/{label}")
async def snapshot_detail(request: Request, label: str):
    """Show the contents of one version snapshot."""
    from kb_manager.versioning.snapshot import VERSIONS_ROOT

    snap_dir = (VERSIONS_ROOT / label).resolve()
    if not snap_dir.exists() or snap_dir.parent != VERSIONS_ROOT.resolve():
        raise HTTPException(status_code=404, detail="Snapshot not found")

    manifest = {}
    manifest_path = snap_dir / "manifest.json"
    if manifest_path.exists():
        with contextlib.suppress(Exception):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    files = sorted(p.name for p in snap_dir.glob("*"))
    benchmarks = []
    bench_dir = snap_dir / "benchmarks"
    if bench_dir.exists():
        benchmarks = sorted(p.name for p in bench_dir.glob("*"))
    plots = []
    plots_dir = snap_dir / "plots"
    if plots_dir.exists():
        plots = sorted(p.name for p in plots_dir.glob("*.png"))

    return templates.TemplateResponse(
        request,
        "snapshot_detail.html",
        {
            "label": label,
            "manifest": manifest,
            "files": files,
            "benchmarks": benchmarks,
            "plots": plots,
        },
    )


@router.get("/snapshots/{label}/file/{name}")
async def snapshot_file(label: str, name: str):
    """Download a file archived inside a snapshot."""
    from kb_manager.versioning.snapshot import VERSIONS_ROOT

    snap_dir = (VERSIONS_ROOT / label).resolve()
    if not snap_dir.exists() or snap_dir.parent != VERSIONS_ROOT.resolve():
        raise HTTPException(status_code=404, detail="Snapshot not found")
    safe = (snap_dir / name).resolve()
    if (not safe.exists()) or (snap_dir not in safe.parents and safe.parent != snap_dir):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(safe)
