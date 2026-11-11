"""Benchmarks UI — proxies to Work_RAG-Evaluation_Suite (sole eval backend).

Retrieval Hit@k and multi-turn suites live in components/evaluation.
This router keeps KB version snapshots and forwards eval jobs to EVAL_SUITE_URL.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from typing import Any

import httpx
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from kb_manager.config import PROJECT_ROOT
from kb_manager.web.deps import templates
from kb_manager.web.eval_suite_client import EvalSuiteClient

try:
    from kb_manager.versioning.snapshot import create_snapshot, list_snapshots
except Exception:  # pragma: no cover
    create_snapshot = None  # type: ignore[assignment]
    list_snapshots = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)
router = APIRouter()

DATA_DIR = PROJECT_ROOT / "data"
PLOTS_DIR = DATA_DIR / "plots"
RESULTS_JSON = DATA_DIR / "benchmark_results.json"
IR_METRICS_JSON = DATA_DIR / "ir_metrics.json"

EVAL_SUITE_URL = os.environ.get("EVAL_SUITE_URL", "http://127.0.0.1:8300")


def _client() -> EvalSuiteClient:
    return EvalSuiteClient(EVAL_SUITE_URL)


async def _suite_available() -> tuple[bool, str]:
    try:
        h = await _client().health()
        return True, str(h.get("status") or "ok")
    except Exception as exc:
        return False, str(exc)[:200]


@router.get("")
async def benchmarks_page(request: Request):
    """Benchmarks dashboard — results come from Evaluation Suite."""
    ok, msg = await _suite_available()
    datasets: list[str] = []
    hitk = None
    multiturn = None
    if ok:
        with contextlib.suppress(Exception):
            datasets = await _client().list_datasets()
        with contextlib.suppress(Exception):
            hitk = await _client().results("hitk")
        with contextlib.suppress(Exception):
            multiturn = await _client().results("multiturn")

    # Mirror suite hitk into local JSON so older template snippets still work
    latest_results = None
    ir_metrics = None
    if hitk:
        latest_results = {
            "total_queries": hitk.get("n"),
            "top_k": hitk.get("top_k"),
            "overall": {
                "hit_rate": hitk.get("hitk_rate"),
                "top1_hit_rate": hitk.get("hit1_rate"),
                "mrr": hitk.get("mrr"),
            },
            "by_format": {
                k: {
                    "queries": v.get("n"),
                    "hit_rate": v.get("hitk_rate"),
                    "top1_hit_rate": v.get("hit1_rate"),
                    "mrr": v.get("mrr"),
                    "avg_rank": None,
                    "avg_latency_ms": None,
                }
                for k, v in (hitk.get("by_category") or {}).items()
            },
            "source": "evaluation-suite",
        }
        ir_metrics = {
            "top_k": hitk.get("top_k"),
            "metrics": {
                "n": hitk.get("n"),
                "hit1_rate": hitk.get("hit1_rate"),
                "hitk_rate": hitk.get("hitk_rate"),
                "mrr": hitk.get("mrr"),
            },
        }
        with contextlib.suppress(Exception):
            RESULTS_JSON.write_text(json.dumps(latest_results, ensure_ascii=False, indent=2), encoding="utf-8")
            IR_METRICS_JSON.write_text(json.dumps(ir_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    snapshots = list_snapshots() if list_snapshots else []
    plots = sorted(p.name for p in PLOTS_DIR.glob("*.png")) if PLOTS_DIR.exists() else []

    return templates.TemplateResponse(
        request,
        "benchmarks.html",
        {
            "datasets": datasets,
            "formats": [],
            "latest_results": latest_results,
            "ir_metrics": ir_metrics,
            "plots": plots,
            "snapshots": snapshots,
            "hnsw_results": None,
            "eval_suite_ok": ok,
            "eval_suite_msg": msg,
            "eval_suite_url": EVAL_SUITE_URL,
            "multiturn_summary": multiturn,
        },
    )


@router.post("/run")
async def run_benchmark(
    dataset: str = Form("retrieval/eval_clean_remapped.json"),
    top_k: int = Form(5),
    sample_size: int = Form(0),
    remap: str = Form("0"),
):
    """Start Hit@k on Evaluation Suite and redirect to status."""
    ok, msg = await _suite_available()
    if not ok:
        raise HTTPException(
            status_code=503,
            detail=f"Evaluation Suite unavailable at {EVAL_SUITE_URL}: {msg}",
        )
    limit = sample_size if sample_size and sample_size > 0 else None
    remap_from = "retrieval/eval_clean_legacy.json" if remap in {"1", "true", "on"} else None
    # If remapped file missing, force remap
    if dataset.endswith("eval_clean_remapped.json") and remap_from is None:
        remap_from = "retrieval/eval_clean_legacy.json"
    try:
        started = await _client().run_hitk(
            dataset=dataset,
            top_k=top_k,
            limit=limit,
            remap_from=remap_from,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Suite call failed: {exc}") from exc
    job_id = started.get("job_id")
    return RedirectResponse(f"/benchmarks?job={job_id}&suite=hitk", status_code=303)


@router.post("/multiturn/run")
async def run_multiturn():
    ok, msg = await _suite_available()
    if not ok:
        raise HTTPException(status_code=503, detail=f"Evaluation Suite unavailable: {msg}")
    started = await _client().run_multiturn()
    job_id = started.get("job_id")
    return RedirectResponse(f"/benchmarks?job={job_id}&suite=multiturn", status_code=303)


@router.get("/status/{job_id}")
async def benchmark_status(job_id: str):
    """Poll Evaluation Suite job progress."""
    try:
        return JSONResponse(await _client().job(job_id))
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/result")
async def latest_result():
    """Latest Hit@k results from Evaluation Suite."""
    try:
        data = await _client().results("hitk")
        return JSONResponse(data)
    except httpx.HTTPError as exc:
        if RESULTS_JSON.exists():
            return FileResponse(RESULTS_JSON, media_type="application/json")
        raise HTTPException(status_code=404, detail=f"No results yet ({exc})") from exc


@router.get("/plot/{name}")
async def serve_plot(name: str):
    safe = (PLOTS_DIR / name).resolve()
    if not safe.exists() or safe.parent != PLOTS_DIR.resolve():
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(safe, media_type="image/png")


# ---------------------------------------------------------------------------
# Version snapshots (KB-owned; not evaluation)
# ---------------------------------------------------------------------------

@router.post("/snapshot/create")
async def create_kb_snapshot(label: str = Form(...), notes: str = Form("")):
    if create_snapshot is None:
        raise HTTPException(status_code=501, detail="snapshot module unavailable")
    label = label.strip().replace(" ", "_")
    if not label:
        raise HTTPException(status_code=400, detail="Label required")

    def _work() -> str:
        create_snapshot(label, notes=notes)
        return label

    created = await asyncio.to_thread(_work)
    return RedirectResponse(f"/benchmarks?snapshot={created}", status_code=303)


@router.get("/comparison")
async def comparison_page(request: Request):
    latest_results = None
    with contextlib.suppress(Exception):
        latest_results = await _client().results("hitk")
    comparison_data = None
    comparison_path = DATA_DIR / "benchmark_comparison.json"
    if comparison_path.exists():
        with contextlib.suppress(Exception):
            comparison_data = json.loads(comparison_path.read_text(encoding="utf-8"))
    plots = sorted(p.name for p in PLOTS_DIR.glob("*.png")) if PLOTS_DIR.exists() else []
    return templates.TemplateResponse(
        request,
        "comparison.html",
        {"latest_results": latest_results, "comparison_data": comparison_data, "plots": plots},
    )


@router.get("/comparison/data")
async def comparison_data():
    comparison_path = DATA_DIR / "benchmark_comparison.json"
    if comparison_path.exists():
        return JSONResponse(json.loads(comparison_path.read_text(encoding="utf-8")))
    with contextlib.suppress(Exception):
        return JSONResponse({"latest": await _client().results("hitk")})
    raise HTTPException(status_code=404, detail="No comparison data")


@router.get("/snapshots")
async def snapshots_page(request: Request):
    return templates.TemplateResponse(
        request, "snapshots.html", {"snapshots": list_snapshots() if list_snapshots else []}
    )


@router.get("/snapshots/{label}")
async def snapshot_detail(request: Request, label: str):
    from kb_manager.versioning.snapshot import VERSIONS_ROOT

    snap_dir = (VERSIONS_ROOT / label).resolve()
    if not snap_dir.exists() or snap_dir.parent != VERSIONS_ROOT.resolve():
        raise HTTPException(status_code=404, detail="Snapshot not found")
    manifest: dict[str, Any] = {}
    if (snap_dir / "manifest.json").exists():
        with contextlib.suppress(Exception):
            manifest = json.loads((snap_dir / "manifest.json").read_text(encoding="utf-8"))
    files = sorted(p.name for p in snap_dir.glob("*"))
    return templates.TemplateResponse(
        request,
        "snapshot_detail.html",
        {
            "label": label,
            "manifest": manifest,
            "files": files,
            "benchmarks": [],
            "plots": [],
        },
    )


@router.get("/snapshots/{label}/file/{name:path}")
async def snapshot_file(label: str, name: str):
    from kb_manager.versioning.snapshot import VERSIONS_ROOT

    snap_dir = (VERSIONS_ROOT / label).resolve()
    if not snap_dir.exists() or snap_dir.parent != VERSIONS_ROOT.resolve():
        raise HTTPException(status_code=404, detail="Snapshot not found")
    safe = (snap_dir / name).resolve()
    if (not safe.exists()) or (snap_dir not in safe.parents and safe.parent != snap_dir):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(safe)


# Retired local RAGAS / massive / IVA runners — use Evaluation Suite.
@router.post("/ragas")
async def run_ragas_evaluation():
    raise HTTPException(
        status_code=410,
        detail="Local RAGAS removed. Use Evaluation Suite multiturn/hitk at EVAL_SUITE_URL.",
    )


@router.get("/ragas/result")
async def ragas_result():
    raise HTTPException(status_code=410, detail="Local RAGAS removed.")


@router.get("/massive")
async def massive_page(request: Request):
    raise HTTPException(
        status_code=410,
        detail="Massive QA bench moved to Evaluation Suite (Hit@k datasets).",
    )


@router.post("/massive/run")
async def run_massive():
    raise HTTPException(status_code=410, detail="Use Evaluation Suite Hit@k.")


@router.get("/massive/result")
async def massive_result():
    raise HTTPException(status_code=410, detail="Use Evaluation Suite Hit@k.")


@router.post("/iva/run")
async def run_iva_route():
    raise HTTPException(status_code=410, detail="IVA bench retired; use Evaluation Suite.")


@router.get("/iva/result")
async def iva_result_json():
    raise HTTPException(status_code=410, detail="IVA bench retired.")
