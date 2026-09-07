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
from kb_manager.web.deps import templates

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

def _evict_old_jobs() -> None:
    """Keep at most 100 jobs, evict oldest completed (D30 fix)."""
    if len(_JOBS) <= 100:
        return
    # Sort by started_at, keep newest 100
    sorted_items = sorted(_JOBS.items(), key=lambda kv: kv[1].get("started_at", ""))
    for k, _ in sorted_items[:-100]:
        _JOBS.pop(k, None)


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
    from kb_manager.evaluation.benchmark import BenchmarkRunner
    from kb_manager.web.routes.search import search_knowledge_base_sync

    job = _JOBS[job_id]
    try:
        dataset_path = DATA_DIR / dataset_name
        dataset = _adapter(str(dataset_path))

        if sample_size and sample_size > 0 and sample_size < len(dataset):
            dataset = dataset[:sample_size]

        # Initialize progress correctly
        job["total"] = len(dataset)
        job["progress"] = 0
        if len(dataset) == 0:
            raise ValueError(f"Dataset {dataset_name} is empty or not found at {dataset_path}")

        def _search_sync(query: str, k: int):
            steps = search_knowledge_base_sync(query, k)
            return [(r.chunk_id, r.hybrid_score) for r in steps.final_results]

        runner = BenchmarkRunner(_search_sync, top_k=top_k, version="live-kb")
        result = await asyncio.to_thread(
            runner.run, dataset,
            progress=lambda done, total: job.update({"progress": done, "total": total}),
        )

        job["progress"] = len(dataset)
        job["total"] = len(dataset)
    except Exception as e:
        logger.exception("Benchmark job %s failed", job_id)
        job["status"] = "error"
        job["error"] = str(e)[:500]
        job["finished_at"] = datetime.now(UTC).isoformat()
        return

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

    # F24 fix: do not fabricate metrics when results are missing — show null/unavailable
    if not latest_results or not latest_results.get("total_queries"):
        latest_results = None  # template will show "no benchmark data" banner

    ir_metrics = None
    if IR_METRICS_JSON.exists():
        with contextlib.suppress(Exception):
            ir_metrics = json.loads(IR_METRICS_JSON.read_text(encoding="utf-8"))

    plots = []
    if PLOTS_DIR.exists():
        plots = sorted(p.name for p in PLOTS_DIR.glob("*.png"))

    # HNSW benchmark (GPU vs CPU, pgvector vs file)
    hnsw_results = None
    hnsw_path = DATA_DIR / "hnsw_benchmark_detailed.json"
    if hnsw_path.exists():
        with contextlib.suppress(Exception):
            hnsw_results = json.loads(hnsw_path.read_text(encoding="utf-8"))

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
            "hnsw_results": hnsw_results,
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

    # Pre-read dataset size so the progress bar shows 0/N not 0/1
    try:
        full_dataset = _adapter(str(DATA_DIR / dataset))
        total_queries = len(full_dataset)
        if sample_size and 0 < sample_size < total_queries:
            total_queries = sample_size
    except Exception:
        total_queries = 1

    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "status": "running",
        "progress": 0,
        "total": total_queries,
        "dataset": dataset,
        "top_k": top_k,
        "sample_size": sample_size,
        "started_at": datetime.now(UTC).isoformat(),
        "finished_at": None,
        "error": "",
    }
    _evict_old_jobs()
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
    # Filter out internal keys (like _task) that can't be serialized to JSON
    return JSONResponse({k: v for k, v in job.items() if not k.startswith("_")})


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

    # Load version comparison data
    comparison_data = None
    comparison_path = DATA_DIR / "benchmark_comparison.json"
    if comparison_path.exists():
        try:
            comparison_data = json.loads(comparison_path.read_text(encoding="utf-8"))
        except Exception:
            comparison_data = None

    # Get list of available plots (including version comparison plots)
    plots = []
    if PLOTS_DIR.exists():
        # Prioritize version comparison plots first
        version_plots = sorted(p.name for p in PLOTS_DIR.glob("version_comparison*.png"))
        other_plots = sorted(p.name for p in PLOTS_DIR.glob("*.png") if not p.name.startswith("version_comparison"))
        plots = version_plots + other_plots

    return templates.TemplateResponse(
        request,
        "comparison.html",
        {
            "latest_results": latest_results,
            "comparison_data": comparison_data,
            "plots": plots,
        },
    )


@router.get("/comparison/data")
async def comparison_data():
    """Version comparison JSON (dynamic)."""
    comparison_path = DATA_DIR / "benchmark_comparison.json"
    if comparison_path.exists():
        try:
            data = json.loads(comparison_path.read_text(encoding="utf-8"))
            return JSONResponse(data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    # Fallback: build from latest benchmark
    if RESULTS_JSON.exists():
        try:
            latest = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
            return JSONResponse({"latest": latest, "note": "benchmark_comparison.json not found, showing latest only"})
        except Exception:
            pass
    raise HTTPException(status_code=404, detail="No comparison data")


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


@router.get("/snapshots/{label}/file/{name:path}")
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


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

RAGAS_RESULTS_JSON = DATA_DIR / "ragas_results.json"


async def _run_ragas_evaluation(
    job_id: str,
    dataset_name: str,
    top_k: int,
    sample_size: int,
) -> None:
    """Run RAGAS quality evaluation (faithfulness, answer relevance, context recall)."""
    from kb_manager.config import load_config
    from kb_manager.evaluation.ragas_metrics import RagasEvaluator
    from kb_manager.web.routes.search import search_knowledge_base_sync

    job = _JOBS[job_id]
    try:
        dataset_path = DATA_DIR / dataset_name
        dataset = _adapter(str(dataset_path))

        if sample_size and sample_size > 0 and sample_size < len(dataset):
            dataset = dataset[:sample_size]

        job["total"] = len(dataset)
        job["progress"] = 0
        if len(dataset) == 0:
            raise ValueError(f"Dataset {dataset_name} is empty")

        config = load_config()
        evaluator = RagasEvaluator(config.ragas)

        if not evaluator.available():
            raise RuntimeError(
                "RAGAS dependencies not installed. "
                "Install with: pip install ragas langchain-openai datasets"
            )

        questions: list[str] = []
        answers: list[str] = []
        retrieved_contexts: list[list[str]] = []
        ground_truth: list[str] = []

        for i, item in enumerate(dataset):
            query = item.get("query", "")
            gt_answer = item.get("answer", item.get("ground_truth", ""))

            # Search
            steps = search_knowledge_base_sync(query, top_k)
            contexts = [r.content_preview for r in steps.final_results]

            # Use ground truth as answer (reference-based evaluation)
            answer = gt_answer if gt_answer else query

            questions.append(query)
            answers.append(answer)
            retrieved_contexts.append(contexts)
            ground_truth.append(gt_answer)

            job["progress"] = i + 1

        # Run RAGAS
        scores = evaluator.evaluate(
            questions=questions,
            answers=answers,
            retrieved_contexts=retrieved_contexts,
            ground_truth=ground_truth if any(ground_truth) else None,
        )

        result = {
            "version": "live-kb",
            "created_at": datetime.now(UTC).isoformat(),
            "dataset": dataset_name,
            "total_queries": len(questions),
            "top_k": top_k,
            "scores": scores,
        }

        with open(RAGAS_RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        job["status"] = "done"
        job["finished_at"] = datetime.now(UTC).isoformat()

    except Exception as e:
        logger.exception("RAGAS evaluation job %s failed", job_id)
        job["status"] = "error"
        job["error"] = str(e)[:500]
        job["finished_at"] = datetime.now(UTC).isoformat()


@router.post("/ragas")
async def run_ragas_evaluation(
    dataset: str = Form("test_questions.json"),
    top_k: int = Form(5),
    sample_size: int = Form(10),
):
    """Start a RAGAS quality evaluation job."""
    dataset = dataset.strip()
    if not (DATA_DIR / dataset).exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset}")

    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "status": "running",
        "progress": 0,
        "total": 0,
        "dataset": dataset,
        "top_k": top_k,
        "sample_size": sample_size,
        "type": "ragas",
        "started_at": datetime.now(UTC).isoformat(),
        "finished_at": None,
        "error": "",
    }
    _evict_old_jobs()
    _JOBS[job_id] = job
    _JOBS[job_id]["_task"] = asyncio.create_task(
        _run_ragas_evaluation(job_id, dataset, top_k, sample_size)
    )
    return RedirectResponse(f"/benchmarks?job={job_id}", status_code=303)


@router.get("/ragas/result")
async def ragas_result():
    """Latest RAGAS evaluation results as JSON."""
    if not RAGAS_RESULTS_JSON.exists():
        raise HTTPException(status_code=404, detail="No RAGAS results yet")
    return FileResponse(RAGAS_RESULTS_JSON, media_type="application/json")


# ---------------------------------------------------------------------------
# Massive QA + IVA benchmarks (for server UI)
# ---------------------------------------------------------------------------

MASSIVE_RESULTS_JSON = DATA_DIR / "massive_results.json"
IVA_RESULTS_JSON = DATA_DIR / "iva_results.json"

async def _run_massive(job_id: str) -> None:
    """Run verbatim QA for every row in kb-source QA files, track progress."""
    from kb_manager.config import load_config
    from kb_manager.models.database import Database
    from sqlalchemy import text
    job = _JOBS[job_id]
    try:
        cfg = load_config()
        import os, pathlib
        # Robust: try new KB first, then fall back; also handle DB-driven if filesystem scan fails
        candidates = []
        env_src = os.getenv("KB_SOURCE_DIR", "")
        if env_src and pathlib.Path(env_src).exists():
            candidates.append(pathlib.Path(env_src))
        # Always try both known KBs in order: new 9.7, then 1405-05-31, then cfg.source_dir
        for cand in [pathlib.Path(r"D:/Code/KB/kb-source/KB_9.7.2026"), pathlib.Path(r"D:/Code/KB/kb-source/1405-05-31"), pathlib.Path(cfg.source_dir)]:
            if cand.exists() and cand not in candidates:
                candidates.append(cand)
        files = []
        for src in candidates:
            for p in src.rglob("*.xlsx"):
                if p.name.startswith("~$") or "TestQuestion" in str(p):
                    continue
                if any(s in p.stem for s in ["واژگان معادل", "محدودیت ها"]):
                    continue
                try:
                    from kb_manager.parsers.xlsx_parser import XlsxParser
                    parsed = XlsxParser().parse(str(p))
                    for sh in parsed.sheets:
                        if sh.get("schema") == "crm_qa":
                            sp = str(p.resolve())
                            if sp not in files:
                                files.append(sp)
                            break
                except Exception:
                    continue
            if files:
                break
        # Build total count for progress
        total = 0
        file_rows = []
        for f in files:
            from kb_manager.parsers.xlsx_parser import XlsxParser
            try:
                sh = [s for s in XlsxParser().parse(f).sheets if s.get("schema") == "crm_qa"][0]
                total += len(sh["rows"])
                file_rows.append((f, sh))
            except Exception:
                pass
        # Fallback: if still 0, use DB qa_pair chunks count (DB-driven, works for any KB)
        if total == 0:
            from kb_manager.models.database import Database
            from sqlalchemy import text as _text2
            tmp_db = Database(cfg.db)
            async with tmp_db.session() as s:
                r = await s.execute(_text2("SELECT count(*) FROM chunks WHERE chunk_type='qa_pair'"))
                total = r.scalar() or 0
            await tmp_db.close()
            job["total"] = total
            job["progress"] = 0
            # need to handle DB-driven mode below: set flag
            if total > 0:
                # will use DB-driven path: no file_rows, directly test DB chunks
                pass
        else:
            job["total"] = total
            job["progress"] = 0
        from kb_manager.web.routes.search import search_knowledge_base
        db = Database(cfg.db)
        # Preload document map to avoid per-query DB hits for doc lookup
        failed = []
        passed = 0
        done = 0
        for f, sh in file_rows:
            headers = [h.lower() for h in sh["headers"]]
            q_idx = headers.index("question") if "question" in headers else 0
            # Find doc id for this file (slash-normalized)
            qpath = str(pathlib.Path(f).resolve()).replace("\\", "/")
            async with db.session() as s:
                r = await s.execute(text("SELECT id, source_path FROM documents"))
                doc_id = None
                for row in r.fetchall():
                    if row[1].replace("\\", "/").lower() == qpath.lower():
                        doc_id = row[0]
                        break
                if not doc_id:
                    # Skip file not indexed (should not happen for 1405-05-31)
                    for _ in sh["rows"]:
                        done += 1
                        job["progress"] = done
                    continue
                r2 = await s.execute(text("SELECT id, content FROM chunks WHERE document_id = :d"), {"d": doc_id})
                chunks = {row[0]: row[1] for row in r2.fetchall()}
            for row in sh["rows"]:
                q = row[q_idx].strip() if q_idx < len(row) else ""
                if not q:
                    done += 1
                    job["progress"] = done
                    continue
                expected = None
                for cid, content in chunks.items():
                    if q[:30] in content:
                        expected = cid
                        break
                if not expected:
                    done += 1
                    job["progress"] = done
                    continue
                steps = await search_knowledge_base(q, top_k=5)
                retrieved = {r.chunk_id for r in steps.final_results}
                if expected in retrieved:
                    passed += 1
                else:
                    failed.append({"file": f, "question": q[:80], "expected": expected, "retrieved": [r.chunk_id for r in steps.final_results]})
                done += 1
                job["progress"] = done
                job["passed"] = passed
                job["failed"] = len(failed)
                job["hit_rate"] = passed/max(done,1)
                # Periodic save so Latest Result updates mid-run
                if done % 10 == 0:
                    with open(MASSIVE_RESULTS_JSON, "w", encoding="utf-8") as out:
                        import json as _js
                        _js.dump({"total": total, "passed": passed, "failed": len(failed), "hit_rate": passed/max(total,1), "failed_samples": failed[:20]}, out, ensure_ascii=False, indent=2)
        await db.close()
        job["status"] = "done"
        job["finished_at"] = datetime.now(UTC).isoformat()
        job["failed_samples"] = failed
        job["passed"] = passed
        job["failed"] = len(failed)
        job["hit_rate"] = passed/max(total,1)
        # Save result
        result = {"total": total, "passed": passed, "failed": len(failed), "hit_rate": passed/max(total,1), "failed_samples": failed}
        with open(MASSIVE_RESULTS_JSON, "w", encoding="utf-8") as out:
            import json
            json.dump(result, out, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception("Massive job %s failed", job_id)
        job["status"] = "error"
        job["error"] = str(e)[:500]
        job["finished_at"] = datetime.now(UTC).isoformat()


@router.get("/massive")
async def massive_page(request: Request):
    latest = None
    if MASSIVE_RESULTS_JSON.exists():
        try:
            import json
            latest = json.loads(MASSIVE_RESULTS_JSON.read_text(encoding="utf-8"))
        except Exception:
            latest = None
    iva_latest = None
    if IVA_RESULTS_JSON.exists():
        try:
            import json
            data = json.loads(IVA_RESULTS_JSON.read_text(encoding="utf-8"))
            # iva_results is list of per-query, compute summary
            hits = sum(1 for r in data if r.get("doc_hit"))
            iva_latest = {"doc_hits": hits, "total": len(data), "mrr": sum(1.0/r["doc_rank"] for r in data if r.get("doc_rank",-1)>0)/len(data) if data else 0, "failed": [str(r["i"]) for r in data if not r.get("doc_hit")]}
        except Exception:
            iva_latest = None
    # Check for running massive job
    job = None
    for j in _JOBS.values():
        if j.get("dataset") == "massive-qa":
            job = j
            break
    return templates.TemplateResponse(request, "massive_benchmark.html", {"result": latest, "iva_result": iva_latest, "job": job})


@router.post("/massive/run")
async def run_massive():
    job_id = uuid.uuid4().hex[:12]
    job = {"id": job_id, "status": "running", "progress": 0, "total": 1, "dataset": "massive-qa", "started_at": datetime.now(UTC).isoformat(), "finished_at": None, "error": "", "failed_samples": []}
    _evict_old_jobs()
    _JOBS[job_id] = job
    _JOBS[job_id]["_task"] = asyncio.create_task(_run_massive(job_id))
    return RedirectResponse(f"/benchmarks/massive?job={job_id}", status_code=303)


@router.get("/massive/result")
async def massive_result():
    if not MASSIVE_RESULTS_JSON.exists():
        raise HTTPException(status_code=404, detail="No massive results yet")
    return FileResponse(MASSIVE_RESULTS_JSON, media_type="application/json")


@router.post("/iva/run")
async def run_iva_route():
    # Reuse existing run_iva_eval logic as background job
    import subprocess, sys
    job_id = uuid.uuid4().hex[:12]
    job = {"id": job_id, "status": "running", "progress": 0, "total": 15, "dataset": "iva-15", "started_at": datetime.now(UTC).isoformat(), "finished_at": None, "error": ""}
    _JOBS[job_id] = job
    async def _run():
        try:
            proc = await asyncio.create_subprocess_exec(sys.executable, "run_iva_eval.py", cwd=str(PROJECT_ROOT))
            await proc.wait()
            job["status"] = "done"
            job["progress"] = 15
            job["finished_at"] = datetime.now(UTC).isoformat()
        except Exception as e:
            job["status"] = "error"
            job["error"] = str(e)
            job["finished_at"] = datetime.now(UTC).isoformat()
    _JOBS[job_id]["_task"] = asyncio.create_task(_run())
    return RedirectResponse(f"/benchmarks/massive?job={job_id}", status_code=303)


@router.get("/iva/result")
async def iva_result_json():
    if not IVA_RESULTS_JSON.exists():
        raise HTTPException(status_code=404, detail="No IVA results yet")
    return FileResponse(IVA_RESULTS_JSON, media_type="application/json")
