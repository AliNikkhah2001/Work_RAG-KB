"""FastAPI web application for KB Manager."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from kb_manager.web.deps import db, templates

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create tables on startup and pre-warm search index."""
    import logging
    log = logging.getLogger(__name__)

    await db.create_tables()
    # Mark stale jobs from previous process runs as interrupted
    try:
        import datetime as _dt

        from sqlalchemy import update

        from kb_manager.models.database import IngestionJob

        async with db.session() as s:
            await s.execute(
                update(IngestionJob)
                .where(IngestionJob.status.in_(["running", "pending"]))
                .values(status="interrupted", completed_at=_dt.datetime.now(_dt.UTC))
            )
    except Exception:
        pass
    # Ensure indexes exist on existing tables (create_all only creates missing tables)
    try:
        async with db.async_engine.begin() as conn:
            await conn.execute(
                __import__("sqlalchemy").text(
                    "CREATE INDEX IF NOT EXISTS ix_chunks_created_at ON chunks (created_at)"
                )
            )
    except Exception:
        pass

    # Pre-warm search index synchronously so the server is ready immediately.
    # This loads BM25 index + MiniLM embeddings + cross-encoder reranker (~30s first time).
    app.state.reranker_ok = False
    app.state.reranker_error = None
    app.state.dense_ok = False
    log.info("Pre-warming search index (this may take a moment)...")
    try:
        await _prewarm_search_index(app)
        log.info("Search index ready (reranker_ok=%s).", app.state.reranker_ok)
    except Exception as exc:
        app.state.reranker_ok = False
        app.state.reranker_error = str(exc)[:500]
        log.exception(
            "Search index pre-warm FAILED — retrieval will error until fixed: %s", exc
        )

    yield
    await db.close()


async def _prewarm_search_index(app: FastAPI | None = None):
    """Load BM25 index + dense embeddings + cross-encoder reranker.

    Also runs the dense/reranker warmup encodes so the GPU kernels and CUDA
    context are live before the first user query (otherwise the first search
    pays the lazy model load, ~7-9 s one-time).

    Reranker warmup failures are NOT swallowed — they set ``app.state.reranker_ok``
    False and re-raise so /ready reports not_ready.
    """
    import logging

    from kb_manager.web.routes.search import _get_index

    chunk_data, bm25_pair, dense, reranker, hyde = await _get_index()
    if not chunk_data:
        log.error(
            "Search index is EMPTY (0 chunks); retrieval will return nothing. "
            "Check KB_DB_URL / KB_DB_MODE (sqlite fallback is likely pointing at an empty DB)."
        )
        return
    dense_warmer = getattr(dense, "warmup", None)
    if callable(dense_warmer):
        dense_warmer()
        if app is not None:
            app.state.dense_ok = True
        logging.getLogger(__name__).info("dense model warmed up")
    rerank_warmer = getattr(reranker, "warmup", None)
    if not callable(rerank_warmer):
        # Force load even if older reranker lacked warmup()
        ensure = getattr(reranker, "_ensure_model", None)
        if callable(ensure):
            ensure()
            if app is not None:
                app.state.reranker_ok = True
            logging.getLogger(__name__).info("reranker model loaded via _ensure_model")
            return
        raise RuntimeError(
            f"Reranker {type(reranker).__name__} has neither warmup() nor _ensure_model()"
        )
    try:
        rerank_warmer()
        if app is not None:
            app.state.reranker_ok = True
            app.state.reranker_error = None
        logging.getLogger(__name__).info("reranker model warmed up")
    except Exception as exc:
        if app is not None:
            app.state.reranker_ok = False
            app.state.reranker_error = str(exc)[:500]
        raise


app = FastAPI(
    title="KB Manager",
    description="Knowledge Base Management System for ICS Credit Scoring",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="static")

from kb_manager.web.routes import (
    benchmarks,
    chunks,
    cleanup,
    documents,
    ingestion_suite,
    kb_history,
    metrics,
    monitoring,
    pipeline,
    registry,
    search,
    transparency,
    versions,
    zip_browser,
    settings,
)  # noqa: E402

# /metrics (no prefix) — Prometheus already scrapes this path.
try:
    app.include_router(metrics.router)
except Exception:  # noqa: BLE001 - observability must never block startup
    pass

app.include_router(settings.router, prefix="/settings", tags=["settings"])
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(chunks.router, prefix="/chunks", tags=["chunks"])
app.include_router(pipeline.router, prefix="/pipeline", tags=["pipeline"])
app.include_router(versions.router, prefix="/versions", tags=["versions"])
app.include_router(monitoring.router, prefix="/monitoring", tags=["monitoring"])
app.include_router(search.router, prefix="/search", tags=["search"])
app.include_router(benchmarks.router, prefix="/benchmarks", tags=["benchmarks"])
app.include_router(cleanup.router, prefix="/cleanup", tags=["cleanup"])
app.include_router(transparency.router, prefix="/transparency", tags=["transparency"])
app.include_router(zip_browser.router, prefix="/transparency", tags=["transparency-zip"])
app.include_router(ingestion_suite.router, prefix="/ingestion", tags=["ingestion"])
app.include_router(kb_history.router, prefix="/ingestion", tags=["ingestion-kbs"])
app.include_router(registry.router, prefix="/registry", tags=["registry"])


@app.get("/health")
async def health(request: Request):
    """Health check — reports DB mode, connectivity, and reranker status."""
    from kb_manager.config import load_config
    from kb_manager.reranker import get_reranker_model_name

    cfg = load_config()
    db_ok = False
    doc_count = chunk_count = None
    error = None
    try:
        from sqlalchemy import func, select, text

        from kb_manager.models.database import Chunk, Document

        async with db.session() as session:
            # lightweight ping
            await session.execute(text("SELECT 1"))
            doc_count = (await session.execute(select(func.count(Document.id)))).scalar()
            chunk_count = (await session.execute(select(func.count(Chunk.id)))).scalar()
        db_ok = True
    except Exception as exc:
        error = str(exc)[:500]

    reranker_ok = bool(getattr(request.app.state, "reranker_ok", False))
    reranker_error = getattr(request.app.state, "reranker_error", None)
    status = "ok" if db_ok and (chunk_count or 0) == 0 else (
        "ok" if db_ok and reranker_ok else ("degraded" if db_ok else "error")
    )
    # Empty DB is ok for bootstrapping; once chunks exist, reranker must work.
    if db_ok and (chunk_count or 0) > 0 and not reranker_ok:
        status = "degraded"

    return {
        "status": status,
        "db_mode": cfg.db.mode,
        "db_driver": cfg.db.driver,
        "db_url": cfg.db.async_url.split("@")[-1] if "@" in cfg.db.async_url else cfg.db.async_url,
        "db_ok": db_ok,
        "doc_count": doc_count,
        "chunk_count": chunk_count,
        "reranker_ok": reranker_ok,
        "reranker_model": get_reranker_model_name(),
        "reranker_error": reranker_error,
        "dense_ok": bool(getattr(request.app.state, "dense_ok", False)),
        "error": error,
    }


@app.get("/api/health")
async def api_health():
    """Alias for /health."""
    return await health()

@app.get("/ready")
async def ready(request: Request):
    """Ready if DB has chunks, index built, and reranker loaded."""
    try:
        from sqlalchemy import func, select
        from kb_manager.models.database import Chunk
        from kb_manager.web.deps import db
        async with db.session() as session:
            cnt = (await session.execute(select(func.count(Chunk.id)))).scalar() or 0
        reranker_ok = bool(getattr(request.app.state, "reranker_ok", False))
        reranker_error = getattr(request.app.state, "reranker_error", None)
        ready_ok = cnt > 0 and reranker_ok
        out = {
            "status": "ready" if ready_ok else "not_ready",
            "chunks": cnt,
            "reranker_ok": reranker_ok,
        }
        if reranker_error:
            out["reranker_error"] = reranker_error
        if cnt > 0 and not reranker_ok:
            out["error"] = (
                "reranker not loaded — search will fail until the cross-encoder "
                "model is available in HF cache (see reranker_error)"
            )
        return out
    except Exception as e:
        return {"status": "not_ready", "error": str(e)}

@app.get("/")
async def dashboard(request: Request):
    """Main dashboard."""
    from sqlalchemy import func, select

    from kb_manager.models.database import Chunk, Document

    async with db.session() as session:
        doc_count = (await session.execute(select(func.count(Document.id)))).scalar() or 0
        chunk_count = (await session.execute(select(func.count(Chunk.id)))).scalar() or 0

        domain_counts = {}
        result = await session.execute(
            select(Document.domain, func.count(Document.id)).group_by(Document.domain)
        )
        for row in result.fetchall():
            domain_counts[row[0]] = row[1]

        category_counts = {}
        result = await session.execute(
            select(Document.category, func.count(Document.id)).group_by(Document.category)
        )
        for row in result.fetchall():
            category_counts[row[0]] = row[1]

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "doc_count": doc_count,
            "chunk_count": chunk_count,
            "domain_counts": domain_counts,
            "category_counts": category_counts,
        },
    )
