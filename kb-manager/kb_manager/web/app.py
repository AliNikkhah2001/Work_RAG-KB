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
    await db.create_tables()
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
    # Pre-warm search index (BM25 + dense embeddings + reranker) so the first
    # query doesn't block for 30+ seconds on model loading.
    try:
        from kb_manager.web.routes.search import _get_index
        import logging
        logging.getLogger(__name__).info("Pre-warming search index...")
        await _get_index()
        logging.getLogger(__name__).info("Search index ready.")
    except Exception as exc:
        logging.getLogger(__name__).warning("Search index pre-warm failed: %s", exc)
    yield
    await db.close()


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
    monitoring,
    pipeline,
    search,
    versions,
)  # noqa: E402

app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(chunks.router, prefix="/chunks", tags=["chunks"])
app.include_router(pipeline.router, prefix="/pipeline", tags=["pipeline"])
app.include_router(versions.router, prefix="/versions", tags=["versions"])
app.include_router(monitoring.router, prefix="/monitoring", tags=["monitoring"])
app.include_router(search.router, prefix="/search", tags=["search"])
app.include_router(benchmarks.router, prefix="/benchmarks", tags=["benchmarks"])

# Workaround for FastAPI 0.141+ include_router not working properly
# Manually add cleanup routes with /cleanup prefix
for route in cleanup.router.routes:
    if hasattr(route, "path"):
        # Create a copy of the route with prefixed path
        from fastapi.routing import APIRoute
        if isinstance(route, APIRoute):
            new_route = APIRoute(
                path="/cleanup" + route.path,
                endpoint=route.endpoint,
                methods=route.methods,
                response_class=route.response_class,
                name=route.name,
                tags=route.tags,
                summary=route.summary,
                description=route.description,
                response_model=route.response_model,
                status_code=route.status_code,
                dependencies=route.dependencies,
                callbacks=route.callbacks,
                openapi_extra=route.openapi_extra,
                include_in_schema=route.include_in_schema,
            )
            app.router.routes.append(new_route)


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
