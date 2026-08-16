"""FastAPI web application for KB Manager."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from kb_manager.config import load_config
from kb_manager.models.database import Database

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

config = load_config()
db = Database(config.db)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create tables on startup."""
    await db.create_tables()
    yield
    await db.close()


app = FastAPI(
    title="KB Manager",
    description="Knowledge Base Management System for ICS Credit Scoring",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

from kb_manager.web.routes import chunks, documents, monitoring, pipeline, search, versions  # noqa: E402

app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(chunks.router, prefix="/chunks", tags=["chunks"])
app.include_router(pipeline.router, prefix="/pipeline", tags=["pipeline"])
app.include_router(versions.router, prefix="/versions", tags=["versions"])
app.include_router(monitoring.router, prefix="/monitoring", tags=["monitoring"])
app.include_router(search.router, prefix="/search", tags=["search"])


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
