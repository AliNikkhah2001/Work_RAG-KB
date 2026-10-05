from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from kb_manager.settings import load_settings, save_settings, SearchSettings
from pathlib import Path

router = APIRouter(tags=["settings"])
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))

@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def get_settings(request: Request):
    settings = load_settings()
    return templates.TemplateResponse(request, "settings.html", {"settings": settings})

@router.post("")
@router.post("/")
async def update_settings(
    request: Request,
    top_k: int = Form(...),
    min_relevance_score: float = Form(...),
    keyword_boost: float = Form(...),
    fusion_alpha: float = Form(...),
    synonym_enabled: bool = Form(False),
    synonym_beam: int = Form(...),
    rerank_pool: int = Form(...)
):
    settings = SearchSettings(
        top_k=top_k,
        min_relevance_score=min_relevance_score,
        keyword_boost=keyword_boost,
        fusion_alpha=fusion_alpha,
        synonym_enabled=synonym_enabled,
        synonym_beam=synonym_beam,
        rerank_pool=rerank_pool
    )
    save_settings(settings)
    return RedirectResponse(url="/settings?success=1", status_code=303)
