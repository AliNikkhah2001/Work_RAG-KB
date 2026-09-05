"""Transparency / pipeline introspection routes.

Steps shown:
  1. File on disk -> sheets via XlsxParser
  2. Header normalization + schema detection
  3. Row -> fields -> chunk mapping (Persian labels)
  4. DB chunks derived from that file
All rendered with explicit UTF-8 + Persian-safe font/dir so that
Persian text is verifiably correct end-to-end.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy import func, select

from kb_manager.models.database import Chunk, Document
from kb_manager.web.deps import db, templates

router = APIRouter()

# Keep in sync with xlsx_parser.py
SCHEMA_DEFS: dict[str, set[str]] = {
    "reason_codes": {
        "reason_code", "model_name", "model_id", "brief_explanation",
        "detailed_explanation", "reason_text", "improvement_suggestions",
        "bin_score", "bin_impact", "feature_score", "feature_impact",
        "bin_details", "keywords", "feature_name", "data_source",
    },
    "crm_qa": {"question", "model", "briefanswer", "answer", "keyword"},
    "articles": {
        "documentname", "title", "sectiontitle", "content", "type",
        "version", "author(s)", "heading", "keywords", "summary",
    },
}

FIELD_FA: dict[str, str] = {
    "question": "سوال",
    "briefanswer": "پاسخ کوتاه",
    "brief_answer": "پاسخ کوتاه",
    "answer": "پاسخ کامل",
    "keyword": "کلیدواژه‌ها",
    "keywords": "کلیدواژه‌ها",
    "model": "مدل",
    "reason_code": "کد دلیل",
    "reason_text": "متن دلیل",
    "brief_explanation": "توضیح کوتاه",
    "detailed_explanation": "توضیح کامل",
    "feature_name": "نام ویژگی",
    "feature_score": "امتیاز ویژگی",
    "feature_impact": "اثر ویژگی",
    "bin_score": "امتیاز بازه",
    "bin_impact": "اثر بازه",
    "improvement_suggestions": "پیشنهاد بهبود",
    "documentname": "نام سند",
    "title": "عنوان",
    "sectiontitle": "عنوان بخش",
    "content": "محتوا",
    "heading": "سرفصل",
    "summary": "خلاصه",
    "type": "نوع",
    "version": "نسخه",
    "author(s)": "نویسنده",
}


def _normalize_col(name: str) -> str:
    return re.sub(r"[\s_\-]+", "", name.strip().lower())


def _detect_schema_debug(headers: list[str]) -> tuple[str | None, dict[str, Any]]:
    """Return (schema, debug_info) with per-schema overlap counts."""
    normalized = {_normalize_col(h) for h in headers if h}
    debug: dict[str, Any] = {"normalized": sorted(normalized), "candidates": []}
    chosen: str | None = None
    for sname, required in SCHEMA_DEFS.items():
        norm_req = {_normalize_col(c) for c in required}
        overlap = len(normalized & norm_req)
        needed = len(norm_req) * 0.6
        matched_cols = sorted(normalized & norm_req)
        missing = sorted(norm_req - normalized)
        entry = {
            "schema": sname,
            "required_count": len(norm_req),
            "overlap": overlap,
            "threshold": round(needed, 1),
            "matched": matched_cols,
            "missing": missing,
            "would_match": overlap >= needed,
        }
        debug["candidates"].append(entry)
        if chosen is None and overlap >= needed:
            chosen = sname
    return chosen, debug


def _parse_file_for_display(source_path: str) -> dict[str, Any]:
    """Parse file via XlsxParser (or fallback) and return display dict."""
    p = Path(source_path)
    result: dict[str, Any] = {
        "exists": p.exists(),
        "path": str(p),
        "suffix": p.suffix.lower(),
        "size_bytes": p.stat().st_size if p.exists() else 0,
        "error": None,
        "parser": None,
        "sheets": [],
        "combined_preview": "",
        "integrity_warnings": [],
    }
    if not p.exists():
        result["error"] = "File not found on disk (document indexed earlier, source removed or moved)"
        return result
    # choose parser
    try:
        from kb_manager.parsers.registry import get_parser  # lazy import

        parser = get_parser(str(p))
        result["parser"] = parser.__class__.__name__
        parsed = parser.parse(str(p))
        result["sheets"] = parsed.sheets or []
        result["combined_preview"] = parsed.content[:8000]
        result["integrity_warnings"] = (parsed.metadata or {}).get("integrity_warnings", []) or \
            (parsed.metadata or {}).get("integrity_issues", []) or []
        # attach per-sheet schema debug
        for sh in result["sheets"]:
            hdrs: list[str] = sh.get("headers", [])
            schema, dbg = _detect_schema_debug(hdrs)
            sh["_schema_debug"] = dbg
            # header detail rows
            hdr_detail = []
            for h in hdrs:
                norm = _normalize_col(h)
                # which schemas claim this header?
                claims = []
                for sname, required in SCHEMA_DEFS.items():
                    if norm in {_normalize_col(c) for c in required}:
                        claims.append(sname)
                hdr_detail.append({"original": h, "normalized": norm, "claims": claims, "fa": FIELD_FA.get(h.strip().lower(), "")})
            sh["_header_detail"] = hdr_detail
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


@router.get("", response_class=HTMLResponse)
async def transparency_index(request: Request, q: str = "", page: int = 1, per_page: int = 20):
    """List documents with transparency inspect links."""
    async with db.session() as session:
        query = select(Document).order_by(Document.updated_at.desc())
        count_q = select(func.count(Document.id))
        if q:
            like = f"%{q}%"
            query = query.where(Document.title.ilike(like) | Document.source_path.ilike(like))
            count_q = count_q.where(Document.title.ilike(like) | Document.source_path.ilike(like))
        total = (await session.execute(count_q)).scalar() or 0
        total_pages = max(1, (total + per_page - 1) // per_page)
        offset = (page - 1) * per_page
        docs = (await session.execute(query.offset(offset).limit(per_page))).scalars().all()
    return templates.TemplateResponse(request, "transparency.html", {
        "documents": docs,
        "q": q,
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": total_pages,
        "field_fa": FIELD_FA,
        "schema_defs": SCHEMA_DEFS,
    })


@router.post("/parse-upload", response_class=HTMLResponse)
async def transparency_parse_upload(request: Request, file: UploadFile = File(...), rows: int = 30):
    """Parse an uploaded Excel without indexing — live transparency preview."""
    suffix = Path(file.filename or "upload.xlsx").suffix.lower()
    if suffix != ".xlsx":
        return templates.TemplateResponse(request, "transparency_upload.html", {
            "error": f"Only .xlsx supported, got {suffix}",
            "filename": file.filename,
        }, status_code=400)
    # save to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        parse_info = _parse_file_for_display(tmp_path)
        parse_info["path"] = file.filename or tmp_path
        parse_info["size_bytes"] = len(content)
        sheets = parse_info.get("sheets", [])
        active_sheet = sheets[0] if sheets else None
        return templates.TemplateResponse(request, "transparency_upload.html", {
            "filename": file.filename,
            "parse_info": parse_info,
            "sheets": sheets,
            "active_sheet": active_sheet,
            "field_fa": FIELD_FA,
            "schema_defs": SCHEMA_DEFS,
            "rows_limit": max(5, min(rows, 200)),
            "error": parse_info.get("error"),
        })
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@router.get("/api/raw/{doc_id}")
async def transparency_api_raw(doc_id: str):
    """JSON dump of parse_info + chunks for programmatic inspection (UTF-8)."""
    async with db.session() as session:
        doc = await session.get(Document, doc_id)
        if not doc:
            return JSONResponse({"error": "not found"}, status_code=404)
        chunks = (await session.execute(
            select(Chunk).where(Chunk.document_id == doc_id).order_by(Chunk.ordinal)
        )).scalars().all()
    parse_info = _parse_file_for_display(doc.source_path)
    # ensure JSON is UTF-8 (FastAPI does ensure_ascii=False by default via jsonable_encoder)
    return JSONResponse({
        "document": {"id": doc.id, "title": doc.title, "source_path": doc.source_path, "chunk_count": doc.chunk_count, "metadata": doc.doc_metadata},
        "parse": parse_info,
        "chunks": [{"ordinal": c.ordinal, "type": c.chunk_type, "heading": c.heading_path, "tokens": c.token_count, "content": c.content, "metadata": c.doc_metadata} for c in chunks],
    })


@router.get("/{doc_id}", response_class=HTMLResponse)
async def transparency_detail(request: Request, doc_id: str, rows: int = 20, sheet: str = ""):
    """Step-by-step view for a single document: raw table -> parsed -> chunks."""
    async with db.session() as session:
        doc: Document | None = await session.get(Document, doc_id)
        if not doc:
            return RedirectResponse("/transparency", status_code=302)
        chunks = (await session.execute(
            select(Chunk).where(Chunk.document_id == doc_id).order_by(Chunk.ordinal)
        )).scalars().all()

    parse_info = _parse_file_for_display(doc.source_path)

    sheets = parse_info.get("sheets", [])
    active_sheet = None
    if sheet and sheets:
        for sh in sheets:
            if sh.get("name") == sheet:
                active_sheet = sh
                break
    if active_sheet is None and sheets:
        active_sheet = sheets[0]

    return templates.TemplateResponse(request, "transparency_detail.html", {
        "document": doc,
        "chunks": chunks,
        "parse_info": parse_info,
        "sheets": sheets,
        "active_sheet": active_sheet,
        "active_sheet_name": active_sheet.get("name", "") if active_sheet else "",
        "rows_limit": max(5, min(rows, 200)),
        "field_fa": FIELD_FA,
        "schema_defs": SCHEMA_DEFS,
    })
