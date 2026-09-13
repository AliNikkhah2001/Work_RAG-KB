"""Massive QA test: verbatim retrieval must return correct chunk for every QA row in KB.

Scans all QA Excel files in kb-source/1405-05-31 (schema crm_qa), runs each question
verbatim through search_knowledge_base, asserts expected chunk in top-5.
Requirement: 100% recall (400+ questions).

Run: pytest tests/test_qa_massive.py -v --tb=short
Or: python -m pytest tests/test_qa_massive.py -v
"""
import asyncio
import json
import pathlib
import pytest
from sqlalchemy import text

from kb_manager.config import load_config
from kb_manager.models.database import Database
from kb_manager.parsers.registry import get_parser

# Collect QA files at test collection time (fast, no DB)
def _collect_qa_files():
    # Use the same source as the live DB (1405-05-31) to avoid testing stale clean_files
    import os
    # Prefer explicit KB_SOURCE_DIR env (as used by DB), fallback to 1405-05-31
    env_src = os.getenv("KB_SOURCE_DIR", "")
    if env_src and pathlib.Path(env_src).exists():
        source = pathlib.Path(env_src)
    else:
        cfg = load_config()
        # Force to 1405-05-31 if cfg points to parent kb-source (which includes clean_files)
        cand = pathlib.Path(cfg.source_dir)
        if cand.name == "kb-source":
            cand = cand / "1405-05-31"
        source = cand if cand.exists() else pathlib.Path(cfg.source_dir)
    files = []
    for p in source.rglob("*.xlsx"):
        if p.name.startswith("~$") or "TestQuestion" in str(p):
            continue
        # Skip preprocessing/guardrail files not in KB
        if any(s in p.stem for s in ["واژگان معادل", "محدودیت ها"]):
            continue
        try:
            from kb_manager.parsers.xlsx_parser import XlsxParser
            parser = XlsxParser()
            parsed = parser.parse(str(p))
            for sheet in parsed.sheets:
                if sheet.get("schema") == "crm_qa":
                    files.append(str(p))
                    break
        except Exception:
            continue
    return sorted(files)

QA_FILES = _collect_qa_files()

@pytest.mark.asyncio
@pytest.mark.parametrize("qa_file", QA_FILES)
async def test_qa_file_verbatim_recall(qa_file):
    """For each QA file, every question must retrieve its own child chunk in top-5."""
    cfg = load_config()
    db = Database(cfg.db)
    # Build question -> expected child chunk ID map for this file
    # Load child chunks (qa_pair) for this document, excluding qa_pair_parent
    from kb_manager.web.routes.search import search_knowledge_base

    # Find document ID for this file (handle slash/case differences between Windows and DB)
    qpath = str(pathlib.Path(qa_file).resolve())
    # Normalize to forward slashes for comparison (DB stores with forward slashes on some builds)
    qpath_norm = qpath.replace("\\", "/")
    async with db.session() as s:
        # Try exact match first, then normalized, then LIKE
        r = await s.execute(text("SELECT id, source_path FROM documents"))
        doc_id = None
        for row in r.fetchall():
            db_path = row[1]
            if db_path == qpath or db_path.replace("\\", "/") == qpath_norm or db_path.replace("\\", "/").lower() == qpath_norm.lower():
                doc_id = row[0]
                break
        if not doc_id:
            pytest.skip(f"Document not indexed: {qa_file}")
        r2 = await s.execute(
            text("SELECT id, content, chunk_type, metadata FROM chunks WHERE document_id = :d"),
            {"d": doc_id},
        )
        children = []
        for row in r2.fetchall():
            cid, content, ctype, meta = row[0], row[1], row[2], row[3]
            if ctype != "qa_pair":
                continue
            if isinstance(meta, str):
                meta = json.loads(meta)
            fields = meta.get("fields", {}) if isinstance(meta, dict) else {}
            meta_q = (
                fields.get("question")
                or fields.get("پرسش")
                or fields.get("سوال")
                or fields.get("متن سوال")
                or fields.get("متن_سوال")
                or ""
            )
            children.append((cid, content, meta_q.strip()))

    # Parse file to get questions and map to child chunk IDs via metadata/content matching
    from kb_manager.parsers.xlsx_parser import XlsxParser
    parser = XlsxParser()
    parsed = parser.parse(qa_file)
    # Find QA sheet
    qa_sheet = None
    for sh in parsed.sheets:
        if sh.get("schema") == "crm_qa":
            qa_sheet = sh
            break
    if not qa_sheet:
        pytest.skip("No QA sheet")

    headers = [h.lower() for h in qa_sheet["headers"]]
    q_idx = headers.index("question") if "question" in headers else 0
    # Map question text -> child chunk ID via metadata question field, then content match
    by_question = {}
    for cid, content, meta_q in children:
        if meta_q:
            by_question.setdefault(meta_q, cid)
    failures = []
    for row in qa_sheet["rows"]:
        q = row[q_idx].strip() if q_idx < len(row) else ""
        if not q:
            continue
        # 1 row = 1 child chunk. Find expected child chunk ID.
        expected = by_question.get(q)
        if not expected:
            for cid, content, meta_q in children:
                if content.startswith(f"سوال: {q}") or q[:30] in content:
                    expected = cid
                    break
        if not expected:
            failures.append((q[:40], "no chunk found"))
            continue
        steps = await search_knowledge_base(q, top_k=5)
        retrieved = {r.chunk_id for r in steps.final_results}
        if expected not in retrieved:
            failures.append((q[:60], f"expected {expected[:8]} not in {[c[:8] for c in retrieved]}"))

    await db.close()
    assert not failures, f"{len(failures)} verbatim misses in {qa_file}: {failures[:3]}"

def test_qa_massive_count():
    """Sanity: ensure we have at least 300 QA rows total."""
    assert len(QA_FILES) >= 5, f"Expected >=5 QA files, got {len(QA_FILES)}"
    total_rows = 0
    for f in QA_FILES:
        from kb_manager.parsers.xlsx_parser import XlsxParser
        try:
            p = XlsxParser().parse(f)
            for sh in p.sheets:
                if sh.get("schema") == "crm_qa":
                    total_rows += len(sh["rows"])
        except Exception:
            pass
    assert total_rows >= 300, f"Expected >=300 QA rows, got {total_rows}"
