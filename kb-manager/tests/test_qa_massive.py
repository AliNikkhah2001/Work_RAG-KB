"""Massive QA test: verbatim retrieval must return correct chunk for every QA row in KB.

Scans all QA Excel files in kb-source/1405-05-31 (schema crm_qa), runs each question
verbatim through search_knowledge_base, asserts expected chunk in top-5.
Requirement: 100% recall (400+ questions).

Run: pytest tests/test_qa_massive.py -v --tb=short
Or: python -m pytest tests/test_qa_massive.py -v
"""
import asyncio
import pathlib
import pytest
from sqlalchemy import text

from kb_manager.config import load_config
from kb_manager.models.database import Database
from kb_manager.parsers.registry import get_parser

# Collect QA files at test collection time (fast, no DB)
def _collect_qa_files():
    cfg = load_config()
    source = pathlib.Path(cfg.source_dir)
    files = []
    for p in source.rglob("*.xlsx"):
        if p.name.startswith("~$") or "TestQuestion" in str(p):
            continue
        # quick check via parser schema detection
        try:
            from kb_manager.parsers.xlsx_parser import XlsxParser
            parser = XlsxParser()
            # Use detection without full parse for speed: just headers
            # Fallback to parsing first sheet headers
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
    """For each QA file, every question must retrieve its own chunk in top-5."""
    cfg = load_config()
    db = Database(cfg.db)
    # Build question -> expected chunk ID map for this file
    # Load chunks for this document
    from kb_manager.web.routes.search import search_knowledge_base

    # Find document ID for this file
    async with db.session() as s:
        r = await s.execute(text("SELECT id FROM documents WHERE source_path = :p"), {"p": str(pathlib.Path(qa_file).resolve())})
        doc = r.fetchone()
        if not doc:
            pytest.skip(f"Document not indexed: {qa_file}")
        doc_id = doc[0]
        r2 = await s.execute(text("SELECT id, content FROM chunks WHERE document_id = :d"), {"d": doc_id})
        chunks = {row[0]: row[1] for row in r2.fetchall()}

    # Parse file to get questions and map to chunk IDs via content matching
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
    # Map question text -> chunk ID via DB content search (question text appears in chunk)
    failures = []
    for row in qa_sheet["rows"]:
        q = row[q_idx].strip() if q_idx < len(row) else ""
        if not q:
            continue
        # Find expected chunk ID that contains this question (should be 1 row=1 chunk)
        expected = None
        for cid, content in chunks.items():
            if q[:30] in content:  # first 30 chars as anchor
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
