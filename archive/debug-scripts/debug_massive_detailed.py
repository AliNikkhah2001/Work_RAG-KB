"""Massive QA case-by-case debug for new KB."""
import os
os.environ['KB_DB_URL']='sqlite+aiosqlite:///D:/Code/KB/kb-manager/data/kb_9_7_2026.db'
os.environ['KB_SOURCE_DIR']=r'D:/Code/KB/kb-source/KB_9.7.2026'
import asyncio, pathlib
import sys
sys.stdout.reconfigure(encoding='utf-8')


def rank_of(results, gt_id):
    for i, r in enumerate(results):
        if getattr(r, "chunk_id", None) == gt_id:
            return i + 1
    return None


def fmt_rank(rank, n):
    return str(rank) if rank is not None else f">{n}"


async def main():
    from kb_manager.config import load_config
    from kb_manager.models.database import Database
    from sqlalchemy import text
    from kb_manager.parsers.xlsx_parser import XlsxParser
    from kb_manager.web.routes.search import search_knowledge_base

    cfg = load_config()
    src = pathlib.Path(cfg.source_dir)
    if src.name == "kb-source":
        src = src / "KB_9.7.2026"
    files = []
    for p in src.rglob("*.xlsx"):
        if p.name.startswith("~$") or "TestQuestion" in str(p):
            continue
        if any(s in p.stem for s in ["واژگان معادل", "محدودیت ها"]):
            continue
        try:
            parsed = XlsxParser().parse(str(p))
            for sh in parsed.sheets:
                if sh.get("schema") == "crm_qa":
                    files.append(str(p))
                    break
        except: continue

    html = ["""<html><head><meta charset='utf-8'><link href='https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css' rel='stylesheet'><style>body{background:#0f172a;color:#f1f5f9;font-family:Vazirmatn;padding:16px} .mono{font-family:Consolas,monospace;font-size:11px} table{border-collapse:collapse;width:100%} td,th{border:1px solid #334155;padding:6px;font-size:12px} th{background:#1e293b} .hit{background:rgba(34,197,94,0.1)} .miss{background:rgba(239,68,68,0.1)}</style></head><body>"""]
    html.append(f"<h1>Massive QA — {len(files)} files</h1>")

    db = Database(cfg.db)
    total = passed = 0
    failed_samples = []
    # Preload for progress
    for f in files:
        parser = XlsxParser()
        try:
            sh = [s for s in parser.parse(f).sheets if s.get("schema")=="crm_qa"][0]
        except: continue
        headers = [h.lower() for h in sh["headers"]]
        q_idx = headers.index("question") if "question" in headers else 0
        # Find doc
        qpath = str(pathlib.Path(f).resolve()).replace("\\","/")
        async with db.session() as s:
            r = await s.execute(text("SELECT id, source_path FROM documents"))
            doc_id = None
            for row in r.fetchall():
                if row[1].replace("\\","/").lower() == qpath.lower():
                    doc_id = row[0]
                    break
            if not doc_id:
                continue
            r2 = await s.execute(text("SELECT id, content FROM chunks WHERE document_id=:d"), {"d": doc_id})
            chunks = {row[0]: row[1] for row in r2.fetchall()}
        html.append(f"<h2 class='mono'>{f} — {len(sh['rows'])} rows</h2><table><tr><th>#</th><th>Question</th><th>Result</th><th>BM25</th><th>Dense</th><th>Merged</th><th>Final</th><th>Top-1 Preview</th></tr>")
        for idx, row in enumerate(sh["rows"]):
            q = row[q_idx].strip() if q_idx < len(row) else ""
            if not q:
                continue
            total += 1
            expected = None
            for cid, content in chunks.items():
                if q[:30] in content:
                    expected = cid
                    break
            if not expected:
                html.append(f"<tr class='miss'><td>{idx+1}</td><td class='p' dir='auto'>{q[:60]}</td><td>NO CHUNK</td><td></td><td></td><td></td><td></td><td></td></tr>")
                continue
            steps = await search_knowledge_base(q, top_k=100)
            retrieved = {r.chunk_id for r in steps.final_results[:5]}
            is_hit = expected in retrieved
            bm25_rank = rank_of(steps.bm25_results, expected)
            dense_rank = rank_of(steps.dense_results, expected)
            merged_rank = rank_of(steps.merged_candidates, expected)
            final_rank = rank_of(steps.final_results, expected)
            ranks_td = f"<td class='mono'>{fmt_rank(bm25_rank, len(steps.bm25_results))}</td><td class='mono'>{fmt_rank(dense_rank, len(steps.dense_results))}</td><td class='mono'>{fmt_rank(merged_rank, len(steps.merged_candidates))}</td><td class='mono'>{fmt_rank(final_rank, len(steps.final_results))}</td>"
            if is_hit:
                passed += 1
                html.append(f"<tr class='hit'><td>{idx+1}</td><td class='p' dir='auto'>{q[:60]}</td><td>HIT</td>{ranks_td}<td class='mono'>{steps.final_results[0].content_preview[:80] if steps.final_results else ''}</td></tr>")
            else:
                failed_samples.append({"file": f, "question": q[:80], "expected": expected[:8], "retrieved": [r.chunk_id[:8] for r in steps.final_results[:5]], "bm25_rank": bm25_rank, "dense_rank": dense_rank, "merged_rank": merged_rank, "final_rank": final_rank})
                html.append(f"<tr class='miss'><td>{idx+1}</td><td class='p' dir='auto'>{q[:60]}</td><td>MISS</td>{ranks_td}<td class='mono'>{steps.final_results[0].content_preview[:80] if steps.final_results else 'no result'}</td></tr>")
        html.append("</table>")

    await db.close()
    html.append(f"<h2>Summary: {passed}/{total} hit ({passed/max(total,1)*100:.1f}%) — {len(failed_samples)} failed</h2>")
    html.append("</body></html>")
    pathlib.Path("data/massive_detailed.html").write_text("\n".join(html), encoding="utf-8")
    print(f"Wrote data/massive_detailed.html total {total} passed {passed} failed {len(failed_samples)}")

if __name__ == "__main__":
    asyncio.run(main())
