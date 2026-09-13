"""Debug IVA: side-by-side expected vs retrieved top-5 with transparency links."""
import asyncio, json, pathlib, unicodedata, os
os.environ["KB_DB_URL"] = "sqlite+aiosqlite:///D:/Code/KB/kb-manager/data/kb_1405.db"
import sys
sys.stdout.reconfigure(encoding="utf-8")

def norm(s): 
    s=unicodedata.normalize("NFC",s).replace("\u200c"," ").replace("\u064a","\u06cc").replace("\u0643","\u06a9")
    return "".join(c for c in s if c not in "؟?،;,.!").strip().lower()

async def main():
    from kb_manager.web.routes.search import search_knowledge_base
    data = json.loads(pathlib.Path("data/test_questions_iva.json").read_text(encoding="utf-8"))
    # also load BEFORE if exists
    before = None
    if pathlib.Path("data/iva_results_BEFORE.json").exists():
        before = json.loads(pathlib.Path("data/iva_results_BEFORE.json").read_text(encoding="utf-8"))

    html = ["<html><head><meta charset='utf-8'><link href='https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css' rel='stylesheet'><style>.p{font-family:Vazirmatn,Tahoma} table{border-collapse:collapse;width:100%} td,th{border:1px solid #334155;padding:6px;font-size:13px} th{background:#1e293b;color:#f1f5f9}</style></head><body style='background:#0f172a;color:#f1f5f9;padding:16px'>"]
    html.append("<h2>IVA Side-by-Side: Expected vs Retrieved (fixed DB)</h2><p>Click doc link → Transparency raw table & chunks</p>")
    html.append("<table><tr><th>#</th><th>Query</th><th>Gold Answer (first 120)</th><th>BEFORE doc_hit</th><th>AFTER doc_hit + Top-5 Retrieved (preview + link)</th></tr>")
    for i, item in enumerate(data, 1):
        q = item["query"]
        exp_ans = item["expected_answer"][:120]
        exp_ids = set(item["expected_chunk_ids"])
        steps = await search_knowledge_base(q, 5)
        hits = [(r.chunk_id, r.doc_id, r.content_preview[:180], r.hybrid_score) for r in steps.final_results]
        doc_hit = any(cid in exp_ids for cid,_ ,_,_ in hits)
        before_hit = ""
        if before and i-1 < len(before):
            before_hit = "✅" if before[i-1].get("doc_hit") else "❌"
        row = f"<tr><td>{i}</td><td class='p' dir='auto'>{q[:80]}</td><td class='p' dir='auto'>{exp_ans}</td><td style='text-align:center'>{before_hit}</td><td>"
        row += f"<b>{'✅ HIT' if doc_hit else '❌ MISS'}</b> (rank {next((j+1 for j,(cid,_,_,_) in enumerate(hits) if cid in exp_ids), -1)})<br>"
        for cid, did, preview, score in hits:
            is_gold = "⭐" if cid in exp_ids else ""
            row += f"<div style='margin:4px 0;padding:4px;background:#1e293b;border-radius:4px'>{is_gold} <a href='/transparency/{did}' target='_blank' style='color:#60a5fa'>{did[:8]}</a> score {score:.3f}<br><span class='p' dir='auto'>{preview}</span></div>"
        row += "</td></tr>"
        html.append(row)
        print(f"{i:2d} {'HIT' if doc_hit else 'MISS'} | {q[:60]}")
    html.append("</table></body></html>")
    pathlib.Path("data/iva_debug.html").write_text("\n".join(html), encoding="utf-8")
    print("Wrote data/iva_debug.html — open http://127.0.0.1:8000/data/iva_debug.html or file://")

if __name__ == "__main__":
    asyncio.run(main())
