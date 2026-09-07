"""Detailed 15q IVA debug: GT vs retrieved with scores and rank of GT."""
import os
os.environ['KB_DB_URL']='sqlite+aiosqlite:///D:/Code/KB/kb-manager/data/kb_9_7_2026.db'
os.environ['KB_SOURCE_DIR']=r'D:/Code/KB/kb-source/KB_9.7.2026'
import asyncio, json, pathlib
import sys
sys.stdout.reconfigure(encoding='utf-8')

async def main():
    from kb_manager.web.routes.search import search_knowledge_base
    import json as _json
    data = _json.loads(pathlib.Path("data/test_questions_iva.json").read_text(encoding="utf-8"))
    # Load iva_results for reference
    iva_results = _json.loads(pathlib.Path("data/iva_results.json").read_text(encoding="utf-8")) if pathlib.Path("data/iva_results.json").exists() else []

    html = ["""<html><head><meta charset='utf-8'>
<link href='https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css' rel='stylesheet'>
<style>
body{background:#0f172a;color:#f1f5f9;font-family:Vazirmatn,Tahoma;padding:16px}
.p{font-family:Vazirmatn,Tahoma}
table{border-collapse:collapse;width:100%;margin-bottom:24px}
td,th{border:1px solid #334155;padding:8px;font-size:13px;vertical-align:top}
th{background:#1e293b;color:#f1f5f9}
.hit{background:rgba(34,197,94,0.15)}
.miss{background:rgba(239,68,68,0.15)}
.mono{font-family:Consolas,monospace;font-size:11px}
.score{font-family:Consolas,monospace;font-size:11px;color:#94a3b8}
.gt{border:2px solid #22c55e}
</style></head><body>"""]
    html.append("<h1>15q IVA — Case-by-Case Debug (New KB: KB_9.7.2026, 21 docs, 1084 chunks)</h1>")
    html.append(f"<p>Generated: {__import__('datetime').datetime.now().isoformat()} | <a href='/benchmarks/massive' style='color:#60a5fa'>Back to Benchmarks</a></p>")

    for idx, item in enumerate(data):
        q = item["query"]
        exp_ans = item["expected_answer"]
        exp_ids = set(item["expected_chunk_ids"])
        # Get search results with full scores
        steps = await search_knowledge_base(q, top_k=10)
        # Find rank of GT
        rank = -1
        gt_chunk = None
        for r_idx, r in enumerate(steps.final_results):
            if r.chunk_id in exp_ids:
                rank = r_idx + 1
                gt_chunk = r
                break
        # Also check if GT is in merged candidates but not in final (reranker demoted)
        merged_rank = -1
        for r_idx, r in enumerate(steps.merged_candidates):
            if r.chunk_id in exp_ids:
                merged_rank = r_idx + 1
                break
        is_hit = rank != -1 and rank <= 5
        # Get IVA result for this query if available
        iva_row = iva_results[idx] if idx < len(iva_results) else {}
        html.append(f"<div style='border:1px solid #334155;border-radius:8px;padding:16px;margin-bottom:20px;background:{'#1e293b' if is_hit else '#1e293b'}'>")
        html.append(f"<h3 style='margin:0 0 8px 0'>Q{idx+1}: <span class='p' dir='auto'>{q}</span> <span style='float:right;padding:4px 8px;border-radius:4px;background:{'#22c55e' if is_hit else '#ef4444'};color:white;font-size:12px'>{'HIT rank '+str(rank) if is_hit else 'MISS rank '+str(rank if rank!=-1 else 'NF') }</span></h3>")
        html.append(f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px'>")
        # Left: GT
        html.append(f"<div style='background:#0f172a;padding:12px;border-radius:6px'><h4 style='margin:0 0 8px 0;color:#22c55e'>Ground Truth</h4>")
        html.append(f"<div class='p' dir='auto' style='background:#1e293b;padding:8px;border-radius:4px;max-height:150px;overflow:auto'>{exp_ans[:500]}</div>")
        html.append(f"<div class='mono' style='margin-top:8px;color:#94a3b8'>Expected chunks: {len(exp_ids)} | Ans coverage in top-5: {iva_row.get('ans_cov','?')} | IVA doc_rank: {iva_row.get('doc_rank','?')}</div>")
        if gt_chunk:
            html.append(f"<div style='margin-top:8px;padding:8px;background:rgba(34,197,94,0.1);border-radius:4px'>")
            html.append(f"<b>GT found at rank {rank} (final) / {merged_rank} (merged before rerank)</b><br>")
            html.append(f"<span class='score'>BM25: {gt_chunk.bm25_score} | Dense: {gt_chunk.dense_score} | Hybrid: {gt_chunk.hybrid_score} | Rerank: {gt_chunk.rerank_score}</span><br>")
            html.append(f"<span class='mono'>Doc: {gt_chunk.doc_title} | Chunk: {gt_chunk.chunk_id[:8]} | {gt_chunk.heading_path}</span><br>")
            html.append(f"<div class='p mono' dir='auto' style='margin-top:4px;max-height:100px;overflow:auto'>{gt_chunk.content_preview[:300]}</div>")
            html.append(f"</div>")
        else:
            html.append(f"<div style='margin-top:8px;padding:8px;background:rgba(239,68,68,0.1);border-radius:4px'>GT <b>NOT in top-10 final</b> — merged rank: {merged_rank if merged_rank!=-1 else 'NF (>50)'}<br>Expected chunks are from doc with {len(exp_ids)} chunks, none matched.</div>")
        html.append(f"</div>")
        # Right: Retrieved top-5
        html.append(f"<div style='background:#0f172a;padding:12px;border-radius:6px'><h4 style='margin:0 0 8px 0'>Top-5 Retrieved (final after rerank)</h4>")
        for r_idx, r in enumerate(steps.final_results[:5]):
            is_gt = r.chunk_id in exp_ids
            html.append(f"<div style='padding:6px;margin-bottom:6px;background:#1e293b;border-radius:4px;border:{'2px solid #22c55e' if is_gt else '1px solid #334155'}'>")
            html.append(f"<b>#{r_idx+1} {'⭐ GT' if is_gt else ''} {r.doc_title}</b> <span class='mono' style='float:right'>{r.chunk_id[:8]}</span><br>")
            html.append(f"<span class='score'>BM25:{r.bm25_score} Dense:{r.dense_score} Hybrid:{r.hybrid_score} Rerank:{r.rerank_score}</span><br>")
            html.append(f"<span class='mono'>{r.heading_path} | ordinal:{r.ordinal}</span><br>")
            html.append(f"<div class='p' dir='auto' style='margin-top:4px;font-size:12px'>{r.content_preview[:200]}</div>")
            html.append(f"</div>")
        html.append(f"</div>")
        html.append(f"</div>")

        # Why miss analysis
        if not is_hit:
            html.append(f"<div style='margin-top:12px;padding:10px;background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);border-radius:6px'>")
            html.append(f"<b>Why MISS?</b> ")
            if merged_rank != -1 and rank == -1:
                html.append(f"GT was in merged candidates at rank {merged_rank} but demoted by cross-encoder reranker (top-50 → final 5). Try <code>RERANKER_TOP_K 50→100</code> or check rerank score: GT rerank {gt_chunk.rerank_score if gt_chunk else 'N/A'} vs top-1 {steps.final_results[0].rerank_score if steps.final_results else 'N/A'}.")
            elif merged_rank == -1:
                html.append(f"GT not even in BM25+Dense merged top-50. Query may be too colloquial/truncated (e.g. Q11 '؛؛وام‌های ضمانت' leading ؛), or vocabulary mismatch. Check <code>query_expansion.py</code> synonyms and <code>clean.py</code> lstrip. BM25 tokens for query: {steps.tokens[:10]}")
            else:
                html.append(f"GT at rank {rank} (>5). Close but outside top-5. Consider increasing top_k or tuning RRF k=60.")
            html.append(f"</div>")
        html.append(f"</div>")

    html.append("</body></html>")
    pathlib.Path("data/iva_detailed_debug.html").write_text("\n".join(html), encoding="utf-8")
    print("Wrote data/iva_detailed_debug.html")

if __name__ == "__main__":
    asyncio.run(main())
