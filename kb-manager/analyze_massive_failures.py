import os, sys
os.environ['KB_DB_URL']='sqlite+aiosqlite:///D:/Code/KB/kb-manager/data/kb_9_7_2026.db'
os.environ['KB_SOURCE_DIR']=r'D:/Code/KB/kb-source/KB_9.7.2026'
os.environ['PYTHONIOENCODING']='utf-8'
sys.stdout.reconfigure(encoding='utf-8')
import json, pathlib, asyncio

async def analyze():
    from kb_manager.web.routes.search import search_knowledge_base
    from sqlalchemy import text
    from kb_manager.config import load_config
    from kb_manager.models.database import Database

    massive = json.loads(pathlib.Path("data/massive_results.json").read_text(encoding="utf-8"))
    print(f"Massive: total {massive['total']} passed {massive['passed']} failed {massive['failed']} hit {massive['hit_rate']:.3f}")
    # Analyze first 3 failed samples in detail
    for i, fail in enumerate(massive['failed_samples'][:3]):
        q = fail['question']
        exp = fail['expected']
        print(f"\n--- Failed #{i+1}: {q[:60]} ---")
        print(f"Expected: {exp}")
        # Get GT content and scores
        cfg = load_config()
        db = Database(cfg.db)
        async with db.session() as s:
            r = await s.execute(text("SELECT content, chunk_type, heading_path FROM chunks WHERE id=:id"), {"id": exp})
            row = r.fetchone()
            if row:
                print(f"GT chunk_type={row[1]} heading={row[2]}")
                print(f"GT content preview: {row[0][:200]}")
            else:
                print("GT not found in DB")
        # Run search and show scores for GT vs top-1
        steps = await search_knowledge_base(q, top_k=10)
        print(f"Retrieved top-5:")
        for r in steps.final_results[:5]:
            marker = " <== GT" if r.chunk_id == exp else ""
            print(f"  {r.chunk_id[:8]} doc:{r.doc_id[:8]} bm25:{r.bm25_score:.4f} dense:{r.dense_score:.4f} hybrid:{r.hybrid_score:.4f} rerank:{r.rerank_score:.4f}{marker} | {r.content_preview[:80]}")
        # Find GT rank in merged candidates (before rerank) and final
        merged_rank = next((idx+1 for idx, r in enumerate(steps.merged_candidates) if r.chunk_id == exp), -1)
        final_rank = next((idx+1 for idx, r in enumerate(steps.final_results) if r.chunk_id == exp), -1)
        print(f"GT merged rank: {merged_rank} (before rerank), final rank: {final_rank} (after rerank)")
        # Also check BM25 and Dense separate
        bm25_rank = next((idx+1 for idx, r in enumerate(steps.bm25_results) if r.chunk_id == exp), -1)
        dense_rank = next((idx+1 for idx, r in enumerate(steps.dense_results) if r.chunk_id == exp), -1)
        print(f"GT BM25 rank: {bm25_rank}, Dense rank: {dense_rank}")
        await db.close()

    # Also check IVA
    iva = json.loads(pathlib.Path("data/iva_results.json").read_text(encoding="utf-8"))
    hits = sum(1 for r in iva if r.get("doc_hit"))
    print(f"\nIVA: {hits}/15 hit, MRR calculated")
    for r in iva:
        if not r.get("doc_hit"):
            print(f"  Miss Q{r['i']}: rank {r['doc_rank']} ans_cov {r['ans_cov']} q:{r['query'][:50]}")

if __name__ == "__main__":
    asyncio.run(analyze())
