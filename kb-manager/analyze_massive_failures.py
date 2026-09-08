import os, sys
os.environ['KB_DB_URL']='sqlite+aiosqlite:///D:/Code/KB/kb-manager/data/kb_9_7_2026.db'
os.environ['KB_SOURCE_DIR']=r'D:/Code/KB/kb-source/KB_9.7.2026'
os.environ['PYTHONIOENCODING']='utf-8'
sys.stdout.reconfigure(encoding='utf-8')
import json, pathlib, asyncio


def rank_of(results, gt_id):
    for i, r in enumerate(results):
        if getattr(r, "chunk_id", None) == gt_id:
            return i + 1
    return None


def fmt_rank(rank, n):
    return str(rank) if rank is not None else f">{n}"


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
        steps = await search_knowledge_base(q, top_k=100)
        print(f"Retrieved top-5:")
        for r in steps.final_results[:5]:
            marker = " <== GT" if r.chunk_id == exp else ""
            print(f"  {r.chunk_id[:8]} doc:{r.doc_id[:8]} bm25:{r.bm25_score:.4f} dense:{r.dense_score:.4f} hybrid:{r.hybrid_score:.4f} rerank:{r.rerank_score:.4f}{marker} | {r.content_preview[:80]}")
        bm25_rank = rank_of(steps.bm25_results, exp)
        dense_rank = rank_of(steps.dense_results, exp)
        merged_rank = rank_of(steps.merged_candidates, exp)
        final_rank = rank_of(steps.final_results, exp)
        print(f"GT ranks: bm25={fmt_rank(bm25_rank, len(steps.bm25_results))} dense={fmt_rank(dense_rank, len(steps.dense_results))} merged={fmt_rank(merged_rank, len(steps.merged_candidates))} final={fmt_rank(final_rank, len(steps.final_results))}")
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
