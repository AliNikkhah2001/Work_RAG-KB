"""IVA eval v2: doc-level hit (any top-5 chunk from golden doc) + answer coverage in top-5."""
import asyncio
import json
import os
import pathlib
import sys
import time
import unicodedata

os.environ["KB_DB_URL"] = "sqlite+aiosqlite:///D:/Code/KB/kb-manager/data/kb_1405.db"
os.environ["KB_SYNONYM_ENABLED"] = "true"
sys.stdout.reconfigure(encoding="utf-8")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\u200c", " ").replace("\u064a", "\u06cc").replace("\u0643", "\u06a9")
    for fa, en in zip("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"):
        s = s.replace(fa, en)
    return "".join(c for c in s if c not in "؟?،;,.!").strip().lower()


def toks(s: str) -> set[str]:
    return set(norm(s).split())


async def main():
    from kb_manager.web.routes.search import search_knowledge_base

    dataset_path = pathlib.Path(r"D:/Code/KB/kb-manager/data/test_questions_iva.json")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    s = time.time()
    rows = []
    doc_hits = 0
    ans_hits = 0
    mrr_ranks = []
    for i, item in enumerate(dataset, 1):
        q = item["query"]
        at = toks(item["expected_answer"])
        exp = set(item["expected_chunk_ids"])
        steps = await search_knowledge_base(q, 5)
        retrieved = [(r.chunk_id, r.content_preview) for r in steps.final_results]
        doc_hit = bool(exp and any(cid in exp for cid, _ in retrieved))
        rank = next((ri + 1 for ri, (cid, _) in enumerate(retrieved) if cid in exp), -1)
        covered = set()
        for cid, content in retrieved:
            covered |= (toks(content) & at)
        ans_cov = len(covered) / max(len(at), 1)
        ans_hit = ans_cov >= 0.7
        if rank > 0:
            mrr_ranks.append(1.0 / rank)
        if doc_hit:
            doc_hits += 1
        if ans_hit:
            ans_hits += 1
        rows.append(
            {
                "i": i,
                "query": q[:70],
                "doc_hit": doc_hit,
                "doc_rank": rank,
                "ans_cov": round(ans_cov, 2),
                "ans_hit": ans_hit,
                "elapsed_ms": steps.elapsed_ms,
            }
        )
        print(
            f"{i:2d} doc_hit={int(doc_hit)} rank={rank:2d} ans_cov={ans_cov:.2f} ans_hit={int(ans_hit)} el={steps.elapsed_ms:7.0f}ms | {q[:55]}",
            flush=True,
        )

    mrr = sum(mrr_ranks) / len(dataset)
    print("\n=== IVA v7 (1405-05-31 KB, synonym beam5) ===")
    print(f"Doc-level hit@5 (golden-doc chunk in top-5): {doc_hits}/{len(dataset)} = {doc_hits/len(dataset):.1%}")
    print(f"Answer-hit@5 (>=70% answer tokens in top-5) : {ans_hits}/{len(dataset)} = {ans_hits/len(dataset):.1%}")
    print(f"Doc-level MRR                              : {mrr:.3f}")
    print(f"Avg latency                               : {sum(r['elapsed_ms'] for r in rows)/len(rows):.0f} ms")

    out = pathlib.Path(r"D:/Code/KB/kb-manager/data/iva_results.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("written", out)


asyncio.run(main())