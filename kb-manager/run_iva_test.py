"""Run the 15 IVA questions through the live v7 search pipeline (BM25+Dense+RRF+rerank)."""
import asyncio
import json
import os
import pathlib
import sys
import time

os.environ["KB_DB_URL"] = "sqlite+aiosqlite:///D:/Code/KB/kb-manager/data/kb_1405.db"
os.environ["KB_SYNONYM_ENABLED"] = "true"
sys.stdout.reconfigure(encoding="utf-8")


async def main():
    from kb_manager.evaluation.benchmark import AsyncBenchmarkRunner

    from kb_manager.web.routes.search import search_knowledge_base

    dataset_path = pathlib.Path(r"D:/Code/KB/kb-manager/data/test_questions_iva.json")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    print(f"IVA questions: {len(dataset)}")

    async def _search(q, k):
        steps = await search_knowledge_base(q, k)
        return [(r.chunk_id, r.hybrid_score) for r in steps.final_results]

    s = time.time()
    runner = AsyncBenchmarkRunner(_search, top_k=5, version="v7-iva-1405")
    result = await runner._run_async(
        dataset, progress=lambda d, t: print(f"  [{d}/{t}] {time.time()-s:.0f}s", flush=True)
    )
    print("\n=== OVERALL ===")
    print(json.dumps(result.overall, ensure_ascii=False, indent=2))
    print("\n=== PER QUERY ===")
    if hasattr(result, "queries"):
        for q in result.queries:
            print(
                f"  hit={q.get('hit')} rank={q.get('rank')} elapsed={q.get('elapsed_ms'):.0f}ms | {str(q.get('query',''))[:60]}"
            )
    # Write results
    out = pathlib.Path(r"D:/Code/KB/kb-manager/data/benchmark_results_iva.json")
    out.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwritten", out)


asyncio.run(main())