import asyncio
import time

async def main():
    from kb_manager.web.routes.search import _get_index, _index_cache, _index_cache_count
    
    print(f"Cache before: cache={_index_cache is not None}, count={_index_cache_count}")
    
    t0 = time.monotonic()
    result = await _get_index()
    elapsed = time.monotonic() - t0
    
    print(f"_get_index took {elapsed:.1f}s")
    print(f"Cache after: cache={result is not None}")
    print(f"Chunk data: {len(result[0])} chunks")
    print(f"BM25 docs: {result[1].doc_count}")
    
    # Test a search
    from kb_manager.web.routes.search import search_knowledge_base
    t1 = time.monotonic()
    steps = await search_knowledge_base("test", 3)
    elapsed2 = time.monotonic() - t1
    print(f"Search took {elapsed2:.1f}s, {len(steps.final_results)} results")

asyncio.run(main())
