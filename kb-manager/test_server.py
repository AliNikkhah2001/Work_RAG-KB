import asyncio
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from kb_manager.web.routes.search import search_knowledge_base

async def test():
    queries = [
        'امتیاز اعتباری چیست؟',
        'تسهیلات چگونه تخصیص می‌شود؟',
        'بازپرداخت قسط چگونه است؟',
    ]
    
    for q in queries:
        result = await search_knowledge_base(q, 5)
        print(f'\nQuery: {q}')
        print(f'Latency: {result.elapsed_ms:.1f}ms')
        print('Top 3 results:')
        for i, r in enumerate(result.final_results[:3]):
            d = r.model_dump()
            print(f'  {i+1}. {r.chunk_id[:8]} - hybrid: {r.hybrid_score:.4f} - rerank: {d.get("rerank_score", "N/A"):.4f}')

if __name__ == "__main__":
    asyncio.run(test())