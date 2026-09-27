"""Check whether correct doc is in pre-rerank candidates (top-50) for the 4 misses."""
import asyncio
import os
import sys

os.environ["KB_DB_URL"] = "sqlite+aiosqlite:///D:/Code/KB/kb-manager/data/kb_1405.db"
os.environ["KB_SYNONYM_ENABLED"] = "true"
sys.stdout.reconfigure(encoding="utf-8")


async def main():
    from kb_manager.web.routes import search as S

    qs = {
        11: "این دلیل کاهش امتیاز یعنی چی: ؛؛وام‌های ضامنت شده توسط فرد، در ماه‌های بسیار کمی",
        12: "این دلیل کاهش امتیاز رو چطوری بهبود بدهم؟ «وام‌های ضامنت شده توسط فرد، در ماه‌ها",
        14: "رتبه چه فرقی با امتیاز داره؟",
        15: "جزئیات قراردادهای منفی در گزارش یعنی چی؟",
    }
    for i, q in qs.items():
        steps = await S.search_knowledge_base(q, 5)
        print(f"\n### Q{i} (elapsed {steps.elapsed_ms:.0f}ms)")
        for rank, r in enumerate(steps.merged_candidates, 1):
            if rank > 15:
                break
            print(f"  pre-rerank r{rank}: [{r.hybrid_score:.5f}] {r.doc_title[:38]} | {(r.heading_path or '')[:34]}")


asyncio.run(main())