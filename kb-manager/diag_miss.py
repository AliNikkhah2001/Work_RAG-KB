"""Diagnose the 4 IVA doc-misses (11,12,14,15)."""
import asyncio
import os
import sys
import unicodedata

os.environ["KB_DB_URL"] = "sqlite+aiosqlite:///D:/Code/KB/kb-manager/data/kb_1405.db"
os.environ["KB_SYNONYM_ENABLED"] = "true"
sys.stdout.reconfigure(encoding="utf-8")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\u200c", " ").replace("\u064a", "\u06cc").replace("\u0643", "\u06a9")
    return "".join(c for c in s if c not in "؟?،;,.!").strip().lower()


QS = {
    11: "این دلیل کاهش امتیاز یعنی چی: ؛؛وام‌های ضمانت شده توسط فرد، در ماه‌های بسیار کمی",
    12: "این دلیل کاهش امتیاز رو چطوری بهبود بدهم؟ «وام‌های ضمانت شده توسط فرد، در ماه‌ها",
    14: "رتبه چه فرقی با امتیاز داره؟",
    15: "جزئیات قراردادهای منفی در گزارش یعنی چی؟",
}


async def main():
    from kb_manager.web.routes.search import search_knowledge_base

    for i in sorted(QS):
        q = QS[i]
        steps = await search_knowledge_base(q, 5)
        print(f"\n### Q{i}: {q[:70]}")
        for r in steps.final_results:
            print(
                f"  [{r.hybrid_score:.4f} rerank={r.rerank_score:.4f}] {r.doc_title[:34]} | {r.heading_path[:40] if r.heading_path else ''}"
            )


asyncio.run(main())