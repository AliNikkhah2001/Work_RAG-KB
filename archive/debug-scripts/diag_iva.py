"""Diagnose why IVA queries miss: show top-5 retrieved content for failing queries."""
import asyncio
import os
import pathlib
import sys
import unicodedata

os.environ["KB_DB_URL"] = "sqlite+aiosqlite:///D:/Code/KB/kb-manager/data/kb_1405.db"
os.environ["KB_SYNONYM_ENABLED"] = "true"
sys.stdout.reconfigure(encoding="utf-8")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\u200c", " ").replace("\u064a", "\u06cc").replace("\u0643", "\u06a9")
    return "".join(c for c in s if c not in "؟?،;,.!").strip().lower()


def toks(s: str) -> set[str]:
    return set(norm(s).split())


FAILING = [
    "چرا یکی از وام هایی که دارم قسطشون رو میدم، توی گزارش اعتباری من نیست؟",
    "چی کار کنم رتبم بهتر بشه؟",
    "گزارش من اشتباه داره. چی کار کنم؟",
    "من قسط وام خودم رو پرداخت کردم. اطلاعات من چه زمانی به روز میشه؟",
    "رتبه چه فرقی با امتیاز داره؟",
]


async def main():
    from kb_manager.web.routes.search import search_knowledge_base

    for q in FAILING:
        steps = await search_knowledge_base(q, 5)
        print("\n### QUERY:", q[:80])
        print("tokens:", steps.tokens[:20])
        print("bm25 results:", len(steps.bm25_results), "dense:", len(steps.dense_results))
        for r in steps.final_results:
            cov = len(steps.tokens and set(steps.tokens) & toks(r.content_preview)) if steps.tokens else 0
            print(
                f"  [{r.hybrid_score:.4f} rerank={r.rerank_score:.4f}] {r.doc_title[:30]} | {r.content_preview[:120].replace(chr(10),' ')}"
            )


asyncio.run(main())