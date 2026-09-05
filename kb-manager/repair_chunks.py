"""Repair ZWNJ-corrupted chunks by re-ingesting.

Run: python repair_chunks.py
Uses KB_SOURCE_DIR (1405-05-31) and rebuilds all chunks with fixed persian.py.
"""
import asyncio
import os
os.environ["KB_DB_URL"] = "sqlite+aiosqlite:///D:/Code/KB/kb-manager/data/kb_1405.db"
os.environ["KB_SOURCE_DIR"] = r"D:/Code/KB/kb-source/1405-05-31"

from kb_manager.config import load_config
from kb_manager.models.database import Database
from kb_manager.pipeline.orchestrator import PipelineOrchestrator
from kb_manager.chunker.semantic import SemanticChunker

async def main():
    cfg = load_config()
    db = Database(cfg.db)
    await db.create_tables()
    # reset dedup state
    chunker = SemanticChunker(parent_scope="sheet")
    orch = PipelineOrchestrator(database=db, chunker=chunker)
    print(f"Rebuilding from {cfg.source_dir} ...")
    summary = await orch.run_full_rebuild(cfg.source_dir)
    print("Done:", summary.to_dict())
    # verify one sample
    from sqlalchemy import select
    from kb_manager.models.database import Chunk
    async with db.session() as s:
        res = await s.execute(select(Chunk).where(Chunk.content.like("%نا‌م%")).limit(5))
        bad = res.scalars().all()
        if bad:
            print(f"WARNING: still {len(bad)} chunks contain 'نا‌م' (ZWNJ corruption)")
            for c in bad[:2]:
                print(c.content[:200])
        else:
            print("OK: no 'نا‌م' corruption found")
        res2 = await s.execute(select(Chunk).where(Chunk.content.like("%نام و نام خانوادگی%")).limit(2))
        good = res2.scalars().all()
        print(f"Found {len(good)} chunks with correct header 'نام و نام خانوادگی'")

if __name__ == "__main__":
    asyncio.run(main())
