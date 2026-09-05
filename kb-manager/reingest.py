"""Re-ingest with QA deduplication enabled."""
import os
os.environ["KB_DB_URL"] = "sqlite+aiosqlite:///C:/Users/10225/Downloads/KB/kb-manager/data/kb_test.db"

import asyncio
import sys
sys.path.insert(0, r"C:\Users\10225\Downloads\KB\kb-manager")

from kb_manager.config import load_config
from kb_manager.chunker.semantic import SemanticChunker
from kb_manager.pipeline.orchestrator import PipelineOrchestrator
from kb_manager.models.database import Database

config = load_config()
config = type(config)(
    **{
        **config.__dict__,
        "chunking": type(config.chunking)(
            **{**config.chunking.__dict__, "dedup_questions": True}
        ),
    }
)

db = Database(config.db)
async def main():
    await db.create_tables()
    async with db.session() as session:
        from sqlalchemy import delete
        from kb_manager.models.database import Chunk, Document, DocumentVersion, IngestionJob, RetrievalLog
        for model in (Chunk, Document, DocumentVersion, IngestionJob, RetrievalLog):
            await session.execute(delete(model))
        await session.flush()
    
    chunker = SemanticChunker(max_tokens=512, min_tokens=100, dedup_questions=True)
    chunker.reset_dedup()
    orchestrator = PipelineOrchestrator(
        database=db,
        preprocessor=None,
        chunker=chunker,
        embedder=None,
    )
    
    combined = {
        "documents_processed": 0,
        "documents_created": 0,
        "chunks_created": 0,
        "documents_failed": 0,
        "errors": [],
        "elapsed_seconds": 0.0,
    }
    
    import time as _time
    total_start = _time.monotonic()
    KB_DIRS = [
        r"C:\Users\10225\Downloads\KB\extracted_new\31Tir1405",
        r"C:\Users\10225\Downloads\KB\1405-05-20",
    ]
    for kdir in KB_DIRS:
        if not os.path.isdir(kdir):
            print(f"Skipping missing source dir: {kdir}")
            continue
        print(f"Full rebuild from {kdir}")
        summary = await orchestrator.run_full_rebuild(kdir)
        for k in ("documents_processed", "documents_created", "chunks_created", "documents_failed"):
            combined[k] += getattr(summary, k, 0)
        combined["errors"].extend(summary.errors or [])
    combined["elapsed_seconds"] = round(_time.monotonic() - total_start, 1)
    
    dedup = chunker.dedup_stats()
    
    await db.close()
    
    from kb_manager.versioning.snapshot import export_compact
    export = export_compact(r"C:\Users\10225\Downloads\KB\kb-manager\data\kb_test.db")
    counts = export["counts"]
    
    print("\n" + "=" * 60)
    print("RE-INGESTION (DEDUP) COMPLETE")
    print("=" * 60)
    print(f"  documents_processed  {combined['documents_processed']}")
    print(f"  documents_created   {combined['documents_created']}")
    print(f"  chunks_created      {combined['chunks_created']}")
    print(f"  elapsed_seconds     {combined['elapsed_seconds']}s")
    print("\n  Dedup (question text):")
    print(f"    enabled            {dedup['enabled']}")
    print(f"    qa_rows_kept       {dedup['qa_rows_kept']}")
    print(f"    duplicates_skipped {dedup['duplicates_skipped']}")
    print(f"    distinct_questions {dedup['distinct_questions']}")
    print("\n  Final KB counts:")
    for k, v in counts.items():
        print(f"    {k:<26} {v}")

asyncio.run(main())