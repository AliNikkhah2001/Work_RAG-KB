"""Fresh full re-ingestion with QA deduplication by question text.

Rebuilds kb_test.db from both source trees, keeping one canonical chunk per
normalized question (drops duplicate QA rows found across sheets/documents).

Prints dedup counters + final duplication stats so the result can be packed
into a new version snapshot (v2) and compared with v1.
"""
import asyncio
import json
import logging
import os
import sys

os.environ["KB_DB_URL"] = "sqlite+aiosqlite:///C:/Users/10225/Downloads/KB/kb-manager/data/kb_test.db"
if not os.getenv("KB_XLSX_ENGINE"):
    os.environ["KB_XLSX_ENGINE"] = "calamine"
sys.path.insert(0, r"C:\Users\10225\Downloads\KB\kb-manager")
os.chdir(r"C:\Users\10225\Downloads\KB\kb-manager")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("reingest_dedup")

KB_DIRS = [
    r"C:\Users\10225\Downloads\KB\extracted_new\31Tir1405",
    r"C:\Users\10225\Downloads\KB\1405-05-20",
]


async def _wipe(session) -> None:
    """Drop every stored document/chunk/version/job for a fresh rebuild."""
    from sqlalchemy import delete

    from kb_manager.models.database import (
        Chunk,
        Document,
        DocumentVersion,
        IngestionJob,
        RetrievalLog,
    )

    for model in (Chunk, Document, DocumentVersion, IngestionJob, RetrievalLog):
        await session.execute(delete(model))
    await session.flush()


async def main() -> None:
    from kb_manager.chunker.semantic import SemanticChunker
    from kb_manager.config import load_config
    from kb_manager.models.database import Database
    from kb_manager.pipeline.orchestrator import PipelineOrchestrator

    config = load_config()
    db = Database(config.db)
    await db.create_tables()
    logger.info("Tables ready (engine=%s)", config.parser.xlsx_engine)

    async with db.session() as session:
        await _wipe(session)
    logger.info("Existing KB wiped for fresh rebuild")

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
    for kdir in KB_DIRS:
        if not os.path.isdir(kdir):
            logger.warning("Skipping missing source dir: %s", kdir)
            continue
        logger.info("Full rebuild from %s", kdir)
        summary = await orchestrator.run_full_rebuild(kdir)
        for k in ("documents_processed", "documents_created", "chunks_created", "documents_failed"):
            combined[k] += getattr(summary, k)
        combined["errors"].extend(summary.errors or [])
    combined["elapsed_seconds"] = round(_time.monotonic() - total_start, 1)

    dedup = chunker.dedup_stats()

    await db.close()

    # Final duplication stats straight from the DB.
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

    with open(r"C:\Users\10225\Downloads\KB\kb-manager\data\reingest_dedup_summary.json", "w", encoding="utf-8") as f:
        json.dump({"combined": combined, "dedup": dedup, "counts": counts}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
