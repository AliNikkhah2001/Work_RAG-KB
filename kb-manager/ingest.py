"""Direct ingestion script - processes all KB files into SQLite."""
import os
import sys
import asyncio
import logging
import time

if not os.getenv("KB_DB_URL"):
    os.environ["KB_DB_URL"] = "sqlite+aiosqlite:///./data/kb_test.db"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest")

KB_DIR = os.getenv("KB_SOURCE_DIR", str(__import__("pathlib").Path(__file__).resolve().parent / "kb-source" / "clean_files"))
# Fallback to legacy nested kb-source location
if not __import__("pathlib").Path(KB_DIR).exists():
    KB_DIR = os.getenv("KB_SOURCE_DIR", str(__import__("pathlib").Path(__file__).resolve().parent.parent / "kb-source" / "clean_files"))

async def main():
    from kb_manager.config import load_config
    from kb_manager.models.database import Database
    from kb_manager.pipeline.orchestrator import PipelineOrchestrator
    from kb_manager.chunker.semantic import SemanticChunker

    config = load_config()
    db = Database(config.db)

    # Create tables
    await db.create_tables()
    logger.info("Tables created")

    # Init components (skip embedder - no GPU/model downloaded)
    chunker = SemanticChunker(max_tokens=512, min_tokens=100)
    orchestrator = PipelineOrchestrator(
        database=db,
        preprocessor=None,  # use default
        chunker=chunker,
        embedder=None,  # skip embedding for now
    )

    logger.info("Starting full rebuild from %s", KB_DIR)
    summary = await orchestrator.run_full_rebuild(KB_DIR)

    await db.close()

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"  Documents processed:  {summary.documents_processed}")
    print(f"  Documents created:    {summary.documents_created}")
    print(f"  Documents updated:    {summary.documents_updated}")
    print(f"  Documents skipped:    {summary.documents_skipped}")
    print(f"  Documents failed:     {summary.documents_failed}")
    print(f"  Chunks created:       {summary.chunks_created}")
    print(f"  Versions created:     {summary.versions_created}")
    print(f"  Elapsed:              {summary.elapsed_seconds:.1f}s")
    if summary.errors:
        print(f"\n  Errors ({len(summary.errors)}):")
        for e in summary.errors[:10]:
            print(f"    - {e['file']}: {e['error']}")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
