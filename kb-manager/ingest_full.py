"""Full re-ingestion from both complete source trees using the new parser.

Uses the parser Unicode-integrity check, calamine or openpyxl engine per
KB_XLSX_ENGINE, and the merged 31Tir + 1405-05-20 corpus.
"""
import os
import sys
import asyncio
import logging
import json

os.environ["KB_DB_URL"] = "sqlite+aiosqlite:///C:/Users/10225/Downloads/KB/kb-manager/data/kb_test.db"
if not os.getenv("KB_XLSX_ENGINE"):
    os.environ["KB_XLSX_ENGINE"] = "calamine"
sys.path.insert(0, r"C:\Users\10225\Downloads\KB\kb-manager")
os.chdir(r"C:\Users\10225\Downloads\KB\kb-manager")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest_full")

KB_DIRS = [
    r"C:\Users\10225\Downloads\KB\extracted_new\31Tir1405",
    r"C:\Users\10225\Downloads\KB\1405-05-20",
]


async def main():
    from kb_manager.config import load_config
    from kb_manager.models.database import Database
    from kb_manager.pipeline.orchestrator import PipelineOrchestrator
    from kb_manager.chunker.semantic import SemanticChunker

    config = load_config()
    logger.info("Using XLSX engine: %s", config.parser.xlsx_engine)
    db = Database(config.db)
    await db.create_tables()
    logger.info("Tables created")

    chunker = SemanticChunker(max_tokens=512, min_tokens=100)
    orchestrator = PipelineOrchestrator(
        database=db,
        preprocessor=None,
        chunker=chunker,
        embedder=None,
    )

    combined = {"documents_processed": 0, "documents_created": 0, "chunks_created": 0,
                "documents_failed": 0, "errors": [], "elapsed_seconds": 0.0}
    total_start = asyncio.get_event_loop().time()

    for kdir in KB_DIRS:
        if not os.path.isdir(kdir):
            logger.warning("Skipping missing source dir: %s", kdir)
            continue
        logger.info("Starting full rebuild from %s", kdir)
        summary = await orchestrator.run_full_rebuild(kdir)
        for k in ("documents_processed", "documents_created", "chunks_created", "documents_failed"):
            combined[k] += getattr(summary, k)
        combined["errors"].extend(summary.errors or [])

    combined["elapsed_seconds"] = asyncio.get_event_loop().time() - total_start
    await db.close()

    print("\n" + "=" * 60)
    print("FULL INGESTION COMPLETE")
    print("=" * 60)
    print(f"  Source dirs:            {KB_DIRS}")
    print(f"  XLSX engine:            {config.parser.xlsx_engine}")
    for k in ("documents_processed", "documents_created", "chunks_created", "documents_failed"):
        print(f"  {k:<22} {combined[k]}")
    print(f"  elapsed_seconds         {combined['elapsed_seconds']:.1f}s")
    if combined["errors"]:
        print(f"\n  Errors ({len(combined['errors'])}):")
        for e in combined["errors"][:10]:
            print(f"    - {e['file']}: {e['error']}")
    print("=" * 60)

    with open(r"C:\Users\10225\Downloads\KB\kb-manager\ingest_full_result.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(combined, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())