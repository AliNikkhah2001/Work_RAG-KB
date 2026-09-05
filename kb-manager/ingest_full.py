"""Full re-ingestion from both complete source trees using the new parser.

Uses the parser Unicode-integrity check, calamine or openpyxl engine per
KB_XLSX_ENGINE, and the merged 31Tir + 1405-05-20 corpus.
"""
import os
import sys
import asyncio
import logging
import json

if not os.getenv("KB_DB_URL"):
    os.environ["KB_DB_URL"] = "sqlite+aiosqlite:///./data/kb_test.db"
if not os.getenv("KB_XLSX_ENGINE"):
    os.environ["KB_XLSX_ENGINE"] = "calamine"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest_full")

KB_DIRS = [
    os.getenv("KB_SOURCE_DIR", str(__import__("pathlib").Path(__file__).resolve().parent.parent / "kb-source" / "clean_files")),
    os.getenv("KB_SOURCE_DIR_2", ""),
]
KB_DIRS = [d for d in KB_DIRS if d and __import__("pathlib").Path(d).exists()]


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

    out_path = __import__("pathlib").Path(__file__).resolve().parent / "ingest_full_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(combined, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())