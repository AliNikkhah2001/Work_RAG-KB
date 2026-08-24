#!/usr/bin/env python
"""Cleanup incomplete QA chunks from the knowledge base.

Usage:
    python scripts/cleanup_incomplete_qa.py --dry-run                    # Preview all documents
    python scripts/cleanup_incomplete_qa.py --doc-id <id> --dry-run     # Preview one document
    python scripts/cleanup_incomplete_qa.py --doc-id <id>               # Clean one document
    python scripts/cleanup_incomplete_qa.py --execute-all               # Clean all documents
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kb_manager.cleanup.qa_cleanup import QACleanup
from kb_manager.config import load_config
from kb_manager.models.database import Database


console = Console()


def print_preview_table(preview, show_details: bool = False) -> None:
    """Print a preview table for a document."""
    table = Table(title=f"Document: {preview.document_title} ({preview.document_id[:8]}...)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total QA Chunks", str(preview.total_qa_chunks))
    table.add_row("Complete", str(preview.complete_chunks))
    table.add_row("Incomplete", str(preview.incomplete_count))
    console.print(table)

    if show_details and preview.incomplete_chunks:
        detail_table = Table(title="Incomplete Chunks")
        detail_table.add_column("#", style="dim")
        detail_table.add_column("Chunk ID", style="cyan")
        detail_table.add_column("Ordinal", style="yellow")
        detail_table.add_column("Missing Fields", style="red")
        detail_table.add_column("Question Preview", style="white", max_width=60)

        for i, chunk in enumerate(preview.incomplete_chunks, 1):
            question = chunk.fields.get("question", "")[:60]
            detail_table.add_row(
                str(i),
                chunk.chunk_id[:8] + "...",
                str(chunk.ordinal),
                ", ".join(chunk.missing_fields),
                question + ("..." if len(chunk.fields.get("question", "")) > 60 else ""),
            )
        console.print(detail_table)


async def run_dry_run_all(cleanup: QACleanup) -> None:
    """Show preview for all documents with incomplete chunks."""
    async with cleanup._db.session() as session:
        previews = await cleanup.get_all_documents_preview(session)

    if not previews:
        console.print("[green]No documents with incomplete QA chunks found.[/]")
        return

    console.print(f"\n[bold]Found {len(previews)} document(s) with incomplete QA chunks:[/]\n")

    total_incomplete = 0
    for preview in previews:
        print_preview_table(preview)
        total_incomplete += preview.incomplete_count
        console.print()

    console.print(f"[bold]Total incomplete chunks across all documents: {total_incomplete}[/]")
    console.print("\n[yellow]Run with --execute-all to clean all, or --doc-id <id> to clean one.[/]")


async def run_dry_run_one(cleanup: QACleanup, doc_id: str) -> None:
    """Show preview for a single document."""
    async with cleanup._db.session() as session:
        preview = await cleanup.get_document_preview(session, doc_id)

    if not preview:
        console.print(f"[red]Document not found: {doc_id}[/]")
        return

    print_preview_table(preview, show_details=True)


async def run_clean_one(cleanup: QACleanup, doc_id: str) -> None:
    """Clean a single document."""
    async with cleanup._db.session() as session:
        result = await cleanup.cleanup_document(session, doc_id)

    if result.error:
        console.print(f"[red]Error: {result.error}[/]")
        sys.exit(1)

    console.print(f"\n[green]Cleanup completed for {result.document_title}[/]")
    console.print(f"  Deleted: {result.deleted_count} incomplete chunks")
    console.print(f"  Kept: {result.kept_count} complete chunks")
    if result.version_created:
        console.print(f"  [blue]Version snapshot created[/]")


async def run_clean_all(cleanup: QACleanup) -> None:
    """Clean all documents."""
    async with cleanup._db.session() as session:
        results = await cleanup.cleanup_all_documents(session)

    if not results:
        console.print("[green]No documents with incomplete QA chunks found.[/]")
        return

    console.print(f"\n[bold]Cleaned {len(results)} document(s):[/]\n")

    total_deleted = 0
    for result in results:
        if result.error:
            console.print(f"[red]{result.document_title}: {result.error}[/]")
        else:
            console.print(f"  {result.document_title}: deleted {result.deleted_count}, kept {result.kept_count}")
            total_deleted += result.deleted_count

    console.print(f"\n[bold]Total chunks deleted: {total_deleted}[/]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cleanup incomplete QA chunks from the knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be cleaned without making changes",
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        help="Document ID to preview or clean (optional, defaults to all)",
    )
    parser.add_argument(
        "--execute-all",
        action="store_true",
        help="Execute cleanup for all documents with incomplete chunks",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        help="Database URL override (default: from config/env)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config()
    if args.db_url:
        # Create new DatabaseConfig with override (frozen dataclass)
        from kb_manager.config import DatabaseConfig
        config = type(config)(
            **{**config.__dict__, "db": DatabaseConfig(
                **{**config.db.__dict__, "url_override": args.db_url}
            )}
        )

    # Create cleanup instance
    db = Database(config.db)
    cleanup = QACleanup(db)

    async def _run() -> None:
        if args.dry_run:
            if args.doc_id:
                await run_dry_run_one(cleanup, args.doc_id)
            else:
                await run_dry_run_all(cleanup)
        elif args.execute_all:
            await run_clean_all(cleanup)
        elif args.doc_id:
            await run_clean_one(cleanup, args.doc_id)
        else:
            parser.print_help()
            sys.exit(1)

        await db.close()

    asyncio.run(_run())


if __name__ == "__main__":
    main()