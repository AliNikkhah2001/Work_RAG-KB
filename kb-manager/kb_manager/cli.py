"""CLI interface for KB Manager."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from kb_manager.config import load_config

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """KB Manager - Knowledge Base Management System for ICS Credit Scoring."""


@main.command()
@click.option("--source-dir", "-s", type=click.Path(exists=True), help="Source directory")
@click.option("--full", is_flag=True, help="Full rebuild (ignore changes)")
@click.option("--model", "-m", default=None, help="Embedding model name")
@click.option(
    "--parent-scope",
    type=click.Choice(["sheet", "document"], case_sensitive=False),
    default=None,
    help="Parent chunk scope: one parent per sheet or per document (default: config)",
)
def ingest(source_dir: str | None, full: bool, model: str | None, parent_scope: str | None) -> None:
    """Ingest documents into the knowledge base."""
    config = load_config()
    if source_dir:
        config = type(config)(**{**config.__dict__, "source_dir": source_dir})
    if parent_scope:
        old_chunking = config.chunking
        config = type(config)(
            **{
                **config.__dict__,
                "chunking": type(old_chunking)(
                    **{**old_chunking.__dict__, "parent_scope": parent_scope.lower()}
                ),
            }
        )

    console.print("[bold blue]Starting ingestion pipeline...[/]")

    async def _run() -> None:
        from kb_manager.models.database import Database
        from kb_manager.pipeline.orchestrator import PipelineOrchestrator

        db = Database(config.db)
        await db.create_tables()

        try:
            async with db.session() as session:
                from kb_manager.chunker import get_chunker
                from kb_manager.embedder import get_embedder
                from kb_manager.parsers import get_parser
                from kb_manager.preprocessor import PreprocessingPipeline

                parser = get_parser
                preprocessor = PreprocessingPipeline()
                chunker = get_chunker(
                    config.chunking.strategy,
                    max_tokens=config.chunking.max_tokens,
                    min_tokens=config.chunking.min_tokens,
                    overlap_tokens=config.chunking.overlap_tokens,
                    parent_scope=config.chunking.parent_scope,
                    parent_max_tokens=config.chunking.parent_max_tokens,
                    dedup_questions=config.chunking.dedup_questions,
                )
                embedder = get_embedder(
                    "sentence_transformer",
                    model_name=model or config.embedding.model_name,
                    dimensions=config.embedding.dimensions,
                    batch_size=config.embedding.batch_size,
                )

                orchestrator = PipelineOrchestrator(
                    db=db,
                    preprocessor=preprocessor,
                    chunker=chunker,
                    embedder=embedder,
                )

                if full:
                    result = await orchestrator.run_full_rebuild(
                        source_dir=config.source_dir, session=session
                    )
                else:
                    result = await orchestrator.run_incremental(
                        source_dir=config.source_dir, session=session
                    )

                console.print(f"\n[bold green]Pipeline completed![/]")
                console.print(
                    f"  Documents processed: {result.documents_ok}/{result.documents_total}"
                )
                console.print(f"  Chunks created: {result.chunks_total}")
                console.print(f"  Failed: {result.documents_failed}")
                if result.chunks_skipped_incomplete > 0:
                    console.print(f"  QA rows skipped (incomplete): {result.chunks_skipped_incomplete}")
        finally:
            await db.close()

    asyncio.run(_run())


@main.command()
def status() -> None:
    """Show current KB status."""
    config = load_config()

    async def _run() -> None:
        from sqlalchemy import func, select

        from kb_manager.models.database import Chunk, Database, Document

        db = Database(config.db)
        await db.create_tables()

        try:
            async with db.session() as session:
                doc_count = (await session.execute(select(func.count(Document.id)))).scalar() or 0
                chunk_count = (await session.execute(select(func.count(Chunk.id)))).scalar() or 0

                table = Table(title="KB Status")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")
                table.add_row("Documents", str(doc_count))
                table.add_row("Chunks", str(chunk_count))
                console.print(table)
        finally:
            await db.close()

    asyncio.run(_run())


@main.command()
@click.option("--query", "-q", required=True, help="Search query")
@click.option("--top-k", "-k", default=5, help="Number of results")
def search(query: str, top_k: int) -> None:
    """Search the knowledge base."""
    config = load_config()

    async def _run() -> None:
        from kb_manager.models.database import Database

        db = Database(config.db)
        await db.create_tables()

        try:
            from kb_manager.embedder import get_embedder

            embedder = get_embedder(
                "sentence_transformer",
                model_name=config.embedding.model_name,
                dimensions=config.embedding.dimensions,
            )
            query_embedding = embedder.embed_query(query)

            async with db.session() as session:
                from sqlalchemy import text

                sql = text("""
                    SELECT c.id, c.content, c.heading_path, d.title,
                           1 - (c.embedding <=> :embedding) AS similarity
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE c.embedding IS NOT NULL
                    ORDER BY c.embedding <=> :embedding
                    LIMIT :limit
                """)
                result = await session.execute(
                    sql, {"embedding": str(query_embedding), "limit": top_k}
                )
                rows = result.fetchall()

                if not rows:
                    console.print("[yellow]No results found.[/]")
                    return

                table = Table(title=f"Search Results for: {query}")
                table.add_column("#", style="dim")
                table.add_column("Score", style="green")
                table.add_column("Document", style="cyan")
                table.add_column("Heading")
                table.add_column("Content", max_width=60)

                for i, row in enumerate(rows, 1):
                    table.add_row(
                        str(i),
                        f"{float(row[4]):.3f}",
                        row[3],
                        row[2][:50] if row[2] else "",
                        row[1][:100],
                    )
                console.print(table)
        finally:
            await db.close()

    asyncio.run(_run())


@main.command()
def serve() -> None:
    """Start the web server."""
    config = load_config()
    console.print(f"[bold green]Starting web server on {config.web_host}:{config.web_port}[/]")

    import uvicorn

    uvicorn.run(
        "kb_manager.web.app:app",
        host=config.web_host,
        port=config.web_port,
        reload=True,
    )


@main.command()
@click.option("--file", "-f", type=click.Path(exists=True), help="File to inspect")
def inspect(file: str) -> None:
    """Inspect a file's structure and content."""
    from kb_manager.parsers import get_parser

    path = Path(file)
    parser = get_parser(str(path))

    if parser is None:
        console.print(f"[red]No parser found for: {path.name}[/]")
        return

    try:
        doc = parser.parse(str(path))
        console.print(f"[bold]File:[/] {path.name}")
        console.print(f"[bold]Title:[/] {doc.title}")
        console.print(f"[bold]Type:[/] {doc.file_type}")
        console.print(f"[bold]Content length:[/] {len(doc.content)} chars")

        if doc.sheets:
            console.print(f"\n[bold]Sheets ({len(doc.sheets)}):[/]")
            for sheet in doc.sheets:
                console.print(
                    f"  - {sheet['name']}: {len(sheet.get('headers', []))} cols, {len(sheet.get('rows', []))} rows"
                )
                if sheet.get("headers"):
                    console.print(f"    Headers: {sheet['headers'][:5]}")

        if doc.sections:
            console.print(f"\n[bold]Sections ({len(doc.sections)}):[/]")
            for sec in doc.sections[:5]:
                console.print(f"  - {sec.get('heading', 'N/A')}: {len(sec.get('text', ''))} chars")

        console.print(f"\n[bold]Preview (first 500 chars):[/]")
        console.print(doc.content[:500])

    except Exception as e:
        console.print(f"[red]Error parsing file: {e}[/]")


# ---------------------------------------------------------------------------
# Evaluation commands
# ---------------------------------------------------------------------------


@main.command("eval-generate")
@click.option("--db-path", "-d", default="data/kb_test.db", help="SQLite database path")
@click.option("--output", "-o", default="kb_manager/evaluation/datasets/eval.json", help="Output dataset path")
@click.option("--max-queries", "-n", default=100, help="Max queries to generate")
def eval_generate(db_path: str, output: str, max_queries: int) -> None:
    """Generate synthetic evaluation dataset from ingested chunks."""
    from kb_manager.evaluation.generator import SyntheticDataGenerator

    console.print("[bold blue]Generating synthetic evaluation dataset...[/]")
    gen = SyntheticDataGenerator(db_path)
    queries = gen.generate_full_dataset(output, max_queries=max_queries)
    console.print(f"[bold green]Generated {len(queries)} evaluation queries[/]")
    console.print(f"  Saved to: {output}")


@main.command("eval-run")
@click.option("--db-path", "-d", default="data/kb_test.db", help="SQLite database path")
@click.option("--dataset", "-i", default="kb_manager/evaluation/datasets/eval.json", help="Evaluation dataset path")
@click.option("--top-k", "-k", default=10, help="Number of results to evaluate")
def eval_run(db_path: str, dataset: str, top_k: int) -> None:
    """Run retrieval evaluation metrics on the KB."""
    from kb_manager.evaluation.metrics import EvaluationRunner
    from kb_manager.evaluation.metrics import RetrievalMetrics

    console.print("[bold blue]Running retrieval evaluation...[/]")

    # Simple BM25-like search using SQLite FTS or content matching
    import json
    import sqlite3

    conn = sqlite3.connect(db_path)

    def simple_search(query: str, k: int) -> list[tuple[str, float]]:
        """Simple keyword search for evaluation."""
        cur = conn.cursor()
        terms = query.split()
        # Build a simple LIKE-based search
        conditions = " OR ".join(["content LIKE ?" for _ in terms])
        params = [f"%{t}%" for t in terms]
        cur.execute(
            f"SELECT id, content FROM chunks WHERE {conditions} LIMIT ?",
            params + [k * 2],
        )
        rows = cur.fetchall()

        # Score by term overlap
        scored = []
        for cid, content in rows:
            content_lower = content.lower()
            score = sum(1 for t in terms if t.lower() in content_lower)
            scored.append((cid, score / len(terms) if terms else 0.0))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    runner = EvaluationRunner(db_path)
    results = runner.run_retrieval_evaluation(dataset, simple_search, k=top_k)

    console.print(f"\n[bold green]Evaluation Results (K={top_k}):[/]")
    for metric, value in results.items():
        console.print(f"  {metric}: {value:.4f}")

    conn.close()


@main.command("status-chunks")
@click.option("--db-path", "-d", default="data/kb_test.db", help="SQLite database path")
def status_chunks(db_path: str) -> None:
    """Show detailed chunk statistics."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Chunk type distribution
    cur.execute("SELECT chunk_type, COUNT(*) FROM chunks GROUP BY chunk_type")
    type_counts = cur.fetchall()

    # Token stats
    cur.execute("SELECT MIN(token_count), MAX(token_count), AVG(token_count) FROM chunks")
    token_stats = cur.fetchone()

    # QA completeness
    cur.execute("SELECT COUNT(*) FROM chunks WHERE chunk_type = 'qa_pair'")
    qa_total = cur.fetchone()[0]
    cur.execute(
        "SELECT COUNT(*) FROM chunks WHERE chunk_type = 'qa_pair' "
        "AND content LIKE '%\u067e\u0627\u0633\u062e%'"
    )
    qa_with_answer = cur.fetchone()[0]

    # Parent chunks
    cur.execute("SELECT COUNT(*) FROM chunks WHERE parent_id IS NOT NULL")
    with_parent = cur.fetchone()[0]

    table = Table(title="Chunk Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for ct, cnt in type_counts:
        table.add_row(f"Type: {ct}", str(cnt))

    table.add_row("---", "---")
    table.add_row("Min tokens", str(token_stats[0]))
    table.add_row("Max tokens", str(token_stats[1]))
    table.add_row("Avg tokens", f"{token_stats[2]:.0f}")
    table.add_row("QA with answer", f"{qa_with_answer}/{qa_total}")
    table.add_row("Chunks with parent", str(with_parent))

    console.print(table)
    conn.close()


if __name__ == "__main__":
    main()
