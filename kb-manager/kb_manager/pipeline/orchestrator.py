"""Main ingestion pipeline orchestrator."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from sqlalchemy import select

from kb_manager.models.database import Chunk as DBChunk
from kb_manager.models.database import Document, IngestionJob
from kb_manager.parsers.registry import get_parser
from kb_manager.pipeline.quality import QualityGate, QualityThresholds
from kb_manager.pipeline.versioning import VersionManager, compute_content_hash
from kb_manager.preprocessor.pipeline import PreprocessingPipeline, PreprocessingResult

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from kb_manager.chunker.base import BaseChunker
    from kb_manager.embedder.base import BaseEmbedder
    from kb_manager.parsers.base import BaseParser, ParsedDocument

from kb_manager.chunker.base import Chunk

logger = logging.getLogger(__name__)


@dataclass
class PipelineSummary:
    """Aggregated results returned after a pipeline run."""

    job_id: str
    job_type: str
    documents_processed: int = 0
    documents_created: int = 0
    documents_updated: int = 0
    documents_skipped: int = 0
    documents_failed: int = 0
    chunks_created: int = 0
    chunks_embedded: int = 0
    chunks_skipped_incomplete: int = 0
    versions_created: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "documents_processed": self.documents_processed,
            "documents_created": self.documents_created,
            "documents_updated": self.documents_updated,
            "documents_skipped": self.documents_skipped,
            "documents_failed": self.documents_failed,
            "chunks_created": self.chunks_created,
            "chunks_embedded": self.chunks_embedded,
            "chunks_skipped_incomplete": self.chunks_skipped_incomplete,
            "versions_created": self.versions_created,
            "errors": self.errors,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


class PipelineOrchestrator:
    """Coordinates parsing, preprocessing, chunking, embedding, and storage.

    Parameters
    ----------
    database:
        A :class:`kb_manager.models.database.Database` instance.
    parsers:
        Mapping of file extension to parser.  If ``None``, the global
        registry is used.
    preprocessor:
        Preprocessing pipeline.  A default instance is created when *None*.
    chunker:
        Chunking strategy.  A default instance is created when *None*.
    embedder:
        Text embedder.  A default instance is created when *None*.
    quality_thresholds:
        Optional custom quality thresholds.
    """

    _SUPPORTED_EXTENSIONS: ClassVar[frozenset[str]] = frozenset({".xlsx", ".pdf", ".docx"})

    def __init__(
        self,
        database: Any,
        parsers: dict[str, BaseParser] | None = None,
        preprocessor: PreprocessingPipeline | None = None,
        chunker: BaseChunker | None = None,
        embedder: BaseEmbedder | None = None,
        quality_thresholds: QualityThresholds | None = None,
    ) -> None:
        self._db = database
        self._parsers = parsers
        self._preprocessor = preprocessor or PreprocessingPipeline()
        self._chunker = chunker
        self._embedder = embedder
        self._quality_gate = QualityGate(quality_thresholds)
        self._version_mgr = VersionManager()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_full_rebuild(self, source_dir: str) -> PipelineSummary:
        """Full rebuild: parse all files, re-chunk, re-embed, store.

        1. Scan *source_dir* for supported files.
        2. For each file: parse -> preprocess -> chunk -> embed -> store.
        3. Create version snapshots for every processed document.
        4. Return a summary with counts.

        Args:
            source_dir: Root directory to scan for source files.

        Returns:
            :class:`PipelineSummary` with aggregated stats.
        """
        start = time.monotonic()
        job = await self._create_job("full_rebuild", source_dir)

        files = self._scan_files(source_dir)
        summary = PipelineSummary(job_id=job.id, job_type="full_rebuild")
        summary.documents_processed = len(files)

        # F36 fix: reset dedup state at start of rebuild so queries are isolated per job
        if self._chunker is not None and hasattr(self._chunker, "reset_dedup"):
            self._chunker.reset_dedup()

        async with self._db.session() as session:
            for file_path in files:
                try:
                    result = await self._process_file(file_path, session, force=True)
                    summary.documents_created += result["created"]
                    summary.documents_updated += result["updated"]
                    summary.chunks_created += result["chunks"]
                    summary.chunks_embedded += result["embedded"]
                    summary.chunks_skipped_incomplete += result.get("chunks_skipped_incomplete", 0)
                    summary.versions_created += result["versions"]
                except Exception:
                    logger.exception("Failed to process %s", file_path)
                    summary.documents_failed += 1
                    summary.errors.append(
                        {
                            "file": file_path,
                            "error": "Processing failed (see logs for details)",
                        }
                    )

            await self._finalize_job(job, summary, session, status="completed")

        summary.elapsed_seconds = time.monotonic() - start
        logger.info("Full rebuild finished: %s", summary.to_dict())
        return summary

    async def run_incremental(self, source_dir: str) -> PipelineSummary:
        """Incremental ingestion: only process changed or new files.

        Uses ``content_hash`` comparison to detect changes.

        Args:
            source_dir: Root directory to scan for source files.

        Returns:
            :class:`PipelineSummary` with aggregated stats.
        """
        start = time.monotonic()
        job = await self._create_job("incremental", source_dir)

        files = self._scan_files(source_dir)
        summary = PipelineSummary(job_id=job.id, job_type="incremental")
        summary.documents_processed = len(files)

        if self._chunker is not None and hasattr(self._chunker, "reset_dedup"):
            self._chunker.reset_dedup()

        async with self._db.session() as session:
            for file_path in files:
                try:
                    result = await self._process_file(file_path, session, force=False)
                    if result["skipped"]:
                        summary.documents_skipped += 1
                    else:
                        summary.documents_created += result["created"]
                        summary.documents_updated += result["updated"]
                        summary.chunks_created += result["chunks"]
                        summary.chunks_embedded += result["embedded"]
                        summary.chunks_skipped_incomplete += result.get("chunks_skipped_incomplete", 0)
                        summary.versions_created += result["versions"]
                except Exception:
                    logger.exception("Failed to process %s", file_path)
                    summary.documents_failed += 1
                    summary.errors.append(
                        {
                            "file": file_path,
                            "error": "Processing failed (see logs for details)",
                        }
                    )

            await self._finalize_job(job, summary, session, status="completed")

        summary.elapsed_seconds = time.monotonic() - start
        logger.info("Incremental run finished: %s", summary.to_dict())
        return summary

    # ------------------------------------------------------------------
    # File scanning
    # ------------------------------------------------------------------

    def _scan_files(self, source_dir: str) -> list[str]:
        """Recursively collect all supported files under *source_dir*."""
        root = Path(source_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

        files: list[str] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.name.startswith("~$"):
                continue  # Microsoft Office temporary lock files
            # Skip test-question datasets — they are evaluation data, not KB content
            if any(seg.startswith("TestQuestion") for seg in path.parts):
                continue
            if path.suffix.lower() in self._SUPPORTED_EXTENSIONS:
                files.append(str(path.resolve()))
        files.sort()
        return files

    # ------------------------------------------------------------------
    # Per-file processing
    # ------------------------------------------------------------------

    async def _process_file(
        self,
        file_path: str,
        session: AsyncSession,
        *,
        force: bool = False,
    ) -> dict[str, int]:
        """Parse, preprocess, chunk, embed, and store a single file.

        When *force* is False the file is skipped if its content hash
        matches the existing document in the database.

        Returns a dict of counters:
        ``created``, ``updated``, ``skipped``, ``chunks``, ``embedded``,
        ``versions``.
        """
        result: dict[str, int] = {
            "created": 0,
            "updated": 0,
            "skipped": 0,
            "chunks": 0,
            "embedded": 0,
            "versions": 0,
        }

        # --- Parse ---
        parsed = self._parse_file(file_path)

        # --- Content hash ---
        content_hash = compute_content_hash(parsed.content)

        # --- Existing doc lookup ---
        existing = await self._find_document_by_path(file_path, session)

        if existing is not None:
            if not force and existing.content_hash == content_hash:
                logger.debug("Skipping unchanged file: %s", file_path)
                result["skipped"] = 1
                return result
            doc = existing
            doc.version += 1
            doc.content_hash = content_hash
            doc.source_hash = content_hash
            doc.title = parsed.title
            doc.doc_metadata = parsed.metadata
            doc.updated_at = datetime.now(UTC)
            is_new = False
        else:
            doc = Document(
                source_path=file_path,
                title=parsed.title,
                domain=parsed.metadata.get("domain", "general"),
                category=parsed.metadata.get("category", "article"),
                status="indexed",
                source_hash=content_hash,
                content_hash=content_hash,
                doc_metadata=parsed.metadata,
            )
            session.add(doc)
            await session.flush()
            is_new = True

        # --- Preprocess ---
        prep: PreprocessingResult = self._preprocessor.run(parsed.content)

        # --- Determine doc_type from schema detection ---
        doc_type = "body"
        if parsed.sheets:
            for sheet in parsed.sheets:
                if sheet.get("schema"):
                    schema_map = {
                        "reason_codes": "reason_detail",
                        "crm_qa": "qa_pair",
                        "articles": "article",
                    }
                    doc_type = schema_map.get(sheet["schema"], "body")
                    break

        # --- Chunk ---
        chunks = self._chunker.chunk(
            prep.normalised_text,
            metadata={
                "document_id": doc.id,
                "doc_type": doc_type,
                "sheets": parsed.sheets,
                "parent_scope": getattr(self._chunker, "parent_scope", "sheet"),
                "parent_max_tokens": getattr(self._chunker, "parent_max_tokens", 1536),
            },
        )

        # Capture skipped incomplete QA rows
        if hasattr(self._chunker, "get_skipped_incomplete"):
            result["chunks_skipped_incomplete"] = self._chunker.get_skipped_incomplete()

        if not chunks:
            logger.warning("No chunks produced for %s", file_path)
            doc.status = "draft"
            doc.chunk_count = 0
            result["updated" if not is_new else "created"] = 1
            return result

        # --- Delete old chunks if updating ---
        if not is_new:
            old_chunks = await session.execute(select(DBChunk).where(DBChunk.document_id == doc.id))
            for old in old_chunks.scalars().all():
                await session.delete(old)
            await session.flush()

        # F6 fix: SQLite has no vector column; dense index is rebuilt at search from .npz cache.
        # Do not claim embeddings are persisted. Dense rebuild is lazy via search._build_index.
        result["embedded"] = 0

        # --- Store chunks (F28 fix: O(1) parent key map, no ordinal/type scan) ---
        parents = [c for c in chunks if c.metadata.get("is_parent")]
        children = [c for c in chunks if not c.metadata.get("is_parent")]
        parent_id_map: dict[str, str] = {}

        # Store parents first and capture their DB ids
        parent_db_chunks: list[DBChunk] = []
        for pc in parents:
            db_chunk = DBChunk(
                document_id=doc.id,
                ordinal=pc.ordinal,
                chunk_type=pc.chunk_type,
                content=pc.content,
                heading_path=pc.heading_path,
                keywords=pc.keywords,
                token_count=pc.token_count,
                embedding_model=self._embedder.model_name if self._embedder else None,
                doc_metadata=pc.metadata,
            )
            parent_db_chunks.append(db_chunk)
            session.add(db_chunk)
        if parent_db_chunks:
            await session.flush()
            for pc, dbc in zip(parents, parent_db_chunks):
                key = pc.metadata.get("parent_key", pc.metadata.get("sheet_name", ""))
                parent_id_map[key] = dbc.id

        # Store children with correct parent_id in one pass
        for cc in children:
            key = cc.metadata.get("parent_key", cc.metadata.get("sheet_name", ""))
            parent_id = parent_id_map.get(key)
            db_chunk = DBChunk(
                document_id=doc.id,
                parent_id=parent_id,
                ordinal=cc.ordinal,
                chunk_type=cc.chunk_type,
                content=cc.content,
                heading_path=cc.heading_path,
                keywords=cc.keywords,
                token_count=cc.token_count,
                embedding_model=self._embedder.model_name if self._embedder else None,
                doc_metadata=cc.metadata,
            )
            session.add(db_chunk)
        if children or parents:
            await session.flush()
        # Also store parent chunks that were already flushed are in DB; no second O(n²) scan needed
        # For completeness, ensure parents that are also in children list? No, already handled.

        doc.chunk_count = len(chunks)
        result["chunks"] = len(chunks)

        # --- Quality gate ---
        chunk_dtos = [
            Chunk(
                content=c.content,
                ordinal=c.ordinal,
                chunk_type=c.chunk_type,
                heading_path=c.heading_path,
                keywords=c.keywords,
                token_count=c.token_count,
            )
            for c in chunks
        ]
        validation = self._quality_gate.validate_document(doc, chunk_dtos)
        if not validation["valid"]:
            logger.warning(
                "Document %s failed quality validation: %s",
                file_path,
                validation["errors"],
            )

        # --- Version snapshot ---
        try:
            await self._version_mgr.create_snapshot(
                doc.id,
                session,
                change_summary=f"{'Initial indexing' if is_new else 'Updated content'}",
                changed_by="pipeline",
            )
            result["versions"] = 1
        except Exception:
            logger.exception("Failed to create version snapshot for %s", file_path)

        if is_new:
            result["created"] = 1
        else:
            result["updated"] = 1

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_file(self, file_path: str) -> ParsedDocument:
        """Parse a file using the appropriate parser."""
        if self._parsers is not None:
            ext = Path(file_path).suffix.lower()
            parser = self._parsers.get(ext)
            if parser is None:
                raise ValueError(f"No parser registered for extension: {ext}")
        else:
            parser = get_parser(file_path)
        return parser.parse(file_path)

    async def _find_document_by_path(
        self,
        source_path: str,
        session: AsyncSession,
    ) -> Document | None:
        """Find an existing document by its source file path."""
        result = await session.execute(select(Document).where(Document.source_path == source_path))
        return result.scalar_one_or_none()

    async def _create_job(self, job_type: str, source_dir: str) -> IngestionJob:
        """Create and persist an ingestion job record."""
        async with self._db.session() as session:
            job = IngestionJob(
                job_type=job_type,
                status="running",
                source_dir=source_dir,
                started_at=datetime.now(UTC),
            )
            session.add(job)
            await session.flush()
            job_id = job.id
            await session.commit()
        # Re-fetch outside the context manager so the id is accessible
        async with self._db.session() as session:
            return await session.get(IngestionJob, job_id)  # type: ignore[return-value]

    async def _finalize_job(
        self,
        job: IngestionJob,
        summary: PipelineSummary,
        session: AsyncSession,
        *,
        status: str = "completed",
    ) -> None:
        """Update the ingestion job with final stats."""
        job.status = status
        job.documents_total = summary.documents_processed
        job.documents_ok = summary.documents_created + summary.documents_updated
        job.documents_failed = summary.documents_failed
        job.chunks_total = summary.chunks_created
        job.completed_at = datetime.now(UTC)

        if summary.errors:
            import json

            job.error_log = json.dumps(summary.errors, ensure_ascii=False)

        await session.flush()
