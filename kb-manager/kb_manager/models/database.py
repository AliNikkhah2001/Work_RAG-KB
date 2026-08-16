"""Database engine, ORM models, and session management."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from kb_manager.config import DatabaseConfig


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


def _json_col(name: str, default: Any = None, nullable: bool = False) -> Column:
    """Create a JSON column (uses JSONB for PostgreSQL, JSON for SQLite)."""
    return Column(name, JSON, nullable=nullable, default=default)


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=_uuid)
    source_path = Column(String(1024), nullable=False)
    title = Column(String(512), nullable=False)
    domain = Column(String(64), nullable=False, default="general")
    category = Column(String(64), nullable=False, default="article")
    status = Column(String(32), nullable=False, default="draft")
    version = Column(Integer, nullable=False, default=1)
    source_hash = Column(String(64), nullable=False, default="")
    content_hash = Column(String(64), nullable=False, default="")
    chunk_count = Column(Integer, nullable=False, default=0)
    doc_metadata = Column("metadata", JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    versions = relationship(
        "DocumentVersion", back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_documents_domain", "domain"),
        Index("ix_documents_category", "category"),
        Index("ix_documents_status", "status"),
        Index("ix_documents_source_hash", "source_hash"),
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(String(36), primary_key=True, default=_uuid)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    parent_id = Column(String(36), ForeignKey("chunks.id", ondelete="SET NULL"), nullable=True)
    ordinal = Column(Integer, nullable=False)
    chunk_type = Column(String(32), nullable=False, default="body")
    content = Column(Text, nullable=False)
    heading_path = Column(String(1024), nullable=False, default="")
    keywords = Column(JSON, nullable=False, default=list)
    token_count = Column(Integer, nullable=False, default=0)
    embedding_model = Column(String(128), nullable=True)
    quality_score = Column(Float, nullable=True)
    is_verified = Column(Boolean, nullable=False, default=False)
    doc_metadata = Column("metadata", JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    document = relationship("Document", back_populates="chunks")
    parent = relationship("Chunk", remote_side=[id], backref="children")

    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_ordinal", "document_id", "ordinal"),
        Index("ix_chunks_parent_id", "parent_id"),
    )


class DocumentVersion(Base):
    __tablename__ = "document_versions"

    id = Column(String(36), primary_key=True, default=_uuid)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    version = Column(Integer, nullable=False)
    content_hash = Column(String(64), nullable=False)
    change_summary = Column(Text, nullable=True)
    changed_by = Column(String(128), nullable=True)
    chunk_count = Column(Integer, nullable=True)
    status = Column(String(32), nullable=False, default="snapshot")
    snapshot_data = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    document = relationship("Document", back_populates="versions")

    __table_args__ = (
        Index("ix_docversions_document_id", "document_id"),
        Index("ix_docversions_version", "document_id", "version"),
    )


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id = Column(String(36), primary_key=True, default=_uuid)
    job_type = Column(String(32), nullable=False, default="incremental")
    status = Column(String(32), nullable=False, default="pending")
    source_dir = Column(String(1024), nullable=True)
    documents_total = Column(Integer, nullable=False, default=0)
    documents_ok = Column(Integer, nullable=False, default=0)
    documents_failed = Column(Integer, nullable=False, default=0)
    chunks_total = Column(Integer, nullable=False, default=0)
    error_log = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class RetrievalLog(Base):
    __tablename__ = "retrieval_logs"

    id = Column(String(36), primary_key=True, default=_uuid)
    query = Column(Text, nullable=False)
    results_count = Column(Integer, nullable=False, default=0)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class Database:
    """Manages async/sync engines and session factories."""

    def __init__(self, config: DatabaseConfig) -> None:
        self._config = config
        self._async_engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    @property
    def async_engine(self) -> AsyncEngine:
        if self._async_engine is None:
            connect_args: dict[str, Any] = {}
            if self._config.is_sqlite:
                connect_args["check_same_thread"] = False
            self._async_engine = create_async_engine(
                self._config.async_url,
                echo=self._config.echo,
                connect_args=connect_args,
            )
        return self._async_engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is None:
            self._session_factory = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
        return self._session_factory

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def create_tables(self) -> None:
        """Create all tables (for dev/test)."""
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_tables(self) -> None:
        """Drop all tables (for test teardown)."""
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def close(self) -> None:
        if self._async_engine:
            await self._async_engine.dispose()

    def get_sync_engine(self):
        """Get a synchronous engine for scripts."""
        return create_engine(self._config.sync_url, echo=self._config.echo)

    def run_sync(self, fn):
        """Run a function with a sync session."""
        engine = self.get_sync_engine()
        try:
            with Session(engine) as session:
                return fn(session)
        finally:
            engine.dispose()
