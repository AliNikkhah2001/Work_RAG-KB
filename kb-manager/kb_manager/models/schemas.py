"""Pydantic schemas for API request/response validation."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    DRAFT = "draft"
    INDEXED = "indexed"
    ACTIVE = "active"
    STALE = "stale"
    ARCHIVED = "archived"


class Domain(str, Enum):
    INDIVIDUAL = "individual"
    CORPORATE = "corporate"
    CHEQUE = "cheque"
    GENERAL = "general"
    API = "api"
    REGULATORY = "regulatory"


class Category(str, Enum):
    REASON_CODE = "reason_code"
    QA_PAIR = "qa_pair"
    ARTICLE = "article"
    MODEL_CONTENT = "model_content"
    REGULATION = "regulation"
    COMPANY_INFO = "company_info"
    REFERENCE = "reference"


class ChunkType(str, Enum):
    SEMANTIC = "semantic"
    QA_PAIR = "qa_pair"
    REASON_DETAIL = "reason_detail"
    BODY = "body"
    HEADER = "header"


# --- Document Schemas ---


class DocumentCreate(BaseModel):
    source_path: str
    title: str
    domain: Domain
    category: Category
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentRead(BaseModel):
    id: str
    source_path: str
    title: str
    domain: Domain
    category: Category
    status: DocumentStatus
    version: int
    source_hash: str
    content_hash: str
    chunk_count: int = 0
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentList(BaseModel):
    total: int
    items: list[DocumentRead]


# --- Chunk Schemas ---


class ChunkCreate(BaseModel):
    document_id: str
    ordinal: int
    chunk_type: ChunkType
    content: str
    heading_path: str = ""
    keywords: list[str] = Field(default_factory=list)
    token_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkRead(BaseModel):
    id: str
    document_id: str
    ordinal: int
    chunk_type: ChunkType
    content: str
    heading_path: str
    keywords: list[str]
    token_count: int
    embedding_model: str | None = None
    quality_score: float | None = None
    is_verified: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ChunkList(BaseModel):
    total: int
    items: list[ChunkRead]


# --- Version Schemas ---


class VersionRead(BaseModel):
    id: str
    document_id: str
    version: int
    content_hash: str
    change_summary: str | None = None
    changed_by: str | None = None
    chunk_count: int | None = None
    status: str
    created_at: datetime


class VersionList(BaseModel):
    total: int
    items: list[VersionRead]


# --- Pipeline Schemas ---


class PipelineRunRequest(BaseModel):
    rebuild_type: str = "incremental"  # full_rebuild or incremental
    source_dir: str | None = None
    embedding_model: str | None = None


class PipelineRunResponse(BaseModel):
    job_id: str
    status: str
    message: str


class PipelineStatus(BaseModel):
    job_id: str
    status: str
    documents_total: int = 0
    documents_ok: int = 0
    documents_failed: int = 0
    chunks_total: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_log: str | None = None


# --- Monitoring Schemas ---


class HealthResponse(BaseModel):
    status: str
    database: str
    version: str
    documents_count: int = 0
    chunks_count: int = 0


class StalenessReport(BaseModel):
    total_active: int
    stale_count: int
    stale_percentage: float
    by_category: list[dict[str, Any]]


class MetricsSummary(BaseModel):
    total_documents: int
    total_chunks: int
    documents_by_domain: dict[str, int]
    documents_by_category: dict[str, int]
    average_chunks_per_document: float


# --- Search Schemas ---


class SearchRequest(BaseModel):
    query: str
    domain: Domain | None = None
    category: Category | None = None
    top_k: int = Field(default=5, ge=1, le=50)


class SearchResult(BaseModel):
    chunk_id: str
    content: str
    heading_path: str
    document_title: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total: int
