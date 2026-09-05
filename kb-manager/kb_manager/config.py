"""Central configuration loaded from env / config files."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    name: str = "kb_manager"
    user: str = "postgres"
    password: str = "postgres"
    echo: bool = False
    url_override: str = ""
    mode: str = "sqlite"  # sqlite (light) | pgvector (postgres)
    sqlite_path: str = "./data/kb_test.db"
    _sqlite_default: bool = False

    @property
    def async_url(self) -> str:
        if self.url_override:
            return self.url_override
        if self.mode == "pgvector":
            return (
                f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
            )
        # sqlite mode — ensure aiosqlite URL
        path = self.sqlite_path
        if path.startswith("sqlite"):
            return path
        # absolute path
        if path.startswith("/"):
            return f"sqlite+aiosqlite:///{path}"
        # relative path — keep ./ prefix for consistency with previous default
        clean = path.lstrip("./")
        return f"sqlite+aiosqlite:///./{clean}"

    @property
    def sync_url(self) -> str:
        url = self.async_url
        if "aiosqlite" in url:
            return url.replace("sqlite+aiosqlite", "sqlite")
        if "asyncpg" in url:
            return url.replace("postgresql+asyncpg", "postgresql+psycopg2")
        return url.replace("sqlite+aiosqlite", "sqlite").replace(
            "postgresql+asyncpg", "postgresql+psycopg2"
        )

    @property
    def is_sqlite(self) -> bool:
        return "sqlite" in self.async_url

    @property
    def is_pgvector(self) -> bool:
        return not self.is_sqlite

    @property
    def driver(self) -> str:
        if "aiosqlite" in self.async_url:
            return "aiosqlite"
        if "asyncpg" in self.async_url:
            return "asyncpg"
        return "unknown"


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    dimensions: int = 384
    batch_size: int = 64
    device: str = "cpu"
    normalize: bool = True


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str = "semantic"
    max_tokens: int = 512
    min_tokens: int = 100
    overlap_tokens: int = 50
    parent_max_tokens: int = 1536
    parent_scope: str = "sheet"  # "sheet" or "document"
    dedup_questions: bool = True  # keep one canonical chunk per normalized question


@dataclass(frozen=True)
class ParserConfig:
    xlsx_engine: str = "auto"  # "auto" | "openpyxl" | "calamine"


@dataclass(frozen=True)
class HyDEConfig:
    """Configuration for HyDE (Hypothetical Document Embeddings)."""

    enabled: bool = False
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_base_url: str = ""
    num_hypotheses: int = 1
    prompt_template: str = ""  # empty = use default Persian template


@dataclass(frozen=True)
class RagasConfig:
    """Configuration for RAGAS LLM-based evaluation."""

    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    api_key: str = ""
    base_url: str = ""
    metrics: tuple[str, ...] = ("faithfulness", "answer_relevancy", "context_recall")
    k: int = 5


@dataclass(frozen=True)
class AppConfig:
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)
    hyde: HyDEConfig = field(default_factory=HyDEConfig)
    ragas: RagasConfig = field(default_factory=RagasConfig)
    source_dir: str = str(PROJECT_ROOT / "kb-source")
    output_dir: str = str(PROJECT_ROOT / "data" / "processed")
    web_host: str = "0.0.0.0"
    web_port: int = 8000


def _resolve_db_mode_and_url() -> tuple[str, str, str]:
    """Resolve KB_DB_MODE and URL.

    Priority:
      1. If KB_DB_URL explicitly set in env -> use it as url_override (mode inferred from URL).
      2. Else if KB_DB_MODE set -> derive URL from mode.
      3. Else default to sqlite/light (``./data/kb_test.db``).

    Aliases: light/sqlite/sqlite3 -> sqlite,  pg/pgvector/postgres/postgresql -> pgvector
    Returns: (mode, url_override, sqlite_path)
    """
    raw_mode = os.getenv("KB_DB_MODE", "").strip().lower()
    # Also check legacy KB_USE_PGVECTOR flag
    legacy_pg = os.getenv("KB_USE_PGVECTOR", "").strip().lower()
    if legacy_pg in ("1", "true", "yes", "on") and not raw_mode:
        raw_mode = "pgvector"

    explicit_url = os.getenv("KB_DB_URL")  # None if not set

    # Normalize mode
    sqlite_aliases = {"", "light", "sqlite", "sqlite3", "test", "local"}
    pg_aliases = {"pg", "postgres", "postgresql", "pgvector", "production", "prod"}

    if explicit_url is not None and not raw_mode:
        # Infer mode from explicit URL if user didn't set mode
        if "sqlite" in explicit_url:
            mode = "sqlite"
        elif "postgres" in explicit_url or "asyncpg" in explicit_url or "psycopg2" in explicit_url:
            mode = "pgvector"
        else:
            mode = "sqlite"
        return mode, explicit_url, os.getenv("KB_SQLITE_PATH", "./data/kb_test.db")

    # Mode explicitly set (or default)
    if not raw_mode:
        raw_mode = "sqlite"
    if raw_mode in sqlite_aliases:
        mode = "sqlite"
    elif raw_mode in pg_aliases:
        mode = "pgvector"
    else:
        mode = "sqlite"

    # If explicit URL also set together with mode, explicit URL wins only if driver matches mode
    if explicit_url is not None:
        is_sqlite_url = "sqlite" in explicit_url
        if (mode == "sqlite" and is_sqlite_url) or (mode == "pgvector" and not is_sqlite_url):
            return mode, explicit_url, os.getenv("KB_SQLITE_PATH", "./data/kb_test.db")
        # Mismatch: mode takes precedence, ignore explicit URL that doesn't match mode
        # (user likely changed mode but left old KB_DB_URL in env)
        pass

    sqlite_path = os.getenv("KB_SQLITE_PATH", os.getenv("KB_DB_SQLITE_PATH", "./data/kb_test.db"))
    # Allow KB_DB_URL to also serve as sqlite path if it looks like a path (legacy)
    return mode, "", sqlite_path


def load_config() -> AppConfig:
    """Load config from environment variables with sensible defaults."""
    _db_mode, _db_url_override, _sqlite_path = _resolve_db_mode_and_url()
    # F27 fix: prefer kb-source submodule if it exists (check both kb-manager/kb-source and ../kb-source), fallback to data
    _kb_source_candidates = [PROJECT_ROOT / "kb-source", PROJECT_ROOT.parent / "kb-source"]
    _default_source = next((str(p) for p in _kb_source_candidates if p.exists()), str(PROJECT_ROOT / "data"))
    return AppConfig(
        db=DatabaseConfig(
            host=os.getenv("KB_DB_HOST", "localhost"),
            port=int(os.getenv("KB_DB_PORT", "5432")),
            name=os.getenv("KB_DB_NAME", "kb_manager"),
            user=os.getenv("KB_DB_USER", "postgres"),
            password=os.getenv("KB_DB_PASSWORD", "postgres"),
            echo=os.getenv("KB_DB_ECHO", "false").lower() == "true",
            url_override=_db_url_override,
            mode=_db_mode,
            sqlite_path=_sqlite_path,
        ),
        embedding=EmbeddingConfig(
            model_name=os.getenv(
                "KB_EMBED_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            dimensions=int(os.getenv("KB_EMBED_DIM", "384")),
            batch_size=int(os.getenv("KB_EMBED_BATCH", "64")),
        ),
        chunking=ChunkingConfig(
            strategy=os.getenv("KB_CHUNK_STRATEGY", "semantic"),
            max_tokens=int(os.getenv("KB_CHUNK_MAX", "512")),
            parent_max_tokens=int(os.getenv("KB_CHUNK_PARENT_MAX", "1536")),
            parent_scope=os.getenv("KB_CHUNK_PARENT_SCOPE", "sheet"),
            dedup_questions=os.getenv("KB_CHUNK_DEDUP_QUESTIONS", "true").lower() == "true",
        ),
        parser=ParserConfig(
            xlsx_engine=os.getenv("KB_XLSX_ENGINE", "auto"),
        ),
        hyde=HyDEConfig(
            enabled=os.getenv("KB_HYDE_ENABLED", "false").lower() == "true",
            llm_model=os.getenv("KB_HYDE_LLM", "gpt-4o-mini"),
            llm_api_key=os.getenv("KB_HYDE_API_KEY", os.getenv("OPENAI_API_KEY", "")),
            llm_base_url=os.getenv("KB_HYDE_BASE_URL", os.getenv("OPENAI_BASE_URL", "")),
            num_hypotheses=int(os.getenv("KB_HYDE_NUM", "1")),
        ),
        ragas=RagasConfig(
            llm_model=os.getenv("KB_RAGAS_LLM", "gpt-4o-mini"),
            embedding_model=os.getenv("KB_RAGAS_EMBED", "text-embedding-3-small"),
            api_key=os.getenv("KB_RAGAS_API_KEY", ""),
            base_url=os.getenv("KB_RAGAS_BASE_URL", ""),
            k=int(os.getenv("KB_RAGAS_K", "5")),
        ),
        source_dir=os.getenv("KB_SOURCE_DIR", _default_source),
        output_dir=os.getenv("KB_OUTPUT_DIR", str(PROJECT_ROOT / "data" / "processed")),
        web_host=os.getenv("KB_WEB_HOST", "0.0.0.0"),
        web_port=int(os.getenv("KB_WEB_PORT", "8000")),
    )
