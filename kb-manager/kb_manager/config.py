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

    @property
    def async_url(self) -> str:
        if self.url_override:
            return self.url_override
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
        )

    @property
    def sync_url(self) -> str:
        if self.url_override:
            return self.url_override
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
        )

    @property
    def is_sqlite(self) -> bool:
        return "sqlite" in self.async_url

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


@dataclass(frozen=True)
class AppConfig:
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    source_dir: str = str(PROJECT_ROOT / "data")
    output_dir: str = str(PROJECT_ROOT / "data" / "processed")
    web_host: str = "0.0.0.0"
    web_port: int = 8000


def load_config() -> AppConfig:
    """Load config from environment variables with sensible defaults."""
    db_url = os.getenv("KB_DB_URL", "")
    return AppConfig(
        db=DatabaseConfig(
            host=os.getenv("KB_DB_HOST", "localhost"),
            port=int(os.getenv("KB_DB_PORT", "5432")),
            name=os.getenv("KB_DB_NAME", "kb_manager"),
            user=os.getenv("KB_DB_USER", "postgres"),
            password=os.getenv("KB_DB_PASSWORD", "postgres"),
            echo=os.getenv("KB_DB_ECHO", "false").lower() == "true",
            url_override=db_url,
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
        ),
        source_dir=os.getenv("KB_SOURCE_DIR", str(PROJECT_ROOT / "data")),
        output_dir=os.getenv("KB_OUTPUT_DIR", str(PROJECT_ROOT / "data" / "processed")),
        web_host=os.getenv("KB_WEB_HOST", "0.0.0.0"),
        web_port=int(os.getenv("KB_WEB_PORT", "8000")),
    )
