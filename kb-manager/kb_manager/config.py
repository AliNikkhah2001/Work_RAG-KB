"""Central configuration loaded from env / config files / YAML."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"


# ---------------------------------------------------------------------------
# Helper: load YAML with env var interpolation
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file, return empty dict if not found."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Interpolate ${ENV_VAR} or ${ENV_VAR:-default}
    import re
    def replace_env(match: re.Match) -> str:
        expr = match.group(1)
        if ":-" in expr:
            var, default = expr.split(":-", 1)
            return os.getenv(var, default)
        return os.getenv(expr, "")
    def interpolate(obj: Any) -> Any:
        if isinstance(obj, str):
            return re.sub(r"\$\{([^}]+)\}", replace_env, obj)
        if isinstance(obj, dict):
            return {k: interpolate(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [interpolate(v) for v in obj]
        return obj
    return interpolate(data)


def _merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Deep merge multiple config dicts (later overrides earlier)."""
    result: dict[str, Any] = {}
    for cfg in configs:
        for k, v in cfg.items():
            if isinstance(v, dict) and k in result and isinstance(result[k], dict):
                result[k] = _merge_configs(result[k], v)
            else:
                result[k] = v
    return result


# ---------------------------------------------------------------------------
# Config Dataclasses
# ---------------------------------------------------------------------------

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
    # Advanced
    use_context: bool = True          # Anthropic-style contextual retrieval
    context_template: str = "Title: {title}\nHeading: {heading}\nContent: {content}"
    cache_dir: str = "data/embeddings_cache"


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str = "semantic"        # "semantic" | "fixed"
    max_tokens: int = 512
    min_tokens: int = 100
    overlap_tokens: int = 50
    parent_max_tokens: int = 1536
    parent_scope: str = "sheet"       # "sheet" | "document"
    dedup_questions: bool = False     # keep one canonical chunk per normalized question
    # Per-type overrides (optional, loaded from YAML)
    type_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class ParserConfig:
    xlsx_engine: str = "auto"         # "auto" | "openpyxl" | "calamine"
    pdf_engine: str = "pymupdf"       # "pymupdf" | "pdfplumber"
    docx_engine: str = "python-docx"


@dataclass(frozen=True)
class RagasConfig:
    """Configuration for RAGAS LLM-based evaluation."""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    api_key: str = ""
    base_url: str = ""
    metrics: tuple[str, ...] = ("faithfulness", "answer_relevancy", "context_recall", "context_precision")
    k: int = 5


# ---------------------------------------------------------------------------
# NEW: Advanced Retrieval Configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalConfig:
    """Hybrid retrieval pipeline configuration."""
    # Weights
    bm25_weight: float = 0.3
    dense_weight: float = 0.7
    # RRF
    rrf_k: int = 60
    rrf_enabled: bool = True
    # Candidate pool sizes
    bm25_candidates: int = 50
    dense_candidates: int = 50
    rerank_candidates: int = 20
    final_top_k: int = 10
    # Adaptive strategy selection by detected query type
    adaptive_enabled: bool = True
    hyde_enabled: bool = True
    multi_query_enabled: bool = True
    # Query-type adaptive weights (loaded from YAML)
    adaptive_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    # Strategy presets
    strategies: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class RerankerConfig:
    """Cross-encoder reranker configuration."""
    model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    batch_size: int = 32
    max_length: int = 512
    device: str = ""                  # "" = auto (cuda if available else cpu)
    enabled: bool = True
    threshold: float = 0.0            # minimum rerank score to keep


@dataclass(frozen=True)
class HyDEConfig:
    """Hypothetical Document Embeddings (HyDE) configuration."""
    enabled: bool = True
    model_name: str = "gpt-4o-mini"   # LLM for generating pseudo-doc
    prompt_template: str = (
        "<start_of_turn>user\n"
        "شما یک متخصص سیستم اعتبارسنجی اعتباری ایران هستید. برای پرسش زیر، یک پاسخ کامل و دقیق بنویسید که حاوی اطلاعات کلیدی برای یافتن اسناد مرتبط باشد.\n\n"
        "پرسش: {query}\n\n"
        "پاسخ فرضی (شامل اصطلاحات کلیدی، مفاهیم تخصصی و جزئیات فنی):\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    max_length: int = 512
    temperature: float = 0.3
    cache_ttl_seconds: int = 3600
    top_k: int = 20                   # candidates from HyDE search


@dataclass(frozen=True)
class MultiQueryConfig:
    """Multi-query rewriting configuration."""
    enabled: bool = True
    model_name: str = "gpt-4o-mini"
    prompt_template: str = (
        "<start_of_turn>user\n"
        "شما یک متخصص سیستم اعتبارسنجی اعتباری ایران هستید. برای پرسش زیر، {num_queries} صيغ مختلف پرسش تولید کنید که کاربر ممکن است بپرسد.\n\n"
        "پرسش اصلی: {query}\n\n"
        "انواع پرسش مورد نیاز:\n"
        "1. Verbatim - بسیار نزدیک به متن اصلی\n"
        "2. Paraphrase - با مترادف‌ها و تغییر ساختار\n"
        "3. Conversational - محاوره‌ای (می‌شه، می‌تونم، یه سوال داشتم)\n"
        "4. Typo - با اشتباهات تایپ واقعی (می‌شود→میشه، دسترسی→دسترسى)\n"
        "5. Keyword-only - فقط اصطلاحات کلیدی تخصصی\n"
        "6. Reworded - حذف کلمات + تغییر ترتیب سنگین\n\n"
        "خروجی JSON (فقط آرایه JSON، بدون متن اضافه):\n"
        "[\n"
        "  {{\"query\": \"...\", \"type\": \"verbatim\"}},\n"
        "  {{\"query\": \"...\", \"type\": \"paraphrase\"}},\n"
        "  {{\"query\": \"...\", \"type\": \"conversational\"}},\n"
        "  {{\"query\": \"...\", \"type\": \"typo\"}},\n"
        "  {{\"query\": \"...\", \"type\": \"keyword_only\"}},\n"
        "  {{\"query\": \"...\", \"type\": \"reworded\"}}\n"
        "]\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    num_queries: int = 6
    beam_size: int = 5
    temperature: float = 0.4
    cache_ttl_seconds: int = 3600
    fusion_method: str = "rrf"        # "rrf" | "weighted_rrf"
    weights: dict[str, float] = field(default_factory=lambda: {
        "verbatim": 1.0,
        "paraphrase": 1.0,
        "conversational": 0.8,
        "typo": 0.9,
        "keyword_only": 0.5,
        "reworded": 0.7,
    })


@dataclass(frozen=True)
class AuthConfig:
    """Authentication & authorization configuration."""
    secret_key: str = ""              # MUST be set via env KB_AUTH_SECRET_KEY
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    password_hash_algorithm: str = "bcrypt"
    # Roles and permissions
    roles: dict[str, list[str]] = field(default_factory=lambda: {
        "admin": ["*"],
        "editor": ["documents:read", "documents:write", "chunks:read", "chunks:write", "versions:read", "versions:write", "editor:*", "monitoring:read"],
        "developer": ["dev:upload", "dev:search", "dev:jobs:read", "api:search"],
        "viewer": ["documents:read", "chunks:read", "versions:read", "monitoring:read", "search:read"],
    })
    default_role: str = "viewer"
    # Session
    session_cookie_name: str = "kb_session"
    session_cookie_secure: bool = True
    session_cookie_httponly: bool = True
    session_cookie_samesite: str = "lax"
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 100
    # API Keys (for developer programmatic access)
    api_key_prefix: str = "kb_"
    api_key_hash_algorithm: str = "bcrypt"


@dataclass(frozen=True)
class MonitoringConfig:
    """Monitoring & observability thresholds."""
    # Retrieval quality
    recall_threshold: float = 0.85
    mrr_threshold: float = 0.60
    ndcg_threshold: float = 0.70
    hit_rate_threshold: float = 0.80
    # Latency (milliseconds)
    retrieval_latency_p95_ms: int = 500
    end_to_end_latency_p95_ms: int = 1500
    bm25_latency_p95_ms: int = 50
    dense_latency_p95_ms: int = 200
    rerank_latency_p95_ms: int = 150
    # Data freshness
    stale_fraction_threshold: float = 0.05
    staleness_check_interval_hours: int = 24
    # Costs
    embedding_cost_per_1k_queries_usd: float = 0.50
    llm_cost_per_1k_queries_usd: float = 1.00
    cost_check_interval_hours: int = 24
    # Alerting
    alert_channels: list[str] = field(default_factory=lambda: ["log"])  # "log", "slack", "email"
    slack_webhook_url: str = ""
    email_smtp_host: str = ""
    email_smtp_port: int = 587
    email_from: str = ""
    email_to: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WebConfig:
    """Web UI configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    language: str = "fa"              # "fa" | "en"
    rtl: bool = True
    theme: str = "auto"               # "light" | "dark" | "auto"
    font_family: str = "Vazirmatn"    # Persian font
    page_size: int = 20
    editor_page_size: int = 50
    # Developer portal
    dev_api_prefix: str = "/api/dev"
    dev_rate_limit_per_minute: int = 60
    # Feature flags
    enable_editor: bool = True
    enable_monitoring: bool = True
    enable_benchmarks: bool = True
    enable_dev_portal: bool = True
    enable_settings_ui: bool = True


@dataclass(frozen=True)
class AppConfig:
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)
    ragas: RagasConfig = field(default_factory=RagasConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    hyde: HyDEConfig = field(default_factory=HyDEConfig)
    multi_query: MultiQueryConfig = field(default_factory=MultiQueryConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    web: WebConfig = field(default_factory=WebConfig)
    source_dir: str = str(PROJECT_ROOT / "data")
    output_dir: str = str(PROJECT_ROOT / "data" / "processed")


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------

def load_config() -> AppConfig:
    """
    Load configuration with precedence:
    1. Default values (dataclass defaults)
    2. configs/default.yaml (base)
    3. configs/<section>.yaml (section-specific)
    4. Environment variables (KB_* prefix)
    
    YAML files support ${ENV_VAR} and ${ENV_VAR:-default} interpolation.
    """
    # Load all YAML configs
    base = _load_yaml(CONFIG_DIR / "default.yaml")
    retrieval_yaml = _load_yaml(CONFIG_DIR / "retrieval.yaml")
    reranker_yaml = _load_yaml(CONFIG_DIR / "reranker.yaml")
    hyde_yaml = _load_yaml(CONFIG_DIR / "hyde.yaml")
    multi_query_yaml = _load_yaml(CONFIG_DIR / "multi_query.yaml")
    auth_yaml = _load_yaml(CONFIG_DIR / "auth.yaml")
    monitoring_yaml = _load_yaml(CONFIG_DIR / "monitoring.yaml")
    web_yaml = _load_yaml(CONFIG_DIR / "web.yaml")
    chunking_yaml = _load_yaml(CONFIG_DIR / "chunking" / "semantic.yaml")  # strategy-specific
    
    # Merge all YAML
    # NOTE: section YAML files already carry their own top-level section key
    # (e.g. retrieval.yaml starts with "retrieval:") so they are merged as-is.
    merged = _merge_configs(
        base,
        retrieval_yaml,
        reranker_yaml,
        hyde_yaml,
        multi_query_yaml,
        auth_yaml,
        monitoring_yaml,
        web_yaml,
        chunking_yaml,
    )
    
    # Helper to get nested value with env override
    def get(path: str, default: Any = None) -> Any:
        keys = path.split(".")
        val = merged
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
            if val is None:
                return default
        # Env override
        env_key = "KB_" + path.upper().replace(".", "_")
        env_val = os.getenv(env_key)
        if env_val is not None:
            # Try to cast to same type as default
            if isinstance(default, bool):
                return env_val.lower() == "true"
            if isinstance(default, int):
                return int(env_val)
            if isinstance(default, float):
                return float(env_val)
            if isinstance(default, list):
                return [v.strip() for v in env_val.split(",")]
            return env_val
        return val if val is not None else default
    
    # Build config objects
    db_cfg = DatabaseConfig(
        host=get("db.host", "localhost"),
        port=get("db.port", 5432),
        name=get("db.name", "kb_manager"),
        user=get("db.user", "postgres"),
        password=get("db.password", "postgres"),
        echo=get("db.echo", False),
        url_override=os.getenv("KB_DB_URL", ""),
    )
    
    embed_cfg = EmbeddingConfig(
        model_name=get("embedding.model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        dimensions=get("embedding.dimensions", 384),
        batch_size=get("embedding.batch_size", 64),
        device=get("embedding.device", "cpu"),
        normalize=get("embedding.normalize", True),
        use_context=get("embedding.use_context", True),
        context_template=get("embedding.context_template", "Title: {title}\nHeading: {heading}\nContent: {content}"),
        cache_dir=get("embedding.cache_dir", "data/embeddings_cache"),
    )
    
    chunk_cfg = ChunkingConfig(
        strategy=get("chunking.strategy", "semantic"),
        max_tokens=get("chunking.max_tokens", 512),
        min_tokens=get("chunking.min_tokens", 100),
        overlap_tokens=get("chunking.overlap_tokens", 50),
        parent_max_tokens=get("chunking.parent_max_tokens", 1536),
        parent_scope=get("chunking.parent_scope", "sheet"),
        dedup_questions=get("chunking.dedup_questions", False),
        type_overrides=get("chunking.type_overrides", {}),
    )
    
    parser_cfg = ParserConfig(
        xlsx_engine=get("parser.xlsx_engine", "auto"),
        pdf_engine=get("parser.pdf_engine", "pymupdf"),
        docx_engine=get("parser.docx_engine", "python-docx"),
    )
    
    ragas_cfg = RagasConfig(
        llm_model=get("ragas.llm_model", "gpt-4o-mini"),
        embedding_model=get("ragas.embedding_model", "text-embedding-3-small"),
        api_key=get("ragas.api_key", ""),
        base_url=get("ragas.base_url", ""),
        metrics=tuple(get("ragas.metrics", ["faithfulness", "answer_relevancy", "context_recall", "context_precision"])),
        k=get("ragas.k", 5),
    )
    
    retrieval_cfg = RetrievalConfig(
        bm25_weight=get("retrieval.bm25_weight", 0.3),
        dense_weight=get("retrieval.dense_weight", 0.7),
        rrf_k=get("retrieval.rrf_k", 60),
        rrf_enabled=get("retrieval.rrf_enabled", True),
        bm25_candidates=get("retrieval.bm25_candidates", 50),
        dense_candidates=get("retrieval.dense_candidates", 50),
        rerank_candidates=get("retrieval.rerank_candidates", 20),
        final_top_k=get("retrieval.final_top_k", 10),
        adaptive_weights=get("retrieval.adaptive_weights", {}),
        strategies=get("retrieval.strategies", {}),
    )
    
    reranker_cfg = RerankerConfig(
        model_name=get("reranker.model_name", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),
        batch_size=get("reranker.batch_size", 32),
        max_length=get("reranker.max_length", 512),
        device=get("reranker.device", ""),
        enabled=get("reranker.enabled", True),
        threshold=get("reranker.threshold", 0.0),
    )
    
    hyde_cfg = HyDEConfig(
        enabled=get("hyde.enabled", True),
        model_name=get("hyde.model_name", "gpt-4o-mini"),
        prompt_template=get("hyde.prompt_template", HyDEConfig.__dataclass_fields__["prompt_template"].default),
        max_length=get("hyde.max_length", 512),
        temperature=get("hyde.temperature", 0.3),
        cache_ttl_seconds=get("hyde.cache_ttl_seconds", 3600),
        top_k=get("hyde.top_k", 20),
    )
    
    multi_query_cfg = MultiQueryConfig(
        enabled=get("multi_query.enabled", True),
        model_name=get("multi_query.model_name", "gpt-4o-mini"),
        prompt_template=get("multi_query.prompt_template", MultiQueryConfig.__dataclass_fields__["prompt_template"].default),
        num_queries=get("multi_query.num_queries", 6),
        beam_size=get("multi_query.beam_size", 5),
        temperature=get("multi_query.temperature", 0.4),
        cache_ttl_seconds=get("multi_query.cache_ttl_seconds", 3600),
        fusion_method=get("multi_query.fusion_method", "rrf"),
        weights=get("multi_query.weights", {}),
    )
    
    auth_cfg = AuthConfig(
        secret_key=os.getenv("KB_AUTH_SECRET_KEY", get("auth.secret_key", "")),
        algorithm=get("auth.algorithm", "HS256"),
        access_token_expire_minutes=get("auth.access_token_expire_minutes", 30),
        refresh_token_expire_days=get("auth.refresh_token_expire_days", 7),
        password_hash_algorithm=get("auth.password_hash_algorithm", "bcrypt"),
        roles=get("auth.roles", {}),
        default_role=get("auth.default_role", "viewer"),
        session_cookie_name=get("auth.session_cookie_name", "kb_session"),
        session_cookie_secure=get("auth.session_cookie_secure", True),
        session_cookie_httponly=get("auth.session_cookie_httponly", True),
        session_cookie_samesite=get("auth.session_cookie_samesite", "lax"),
        rate_limit_enabled=get("auth.rate_limit_enabled", True),
        rate_limit_requests_per_minute=get("auth.rate_limit_requests_per_minute", 100),
        api_key_prefix=get("auth.api_key_prefix", "kb_"),
        api_key_hash_algorithm=get("auth.api_key_hash_algorithm", "bcrypt"),
    )
    
    monitoring_cfg = MonitoringConfig(
        recall_threshold=get("monitoring.recall_threshold", 0.85),
        mrr_threshold=get("monitoring.mrr_threshold", 0.60),
        ndcg_threshold=get("monitoring.ndcg_threshold", 0.70),
        hit_rate_threshold=get("monitoring.hit_rate_threshold", 0.80),
        retrieval_latency_p95_ms=get("monitoring.retrieval_latency_p95_ms", 500),
        end_to_end_latency_p95_ms=get("monitoring.end_to_end_latency_p95_ms", 1500),
        bm25_latency_p95_ms=get("monitoring.bm25_latency_p95_ms", 50),
        dense_latency_p95_ms=get("monitoring.dense_latency_p95_ms", 200),
        rerank_latency_p95_ms=get("monitoring.rerank_latency_p95_ms", 150),
        stale_fraction_threshold=get("monitoring.stale_fraction_threshold", 0.05),
        staleness_check_interval_hours=get("monitoring.staleness_check_interval_hours", 24),
        embedding_cost_per_1k_queries_usd=get("monitoring.embedding_cost_per_1k_queries_usd", 0.50),
        llm_cost_per_1k_queries_usd=get("monitoring.llm_cost_per_1k_queries_usd", 1.00),
        cost_check_interval_hours=get("monitoring.cost_check_interval_hours", 24),
        alert_channels=get("monitoring.alert_channels", ["log"]),
        slack_webhook_url=get("monitoring.slack_webhook_url", ""),
        email_smtp_host=get("monitoring.email_smtp_host", ""),
        email_smtp_port=get("monitoring.email_smtp_port", 587),
        email_from=get("monitoring.email_from", ""),
        email_to=get("monitoring.email_to", []),
    )
    
    web_cfg = WebConfig(
        host=get("web.host", "0.0.0.0"),
        port=get("web.port", 8000),
        language=get("web.language", "fa"),
        rtl=get("web.rtl", True),
        theme=get("web.theme", "auto"),
        font_family=get("web.font_family", "Vazirmatn"),
        page_size=get("web.page_size", 20),
        editor_page_size=get("web.editor_page_size", 50),
        dev_api_prefix=get("web.dev_api_prefix", "/api/dev"),
        dev_rate_limit_per_minute=get("web.dev_rate_limit_per_minute", 60),
        enable_editor=get("web.enable_editor", True),
        enable_monitoring=get("web.enable_monitoring", True),
        enable_benchmarks=get("web.enable_benchmarks", True),
        enable_dev_portal=get("web.enable_dev_portal", True),
        enable_settings_ui=get("web.enable_settings_ui", True),
    )
    
    return AppConfig(
        db=db_cfg,
        embedding=embed_cfg,
        chunking=chunk_cfg,
        parser=parser_cfg,
        ragas=ragas_cfg,
        retrieval=retrieval_cfg,
        reranker=reranker_cfg,
        hyde=hyde_cfg,
        multi_query=multi_query_cfg,
        auth=auth_cfg,
        monitoring=monitoring_cfg,
        web=web_cfg,
        source_dir=os.getenv("KB_SOURCE_DIR", get("source_dir", str(PROJECT_ROOT / "data"))),
        output_dir=os.getenv("KB_OUTPUT_DIR", get("output_dir", str(PROJECT_ROOT / "data" / "processed"))),
    )


# Global config instance (loaded once at import)
config = load_config()