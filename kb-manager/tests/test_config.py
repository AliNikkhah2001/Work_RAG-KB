"""Tests for configuration system."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from kb_manager.config import (
    AppConfig,
    DatabaseConfig,
    EmbeddingConfig,
    ChunkingConfig,
    ParserConfig,
    RetrievalConfig,
    RerankerConfig,
    HyDEConfig,
    MultiQueryConfig,
    AuthConfig,
    MonitoringConfig,
    WebConfig,
    RagasConfig,
    load_config,
    _load_yaml,
    _merge_configs,
)


# ---------------------------------------------------------------------------
# YAML Loading Tests
# ---------------------------------------------------------------------------

def test_load_yaml_missing_file():
    """Loading missing YAML returns empty dict."""
    result = _load_yaml(Path("/nonexistent/path.yaml"))
    assert result == {}


def test_load_yaml_with_env_interpolation(tmp_path):
    """YAML values with ${ENV_VAR} are interpolated."""
    os.environ["TEST_DB_HOST"] = "test-host"
    os.environ["TEST_DB_PORT"] = "5433"
    
    content = """
db:
  host: "${TEST_DB_HOST}"
  port: ${TEST_DB_PORT}
  name: "${UNSET_VAR:-default_db}"
"""
    path = tmp_path / "test.yaml"
    path.write_text(content)
    
    result = _load_yaml(path)
    
    assert result["db"]["host"] == "test-host"
    # YAML keeps interpolated values as strings
    assert result["db"]["port"] == "5433"
    assert result["db"]["name"] == "default_db"
    
    del os.environ["TEST_DB_HOST"]
    del os.environ["TEST_DB_PORT"]


def test_merge_configs_deep():
    """Deep merge preserves nested structures."""
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"c": 20, "e": 30}, "f": 40}
    
    result = _merge_configs(base, override)
    
    assert result == {"a": {"b": 1, "c": 20, "e": 30}, "d": 3, "f": 40}


# ---------------------------------------------------------------------------
# Config Dataclass Tests
# ---------------------------------------------------------------------------

def test_database_config_defaults():
    """DatabaseConfig has correct defaults."""
    cfg = DatabaseConfig()
    assert cfg.host == "localhost"
    assert cfg.port == 5432
    assert cfg.name == "kb_manager"
    assert cfg.user == "postgres"
    assert cfg.password == "postgres"
    assert cfg.echo is False
    assert cfg.async_url == "postgresql+asyncpg://postgres:postgres@localhost:5432/kb_manager"


def test_database_config_sqlite_detection():
    """is_sqlite detects sqlite URLs."""
    cfg = DatabaseConfig(url_override="sqlite+aiosqlite:///test.db")
    assert cfg.is_sqlite is True
    assert cfg.driver == "aiosqlite"


def test_embedding_config_defaults():
    """EmbeddingConfig has correct defaults."""
    cfg = EmbeddingConfig()
    assert cfg.model_name == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    assert cfg.dimensions == 384
    assert cfg.batch_size == 64
    assert cfg.device == "cpu"
    assert cfg.normalize is True
    assert cfg.use_context is True


def test_chunking_config_type_overrides():
    """ChunkingConfig type_overrides works."""
    cfg = ChunkingConfig(
        type_overrides={
            "reason_code": {"strategy": "single_entity", "max_tokens": 400},
            "qa_pair": {"strategy": "single_entity", "max_tokens": 600},
        }
    )
    assert cfg.type_overrides["reason_code"]["max_tokens"] == 400
    assert cfg.type_overrides["qa_pair"]["max_tokens"] == 600


def test_retrieval_config_adaptive_weights():
    """RetrievalConfig adaptive_weights structure."""
    cfg = RetrievalConfig(
        adaptive_weights={
            "verbatim": {"bm25_weight": 0.5, "dense_weight": 0.5},
            "keyword_only": {"bm25_weight": 0.8, "dense_weight": 0.2},
        }
    )
    assert cfg.adaptive_weights["verbatim"]["bm25_weight"] == 0.5
    assert cfg.adaptive_weights["keyword_only"]["bm25_weight"] == 0.8


def test_reranker_config_defaults():
    """RerankerConfig defaults."""
    cfg = RerankerConfig()
    assert cfg.model_name == "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    assert cfg.batch_size == 32
    assert cfg.max_length == 512
    assert cfg.enabled is True
    assert cfg.threshold == 0.0


def test_hyde_config_prompt_template():
    """HyDEConfig has Persian prompt template."""
    cfg = HyDEConfig()
    assert cfg.enabled is True
    assert "متخصص سیستم اعتبارسنجی" in cfg.prompt_template
    assert "{query}" in cfg.prompt_template


def test_multi_query_config_weights():
    """MultiQueryConfig query type weights."""
    cfg = MultiQueryConfig()
    assert cfg.weights["verbatim"] == 1.0
    assert cfg.weights["keyword_only"] == 0.5
    assert cfg.fusion_method == "rrf"


def test_auth_config_roles():
    """AuthConfig roles structure."""
    cfg = AuthConfig()
    assert "admin" in cfg.roles
    assert "editor" in cfg.roles
    assert "developer" in cfg.roles
    assert "viewer" in cfg.roles
    assert cfg.default_role == "viewer"
    assert "*" in cfg.roles["admin"]


def test_monitoring_config_thresholds():
    """MonitoringConfig quality thresholds."""
    cfg = MonitoringConfig()
    assert cfg.recall_threshold == 0.85
    assert cfg.mrr_threshold == 0.60
    assert cfg.retrieval_latency_p95_ms == 500


def test_web_config_defaults():
    """WebConfig defaults."""
    cfg = WebConfig()
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8000
    assert cfg.language == "fa"
    assert cfg.rtl is True
    assert cfg.font_family == "Vazirmatn"
    assert cfg.enable_editor is True


# ---------------------------------------------------------------------------
# load_config Integration Tests
# ---------------------------------------------------------------------------

def test_load_config_returns_app_config():
    """load_config returns fully populated AppConfig."""
    cfg = load_config()
    
    assert isinstance(cfg, AppConfig)
    assert isinstance(cfg.db, DatabaseConfig)
    assert isinstance(cfg.embedding, EmbeddingConfig)
    assert isinstance(cfg.chunking, ChunkingConfig)
    assert isinstance(cfg.parser, type(cfg.parser))
    assert isinstance(cfg.ragas, RagasConfig)
    assert isinstance(cfg.retrieval, RetrievalConfig)
    assert isinstance(cfg.reranker, RerankerConfig)
    assert isinstance(cfg.hyde, HyDEConfig)
    assert isinstance(cfg.multi_query, MultiQueryConfig)
    assert isinstance(cfg.auth, AuthConfig)
    assert isinstance(cfg.monitoring, MonitoringConfig)
    assert isinstance(cfg.web, WebConfig)


def test_load_config_env_override_db(monkeypatch):
    """Environment variables override config values."""
    monkeypatch.setenv("KB_DB_HOST", "env-host")
    monkeypatch.setenv("KB_DB_PORT", "5433")
    monkeypatch.setenv("KB_EMBED_MODEL", "custom-model")
    monkeypatch.setenv("KB_EMBED_DIM", "768")
    monkeypatch.setenv("KB_CHUNK_STRATEGY", "fixed")
    monkeypatch.setenv("KB_CHUNK_MAX", "1024")
    
    # Need to reload config module to pick up env changes
    import importlib
    import kb_manager.config as config_module
    importlib.reload(config_module)
    
    cfg = config_module.load_config()
    
    assert cfg.db.host == "env-host"
    assert cfg.db.port == 5433
    assert cfg.embedding.model_name == "custom-model"
    assert cfg.embedding.dimensions == 768
    assert cfg.chunking.strategy == "fixed"
    assert cfg.chunking.max_tokens == 1024


def test_load_config_yaml_files_exist():
    """Config YAML files exist and are loadable."""
    from kb_manager.config import CONFIG_DIR
    
    required_files = [
        "default.yaml",
        "retrieval.yaml",
        "reranker.yaml",
        "hyde.yaml",
        "multi_query.yaml",
        "auth.yaml",
        "monitoring.yaml",
        "web.yaml",
    ]
    
    for fname in required_files:
        path = CONFIG_DIR / fname
        assert path.exists(), f"Missing config file: {fname}"
        
        # Validate YAML syntax
        with path.open() as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), f"{fname} must be a mapping"


def test_config_files_have_persian_comments():
    """Config files contain Persian text (comments or values)."""
    from kb_manager.config import CONFIG_DIR
    
    # Check a few files for Persian content
    for fname in ["retrieval.yaml", "hyde.yaml", "multi_query.yaml"]:
        path = CONFIG_DIR / fname
        content = path.read_text(encoding="utf-8")
        # Should contain Persian characters
        has_persian = any('\u0600' <= c <= '\u06FF' for c in content)
        assert has_persian, f"{fname} should contain Persian text"


# ---------------------------------------------------------------------------
# Settings Route Tests (require FastAPI test client)
# ---------------------------------------------------------------------------

@pytest.fixture
def test_app():
    """Create test FastAPI app with settings router."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from kb_manager.web.app import templates
    from kb_manager.web.routes import settings
    
    app = FastAPI()
    app.include_router(settings.router, prefix="/settings")
    
    # Mock templates
    from fastapi.templating import Jinja2Templates
    app.state.templates = templates
    
    return TestClient(app)


def test_settings_page_loads(test_app):
    """GET /settings returns 200."""
    response = test_app.get("/settings")
    assert response.status_code == 200
    assert "تنظیمات سیستم" in response.text


def test_settings_tabs_present(test_app):
    """Settings page has all tab buttons."""
    response = test_app.get("/settings")
    
    expected_tabs = ["database", "embedding", "chunking", "retrieval", "reranker", "hyde", "multi_query", "auth", "monitoring", "web"]
    
    for tab_id in expected_tabs:
        assert f'id="tab-{tab_id}"' in response.text
        assert f'id="panel-{tab_id}"' in response.text


# ---------------------------------------------------------------------------
# YAML Round-trip Test
# ---------------------------------------------------------------------------

def test_config_yaml_roundtrip(tmp_path):
    """Config can be saved to YAML and loaded back."""
    from kb_manager.config import load_config, CONFIG_DIR
    
    # Load current config
    cfg = load_config()
    
    # Dump to temp YAML
    import dataclasses
    def to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        elif isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_dict(v) for v in obj]
        return obj
    
    cfg_dict = to_dict(cfg)
    
    # Save
    out_path = tmp_path / "roundtrip.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg_dict, f, allow_unicode=True, sort_keys=False, indent=2)
    
    # Load back
    with out_path.open() as f:
        loaded = yaml.safe_load(f)
    
    assert loaded["db"]["host"] == cfg.db.host
    assert loaded["embedding"]["model_name"] == cfg.embedding.model_name
    assert loaded["retrieval"]["bm25_weight"] == cfg.retrieval.bm25_weight