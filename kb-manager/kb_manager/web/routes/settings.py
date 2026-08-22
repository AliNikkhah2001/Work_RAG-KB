"""Settings / Configuration management routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from kb_manager.config import (
    CONFIG_DIR,
    AppConfig,
    load_config,
)
from kb_manager.web.app import templates

router = APIRouter(prefix="/settings", tags=["settings"])

# Config section metadata for UI tabs
CONFIG_SECTIONS = [
    {
        "id": "database",
        "label": "پایگاه داده",
        "icon": "database",
        "fields": [
            ("db.host", "text", "هاست"),
            ("db.port", "number", "پورت"),
            ("db.name", "text", "نام دیتابیس"),
            ("db.user", "text", "کاربر"),
            ("db.password", "password", "رمز عبور"),
            ("db.echo", "checkbox", "نمایش کوئری‌های SQL"),
            ("db.url_override", "text", "URL کامل اتصال (اختیاری)"),
        ],
    },
    {
        "id": "embedding",
        "label": "امبدینگ",
        "icon": "cpu-chip",
        "fields": [
            ("embedding.model_name", "text", "نام مدل"),
            ("embedding.dimensions", "number", "ابعاد"),
            ("embedding.batch_size", "number", "اندازه دسته"),
            ("embedding.device", "select", "دستگاه", ["cpu", "cuda", "mps"]),
            ("embedding.normalize", "checkbox", "نرمال‌سازی"),
            ("embedding.use_context", "checkbox", "استفاده از متن زمینه‌دار"),
            ("embedding.cache_dir", "text", "دایرکتوری کش"),
        ],
    },
    {
        "id": "chunking",
        "label": "چانکینگ",
        "icon": "scissors",
        "fields": [
            ("chunking.strategy", "select", "استراتژی", ["semantic", "fixed"]),
            ("chunking.max_tokens", "number", "حداکثر توکن"),
            ("chunking.min_tokens", "number", "حداقل توکن"),
            ("chunking.overlap_tokens", "number", "توکن‌های همپوشانی"),
            ("chunking.parent_max_tokens", "number", "حداکثر توکن پرنت"),
            ("chunking.parent_scope", "select", "دامنه پرنت", ["sheet", "document"]),
            ("chunking.dedup_questions", "checkbox", "حذف سوالات تکراری"),
        ],
    },
    {
        "id": "retrieval",
        "label": "بازیابی",
        "icon": "magnifying-glass",
        "fields": [
            ("retrieval.bm25_weight", "number", "وزن BM25", 0.0, 1.0, 0.1),
            ("retrieval.dense_weight", "number", "وزن Dense", 0.0, 1.0, 0.1),
            ("retrieval.rrf_k", "number", "ثابت RRF (k)"),
            ("retrieval.rrf_enabled", "checkbox", "فعال‌سازی RRF"),
            ("retrieval.bm25_candidates", "number", "کاندیداهای BM25"),
            ("retrieval.dense_candidates", "number", "کاندیداهای Dense"),
            ("retrieval.rerank_candidates", "number", "کاندیداهای ریرانک"),
            ("retrieval.final_top_k", "number", "نتایج نهایی Top-K"),
        ],
    },
    {
        "id": "reranker",
        "label": "ریرانکر",
        "icon": "arrow-up-right",
        "fields": [
            ("reranker.model_name", "text", "نام مدل"),
            ("reranker.batch_size", "number", "اندازه دسته"),
            ("reranker.max_length", "number", "حداکثر طول"),
            ("reranker.device", "select", "دستگاه", ["", "cpu", "cuda"]),
            ("reranker.enabled", "checkbox", "فعال"),
            ("reranker.threshold", "number", "آستانه امتیاز", 0.0, 1.0, 0.01),
        ],
    },
    {
        "id": "hyde",
        "label": "HyDE",
        "icon": "sparkles",
        "fields": [
            ("hyde.enabled", "checkbox", "فعال"),
            ("hyde.model_name", "text", "مدل LLM"),
            ("hyde.max_length", "number", "حداکثر طول"),
            ("hyde.temperature", "number", "دمای", 0.0, 1.0, 0.1),
            ("hyde.cache_ttl_seconds", "number", "TTL کش (ثانیه)"),
            ("hyde.top_k", "number", "Top-K جستجو"),
        ],
    },
    {
        "id": "multi_query",
        "label": "چندین پرسش",
        "icon": "chat-bubble-left-right",
        "fields": [
            ("multi_query.enabled", "checkbox", "فعال"),
            ("multi_query.model_name", "text", "مدل LLM"),
            ("multi_query.num_queries", "number", "تعداد پرسش‌ها"),
            ("multi_query.beam_size", "number", "اندازه بيم"),
            ("multi_query.temperature", "number", "دمای", 0.0, 1.0, 0.1),
            ("multi_query.fusion_method", "select", "روش ادغام", ["rrf", "weighted_rrf"]),
        ],
    },
    {
        "id": "auth",
        "label": "احراز هویت",
        "icon": "lock-closed",
        "fields": [
            ("auth.algorithm", "select", "الگوریتم", ["HS256", "RS256"]),
            ("auth.access_token_expire_minutes", "number", "انقضای توکن دسترسی (دقیقه)"),
            ("auth.refresh_token_expire_days", "number", "انقضای توکن تازه‌سازی (روز)"),
            ("auth.password_hash_algorithm", "select", "هش رمز", ["bcrypt", "argon2"]),
            ("auth.default_role", "select", "نقش پیش‌فرض", ["viewer", "developer", "editor", "admin"]),
            ("auth.rate_limit_enabled", "checkbox", "محدودیت نرخ"),
            ("auth.rate_limit_requests_per_minute", "number", "درخواست در دقیقه"),
        ],
    },
    {
        "id": "monitoring",
        "label": "مانیتورینگ",
        "icon": "chart-bar",
        "fields": [
            ("monitoring.recall_threshold", "number", "آستانه Recall", 0.0, 1.0, 0.01),
            ("monitoring.mrr_threshold", "number", "آستانه MRR", 0.0, 1.0, 0.01),
            ("monitoring.ndcg_threshold", "number", "آستانه NDCG", 0.0, 1.0, 0.01),
            ("monitoring.retrieval_latency_p95_ms", "number", "_latency بازیابی P95 (ms)"),
            ("monitoring.end_to_end_latency_p95_ms", "number", "Latency اندر-اندیر P95 (ms)"),
            ("monitoring.stale_fraction_threshold", "number", "آستانه بخش منقضی", 0.0, 1.0, 0.01),
            ("monitoring.alert_channels", "text", "کانال‌های هشدار (کاما جدا)"),
        ],
    },
    {
        "id": "web",
        "label": "وب",
        "icon": "globe-alt",
        "fields": [
            ("web.host", "text", "هاست"),
            ("web.port", "number", "پورت"),
            ("web.language", "select", "زبان", ["fa", "en"]),
            ("web.rtl", "checkbox", "RTL"),
            ("web.theme", "select", "تم", ["light", "dark", "auto"]),
            ("web.font_family", "text", "فونت"),
            ("web.page_size", "number", "اندازه صفحه"),
            ("web.dev_api_prefix", "text", "پیشوند API توسعه‌دهنده"),
        ],
    },
]


def _get_nested(config: dict[str, Any], path: str, default: Any = None) -> Any:
    """Get nested config value by dot path."""
    keys = path.split(".")
    val = config
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
        if val is None:
            return default
    return val


def _set_nested(config: dict[str, Any], path: str, value: Any) -> None:
    """Set nested config value by dot path."""
    keys = path.split(".")
    val = config
    for k in keys[:-1]:
        if k not in val or not isinstance(val[k], dict):
            val[k] = {}
        val = val[k]
    val[keys[-1]] = value


def _load_all_yaml() -> dict[str, Any]:
    """Load and merge all YAML config files."""
    from kb_manager.config import _load_yaml, _merge_configs
    
    base = _load_yaml(CONFIG_DIR / "default.yaml")
    retrieval_yaml = _load_yaml(CONFIG_DIR / "retrieval.yaml")
    reranker_yaml = _load_yaml(CONFIG_DIR / "reranker.yaml")
    hyde_yaml = _load_yaml(CONFIG_DIR / "hyde.yaml")
    multi_query_yaml = _load_yaml(CONFIG_DIR / "multi_query.yaml")
    auth_yaml = _load_yaml(CONFIG_DIR / "auth.yaml")
    monitoring_yaml = _load_yaml(CONFIG_DIR / "monitoring.yaml")
    web_yaml = _load_yaml(CONFIG_DIR / "web.yaml")
    chunking_yaml = _load_yaml(CONFIG_DIR / "chunking" / "semantic.yaml")
    
    return _merge_configs(
        base,
        {"retrieval": retrieval_yaml},
        {"reranker": reranker_yaml},
        {"hyde": hyde_yaml},
        {"multi_query": multi_query_yaml},
        {"auth": auth_yaml},
        {"monitoring": monitoring_yaml},
        {"web": web_yaml},
        {"chunking": chunking_yaml},
    )


@router.get("")
async def settings_page(request: Request):
    """Settings page with tabbed configuration forms."""
    config = _load_all_yaml()
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "sections": CONFIG_SECTIONS,
            "config": config,
        },
    )


@router.post("")
async def save_settings(request: Request):
    """Save settings from form submission."""
    form = await request.form()
    
    # Load current merged config
    config = _load_all_yaml()
    
    # Update from form
    for key, value in form.items():
        # key format: "section.field"
        if "." in key:
            _set_nested(config, key, _parse_form_value(value))
    
    # Write back to individual YAML files
    _write_config_files(config)
    
    return RedirectResponse("/settings?saved=1", status_code=303)


def _parse_form_value(value: str) -> Any:
    """Parse form string value to appropriate Python type."""
    # Boolean
    if value.lower() in ("true", "on", "yes", "1"):
        return True
    if value.lower() in ("false", "off", "no", "0"):
        return False
    # Number
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    # List (comma-separated)
    if "," in value:
        return [v.strip() for v in value.split(",") if v.strip()]
    return value


def _write_config_files(config: dict[str, Any]) -> None:
    """Write merged config back to individual YAML files."""
    # Extract sections
    sections = {
        "default": {},
        "retrieval": config.get("retrieval", {}),
        "reranker": config.get("reranker", {}),
        "hyde": config.get("hyde", {}),
        "multi_query": config.get("multi_query", {}),
        "auth": config.get("auth", {}),
        "monitoring": config.get("monitoring", {}),
        "web": config.get("web", {}),
        "chunking": config.get("chunking", {}),
    }
    
    # Base config keys (non-section-specific)
    base_keys = {"db", "embedding", "chunking", "parser", "ragas", "source_dir", "output_dir"}
    for k in base_keys:
        if k in config:
            sections["default"][k] = config[k]
    
    # Write each file
    for section, data in sections.items():
        if section == "chunking":
            path = CONFIG_DIR / "chunking" / "semantic.yaml"
        elif section == "default":
            path = CONFIG_DIR / "default.yaml"
        else:
            path = CONFIG_DIR / f"{section}.yaml"
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False, indent=2)


@router.get("/export")
async def export_config():
    """Export full configuration as YAML."""
    config = _load_all_yaml()
    yaml_str = yaml.dump(config, allow_unicode=True, sort_keys=False, indent=2)
    return JSONResponse(
        content={"config": yaml_str},
        headers={"Content-Disposition": "attachment; filename=kb-config.yaml"},
    )


@router.post("/import")
async def import_config(request: Request):
    """Import configuration from YAML."""
    form = await request.form()
    file = form.get("file")
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    content = await file.read()
    try:
        imported = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")
    
    if not isinstance(imported, dict):
        raise HTTPException(status_code=400, detail="YAML must be a mapping")
    
    _write_config_files(imported)
    return RedirectResponse("/settings?imported=1", status_code=303)