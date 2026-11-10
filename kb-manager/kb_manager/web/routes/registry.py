"""Registry API + UI: versioned packs and IndexManifests."""

from __future__ import annotations

import logging
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from kb_manager.registry import (
    PackNotFoundError,
    build_manifest,
    get_registry,
    list_manifests,
    load_active_manifest,
    save_manifest,
)
from kb_manager.web.deps import templates

logger = logging.getLogger(__name__)
router = APIRouter()


class ActivateBody(BaseModel):
    manifest_id: str = Field(..., description="Manifest id to activate")


class SavePackBody(BaseModel):
    """Save / bump a registry pack from the Registry UI."""

    content: Any = Field(
        ...,
        description="Pack as a JSON object, or a YAML/JSON string",
    )
    version: str | None = Field(
        None, description="Base version to overwrite or bump from (default: latest)"
    )
    scope: str | None = Field(
        None, description="Required for prompts (default orchestrator)"
    )
    bump: str | None = Field(
        "patch",
        description="patch|minor|major|overwrite — default patch creates a new version",
    )
    update_active_pin: bool = Field(
        True, description="Also point retrieval.active_packs at the saved version"
    )


@router.get("")
async def registry_ui(request: Request):
    """HTML dashboard for packs + manifests."""
    reg = get_registry()
    pins = reg.active_pins()
    kinds = {
        "schemas": reg.list_versions("schemas"),
        "taxonomy": reg.list_versions("taxonomy"),
        "canned": reg.list_versions("canned"),
        "prompts/orchestrator": reg.list_versions("prompts", scope="orchestrator"),
        "retrieval": reg.list_versions("retrieval"),
    }
    active = load_active_manifest()
    return templates.TemplateResponse(
        request,
        "registry.html",
        {
            "pins": pins,
            "kinds": kinds,
            "manifests": list_manifests(),
            "active": active.to_dict() if active else None,
            "taxonomy": reg.taxonomy_pack(pins.get("taxonomy")),
        },
    )


@router.get("/api/packs")
async def list_packs():
    reg = get_registry()
    return {
        "pins": reg.active_pins(),
        "versions": {
            "schemas": reg.list_versions("schemas"),
            "taxonomy": reg.list_versions("taxonomy"),
            "canned": reg.list_versions("canned"),
            "prompts_orchestrator": reg.list_versions("prompts", scope="orchestrator"),
            "prompts_guardrails": reg.list_versions("prompts", scope="guardrails"),
            "retrieval": reg.list_versions("retrieval"),
        },
    }


def _resolve_kind_scope(kind: str, scope: str | None) -> tuple[str, str | None]:
    real_kind = kind
    real_scope = scope
    if kind in ("prompts_orchestrator", "prompts-orchestrator"):
        real_kind = "prompts"
        real_scope = scope or "orchestrator"
    if kind in ("prompts_guardrails", "prompts-guardrails"):
        real_kind = "prompts"
        real_scope = scope or "guardrails"
    if real_kind == "prompts" and not real_scope:
        real_scope = "orchestrator"
    return real_kind, real_scope


def _parse_pack_content(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        text = content.strip()
        if not text:
            raise ValueError("empty pack content")
        try:
            data = yaml.safe_load(text)
        except Exception as exc:
            raise ValueError(f"invalid YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("pack YAML/JSON must be a mapping at the top level")
        return data
    raise ValueError("content must be an object or YAML/JSON string")


@router.get("/api/packs/{kind}")
async def get_pack(
    kind: str,
    version: str | None = Query(None),
    scope: str | None = Query(None),
    as_yaml: bool = Query(False, description="Return {yaml, version, kind} instead of JSON object"),
):
    reg = get_registry()
    real_kind, real_scope = _resolve_kind_scope(kind, scope)
    try:
        data = reg.load(real_kind, version=version, scope=real_scope)
    except PackNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if as_yaml:
        from kb_manager.registry.loader import _dump_yaml

        return {
            "kind": real_kind,
            "scope": real_scope,
            "version": data.get("version"),
            "yaml": _dump_yaml(data),
        }
    return data


@router.put("/api/packs/{kind}")
async def save_pack(kind: str, body: SavePackBody):
    """Create or overwrite a pack version (used by the Registry editor UI)."""
    reg = get_registry()
    real_kind, real_scope = _resolve_kind_scope(kind, body.scope)
    try:
        data = _parse_pack_content(body.content)
        result = reg.save(
            real_kind,
            data,
            version=body.version,
            scope=real_scope,
            bump=body.bump,
            update_active_pin=body.update_active_pin,
        )
    except PackNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        logger.exception("pack save failed")
        raise HTTPException(status_code=500, detail=f"failed to write pack: {exc}") from exc
    return result


@router.post("/api/reload")
async def reload_registry():
    """Drop in-memory pack cache so the next read hits disk."""
    reg = get_registry()
    reg.clear_cache()
    return {"ok": True, "pins": reg.active_pins()}


@router.get("/api/taxonomy")
async def get_taxonomy(version: str | None = Query(None)):
    reg = get_registry()
    try:
        tax = reg.taxonomy_pack(version)
    except PackNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "version": tax.get("version"),
        "subspaces": list((tax.get("nodes") or {}).keys()),
        "subspace_map": reg.subspace_map(version),
        "keyword_boost": reg.subspace_keyword_boost(version),
        "nodes": tax.get("nodes"),
        "defaults": tax.get("defaults"),
        "ui_filter_groups": tax.get("ui_filter_groups"),
        "edges": tax.get("edges"),
    }


@router.get("/api/canned")
async def get_canned(version: str | None = Query(None)):
    reg = get_registry()
    try:
        return reg.canned_pack(version)
    except PackNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/api/prompts/{scope}")
async def get_prompts(scope: str, version: str | None = Query(None)):
    reg = get_registry()
    try:
        return reg.prompts_pack(scope=scope, version=version)
    except PackNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/api/manifests")
async def api_list_manifests():
    active = load_active_manifest()
    return {"manifests": list_manifests(), "active": (active.to_dict() if active else None)}


@router.post("/api/manifests/build")
async def api_build_manifest(activate: bool = Query(False)):
    """Build (and optionally activate) a manifest from current DB counts + pins."""
    from sqlalchemy import func, select

    from kb_manager.models.database import Chunk, Document
    from kb_manager.web.deps import db

    doc_count = chunk_count = 0
    try:
        async with db.session() as session:
            doc_count = (await session.execute(select(func.count(Document.id)))).scalar() or 0
            chunk_count = (await session.execute(select(func.count(Chunk.id)))).scalar() or 0
    except Exception as exc:
        logger.warning("manifest build count failed: %s", exc)

    manifest = build_manifest(doc_count=doc_count, chunk_count=chunk_count, activate=activate)
    path = save_manifest(manifest, activate=activate)
    return {"ok": True, "path": str(path), "manifest": manifest.to_dict()}


@router.post("/api/manifests/activate")
async def api_activate_manifest(body: ActivateBody):
    from kb_manager.registry.manifest import MANIFESTS_DIR, IndexManifest

    path = MANIFESTS_DIR / f"{body.manifest_id}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"manifest {body.manifest_id} not found")
    manifest = IndexManifest.from_dict(__import__("json").loads(path.read_text(encoding="utf-8")))
    save_manifest(manifest, activate=True)
    return {"ok": True, "active": body.manifest_id}


@router.get("/api/active")
async def api_active():
    """Single bootstrap payload for orchestrator / clients."""
    reg = get_registry()
    pins = reg.active_pins()
    active = load_active_manifest()
    tax_ver = pins.get("taxonomy")
    canned_ver = pins.get("canned")
    prompt_ver = pins.get("prompts_orchestrator")
    guard_prompt_ver = pins.get("prompts_guardrails")
    return {
        "pins": pins,
        "manifest": active.to_dict() if active else None,
        "taxonomy": {
            "version": tax_ver,
            "subspaces": list(reg.subspace_ids(tax_ver)),
            "subspace_map": reg.subspace_map(tax_ver),
            "keyword_boost": reg.subspace_keyword_boost(tax_ver),
            "ui_filter_groups": (reg.taxonomy_pack(tax_ver).get("ui_filter_groups") if tax_ver else []),
        },
        "canned": reg.canned_pack(canned_ver) if canned_ver else {},
        "prompts": reg.prompts_pack("orchestrator", prompt_ver) if prompt_ver else {},
        "prompts_guardrails": (
            reg.prompts_pack("guardrails", guard_prompt_ver) if guard_prompt_ver else {}
        ),
    }
