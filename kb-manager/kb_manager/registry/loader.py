"""Filesystem-backed versioned pack registry.

Packs live under ``<PROJECT_ROOT>/packs/<kind>/[optional-scope/]<version>.yaml``.
Kinds: schemas, taxonomy, canned, prompts, retrieval.
"""

from __future__ import annotations

import logging
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from kb_manager.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

PACKS_ROOT = PROJECT_ROOT / "packs"
_VERSION_RE = re.compile(r"^v?(\d+\.\d+\.\d+)\.ya?ml$", re.IGNORECASE)


class PackNotFoundError(FileNotFoundError):
    """Raised when a requested pack version does not exist on disk."""


def _normalize_col(name: str) -> str:
    return re.sub(r"[\s_\-]+", "", (name or "").strip().lower())


class Registry:
    """Load and cache versioned YAML packs from disk."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else PACKS_ROOT
        self._lock = threading.RLock()
        self._cache: dict[tuple[str, str | None, str], dict[str, Any]] = {}

    def list_versions(self, kind: str, scope: str | None = None) -> list[str]:
        """Return sorted semver strings available for *kind* (optional *scope*)."""
        directory = self._kind_dir(kind, scope)
        if not directory.is_dir():
            return []
        versions: list[str] = []
        for path in directory.iterdir():
            if not path.is_file():
                continue
            m = _VERSION_RE.match(path.name)
            if m:
                versions.append(m.group(1))
        return sorted(versions, key=lambda v: tuple(int(x) for x in v.split(".")))

    def latest_version(self, kind: str, scope: str | None = None) -> str | None:
        vers = self.list_versions(kind, scope)
        return vers[-1] if vers else None

    def load(self, kind: str, version: str | None = None, scope: str | None = None) -> dict[str, Any]:
        """Load a pack; *version* defaults to the highest available semver."""
        ver = version or self.latest_version(kind, scope)
        if not ver:
            raise PackNotFoundError(f"No packs for kind={kind!r} scope={scope!r}")
        key = (kind, scope, ver)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            path = self._pack_path(kind, ver, scope)
            if not path.is_file():
                raise PackNotFoundError(f"Pack not found: {path}")
            with path.open(encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            if not isinstance(data, dict):
                raise ValueError(f"Pack {path} must be a YAML mapping")
            data.setdefault("version", ver)
            data.setdefault("kind", kind)
            self._cache[key] = data
            logger.debug("Loaded pack %s", path)
            return data

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def active_pins(self) -> dict[str, str]:
        """Return active pack pins from retrieval pack (or latest of each kind)."""
        try:
            retrieval = self.load("retrieval")
            pins = dict(retrieval.get("active_packs") or {})
        except PackNotFoundError:
            pins = {}
        defaults = {
            "schemas": self.latest_version("schemas"),
            "taxonomy": self.latest_version("taxonomy"),
            "canned": self.latest_version("canned"),
            "prompts_orchestrator": self.latest_version("prompts", scope="orchestrator"),
            "retrieval": self.latest_version("retrieval"),
        }
        for k, v in defaults.items():
            if k not in pins and v:
                pins[k] = v
        return {k: str(v) for k, v in pins.items() if v}

    def schemas_pack(self, version: str | None = None) -> dict[str, Any]:
        return self.load("schemas", version)

    def taxonomy_pack(self, version: str | None = None) -> dict[str, Any]:
        return self.load("taxonomy", version)

    def canned_pack(self, version: str | None = None) -> dict[str, Any]:
        return self.load("canned", version)

    def prompts_pack(self, scope: str = "orchestrator", version: str | None = None) -> dict[str, Any]:
        return self.load("prompts", version, scope=scope)

    def retrieval_pack(self, version: str | None = None) -> dict[str, Any]:
        return self.load("retrieval", version)

    def subspace_ids(self, taxonomy_version: str | None = None) -> tuple[str, ...]:
        tax = self.taxonomy_pack(taxonomy_version)
        nodes = tax.get("nodes") or {}
        return tuple(nodes.keys())

    def subspace_map(self, taxonomy_version: str | None = None) -> dict[str, str]:
        """Map subspace id -> filter_path used by KB search."""
        tax = self.taxonomy_pack(taxonomy_version)
        out: dict[str, str] = {}
        for sid, node in (tax.get("nodes") or {}).items():
            if isinstance(node, dict):
                out[sid] = str(node.get("filter_path") or node.get("label_fa") or sid)
        return out

    def subspace_keyword_boost(self, taxonomy_version: str | None = None) -> dict[str, float]:
        tax = self.taxonomy_pack(taxonomy_version)
        out: dict[str, float] = {}
        defaults = (tax.get("defaults") or {}).get("keyword_boost", 3.0)
        for sid, node in (tax.get("nodes") or {}).items():
            if not isinstance(node, dict):
                continue
            retrieval = node.get("retrieval") or {}
            out[sid] = float(retrieval.get("keyword_boost", defaults))
        return out

    def evidence_threshold(self, subspace: str | None, taxonomy_version: str | None = None) -> float:
        tax = self.taxonomy_pack(taxonomy_version)
        defaults = float((tax.get("defaults") or {}).get("evidence_min_score", 0.15))
        if not subspace:
            return defaults
        node = (tax.get("nodes") or {}).get(subspace) or {}
        if not isinstance(node, dict):
            # Allow Persian filter_path / label lookup
            for n in (tax.get("nodes") or {}).values():
                if not isinstance(n, dict):
                    continue
                if subspace in {n.get("filter_path"), n.get("label_fa")} or subspace in (n.get("aliases") or []):
                    node = n
                    break
        retrieval = node.get("retrieval") or {}
        return float(retrieval.get("evidence_min_score", defaults))

    def match_chunk_to_filter(
        self,
        folder_hierarchy: list[str] | None,
        domain: str | None,
        filter_path: str | None = None,
        subspace: str | None = None,
        taxonomy_version: str | None = None,
    ) -> bool:
        """Return True if chunk metadata matches the requested filter/subspace."""
        if not filter_path and not subspace:
            return True
        tax = self.taxonomy_pack(taxonomy_version)
        nodes = tax.get("nodes") or {}
        hierarchy = list(folder_hierarchy or [])
        joined = "/".join(hierarchy)

        # Explicit subspace id
        node = None
        if subspace and subspace in nodes:
            node = nodes[subspace]
        elif filter_path:
            # Prefer exact filter_path / label match on taxonomy nodes
            for n in nodes.values():
                if not isinstance(n, dict):
                    continue
                if filter_path in {n.get("filter_path"), n.get("label_fa")}:
                    node = n
                    break

        if node and isinstance(node, dict):
            tokens = [str(t) for t in (node.get("folder_tokens") or [])]
            domains = [str(d) for d in (node.get("domains") or [])]
            fp = str(node.get("filter_path") or "")
            if fp and (fp in joined or fp in hierarchy or any(fp in h for h in hierarchy)):
                return True
            if tokens and any(tok in h for tok in tokens for h in hierarchy):
                return True
            if domain and domains and domain in domains:
                return True
            # Fall through to legacy substring if taxonomy predicates miss

        # Legacy compatibility: substring membership on hierarchy
        needle = filter_path or subspace or ""
        if not needle:
            return True
        if "/" in needle:
            return needle in joined or all(part in hierarchy for part in needle.split("/") if part)
        return needle in hierarchy or any(needle in h for h in hierarchy)

    def detect_schema(self, headers: list[str], schemas_version: str | None = None) -> str | None:
        """Detect sheet schema id from column headers using the schemas pack."""
        pack = self.schemas_pack(schemas_version)
        normalized = {_normalize_col(h) for h in headers if h}
        schemas = pack.get("schemas") or {}

        crm = schemas.get("crm_qa") or {}
        markers = {_normalize_col(m) for m in (crm.get("persian_qa_markers") or [])}
        if len(normalized & markers) >= 2 or ({"پرسش"} & normalized and {"پاسخ"} & normalized):
            return "crm_qa"
        if {"سوال"} & normalized and {"پاسخ"} & normalized:
            return "crm_qa"

        reason = schemas.get("reason_codes") or {}
        core = {_normalize_col(c) for c in (reason.get("core_normalized") or [])}
        require = {_normalize_col(c) for c in (reason.get("require_core") or [])}
        if require and require <= normalized and len(normalized & core) >= int(reason.get("min_core_overlap") or 3):
            return "reason_codes"

        for sid in ("glossary", "staff_profile", "loan_catalog", "timeline"):
            sch = schemas.get(sid) or {}
            cols = {_normalize_col(c) for c in (sch.get("columns") or [])}
            min_overlap = int(sch.get("min_overlap") or max(1, int(len(cols) * float(pack.get("match_threshold") or 0.6))))
            require_any = {_normalize_col(c) for c in (sch.get("require_normalized_any") or [])}
            exact_count = sch.get("exact_header_count")
            if exact_count is not None and len(headers) != int(exact_count):
                continue
            if require_any and not (normalized & require_any):
                continue
            if len(normalized & cols) >= min_overlap:
                return sid

        non_empty = [h for h in headers if h]
        kv = schemas.get("kv_pair") or {}
        if kv.get("fallback"):
            lo = int(kv.get("min_headers") or 2)
            hi = int(kv.get("max_headers") or 3)
            if lo <= len(non_empty) <= hi:
                return "kv_pair"

        threshold = float(pack.get("match_threshold") or 0.6)
        for sid in ("reason_codes", "crm_qa", "articles"):
            sch = schemas.get(sid) or {}
            required = {_normalize_col(c) for c in (sch.get("columns") or [])}
            # For crm_qa detection in threshold pass, use English-only columns
            if sid == "crm_qa":
                required = {_normalize_col(c) for c in ["question", "model", "briefanswer", "answer", "keyword"]}
            if not required:
                continue
            overlap = len(normalized & required)
            if overlap >= len(required) * threshold:
                return sid
        return None

    def schema_column_sets(self, schemas_version: str | None = None) -> dict[str, set[str]]:
        """Return schema_id -> set of original column names (for transparency UI)."""
        pack = self.schemas_pack(schemas_version)
        out: dict[str, set[str]] = {}
        for sid, sch in (pack.get("schemas") or {}).items():
            if isinstance(sch, dict):
                out[sid] = set(sch.get("columns") or [])
        return out

    def _kind_dir(self, kind: str, scope: str | None) -> Path:
        if kind == "prompts":
            return self.root / "prompts" / (scope or "orchestrator")
        return self.root / kind

    def _pack_path(self, kind: str, version: str, scope: str | None) -> Path:
        ver = version.lstrip("v")
        return self._kind_dir(kind, scope) / f"v{ver}.yaml"


_REGISTRY: Registry | None = None
_REGISTRY_LOCK = threading.Lock()


def get_registry() -> Registry:
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None:
            _REGISTRY = Registry()
        return _REGISTRY


def reset_registry() -> None:
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is not None:
            _REGISTRY.clear_cache()
        _REGISTRY = None


@lru_cache(maxsize=1)
def _cached_default_registry() -> Registry:
    return Registry()
