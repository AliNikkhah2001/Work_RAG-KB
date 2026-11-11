"""IndexManifest — immutable promotion unit tying corpus + packs + code."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kb_manager.config import PROJECT_ROOT
from kb_manager.registry.loader import get_registry

logger = logging.getLogger(__name__)

# Writable data dir (packs/ may be mounted read-only in containers).
MANIFESTS_DIR = PROJECT_ROOT / "data" / "manifests"
ACTIVE_POINTER = MANIFESTS_DIR / "ACTIVE.json"


@dataclass
class IndexManifest:
    manifest_id: str
    schema_version: str = "index-manifest/v1"
    created_at: str = ""
    code_git_sha: str = "unknown"
    corpus: dict[str, Any] = field(default_factory=dict)
    packs: dict[str, str] = field(default_factory=dict)
    embedding_model: str = ""
    embedding_dim: int = 0
    active: bool = False
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IndexManifest":
        return cls(
            manifest_id=str(data["manifest_id"]),
            schema_version=str(data.get("schema_version") or "index-manifest/v1"),
            created_at=str(data.get("created_at") or ""),
            code_git_sha=str(data.get("code_git_sha") or "unknown"),
            corpus=dict(data.get("corpus") or {}),
            packs=dict(data.get("packs") or {}),
            embedding_model=str(data.get("embedding_model") or ""),
            embedding_dim=int(data.get("embedding_dim") or 0),
            active=bool(data.get("active")),
            artifacts=dict(data.get("artifacts") or {}),
        )


def _git_head() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _dvc_rev(source_dir: str | None = None) -> str:
    """Best-effort DVC revision from kb-source (falls back to unknown)."""
    candidates = []
    if source_dir:
        candidates.append(Path(source_dir))
    candidates.append(PROJECT_ROOT.parent / "kb-source")
    candidates.append(PROJECT_ROOT / ".." / "kb-source")
    for root in candidates:
        root = root.resolve()
        if not root.exists():
            continue
        try:
            r = subprocess.run(
                ["dvc", "rev", "parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=str(root),
            )
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()[:12]
        except Exception:
            pass
        # Fallback: git HEAD of corpus repo
        try:
            r = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                cwd=str(root),
            )
            if r.returncode == 0 and r.stdout.strip():
                return f"git:{r.stdout.strip()}"
        except Exception:
            pass
    return "unknown"


def build_manifest(
    *,
    doc_count: int = 0,
    chunk_count: int = 0,
    source_dir: str | None = None,
    embedding_model: str | None = None,
    embedding_dim: int | None = None,
    activate: bool = False,
    artifacts: dict[str, Any] | None = None,
) -> IndexManifest:
    """Build a new IndexManifest from current registry pins + corpus stats."""
    reg = get_registry()
    pins = reg.active_pins()
    retrieval = {}
    try:
        retrieval = reg.retrieval_pack(pins.get("retrieval"))
    except Exception:
        pass
    now = datetime.now(UTC)
    mid = f"kb-{now.strftime('%Y%m%dT%H%M%SZ')}"
    manifest = IndexManifest(
        manifest_id=mid,
        created_at=now.isoformat(),
        code_git_sha=_git_head(),
        corpus={
            "dvc_rev": _dvc_rev(source_dir),
            "source_dir": source_dir or "",
            "doc_count": int(doc_count),
            "chunk_count": int(chunk_count),
        },
        packs=pins,
        embedding_model=embedding_model or str(retrieval.get("dense_model") or ""),
        embedding_dim=int(embedding_dim or retrieval.get("dense_dim") or 0),
        active=activate,
        artifacts=dict(artifacts or {}),
    )
    return manifest


def save_manifest(manifest: IndexManifest, *, activate: bool = False) -> Path:
    """Persist manifest JSON under packs/manifests/; optionally set ACTIVE pointer."""
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    path = MANIFESTS_DIR / f"{manifest.manifest_id}.json"
    data = manifest.to_dict()
    data["active"] = bool(activate or manifest.active)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    if activate or manifest.active:
        with ACTIVE_POINTER.open("w", encoding="utf-8") as fh:
            json.dump({"manifest_id": manifest.manifest_id, "path": str(path.name)}, fh, indent=2)
        # Clear active flag on other manifests (best-effort)
        for other in MANIFESTS_DIR.glob("kb-*.json"):
            if other == path:
                continue
            try:
                od = json.loads(other.read_text(encoding="utf-8"))
                if od.get("active"):
                    od["active"] = False
                    other.write_text(json.dumps(od, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                continue
    logger.info("Saved IndexManifest %s (active=%s)", manifest.manifest_id, activate or manifest.active)
    return path


def load_active_manifest() -> IndexManifest | None:
    if not ACTIVE_POINTER.is_file():
        return None
    try:
        ptr = json.loads(ACTIVE_POINTER.read_text(encoding="utf-8"))
        mid = ptr.get("manifest_id")
        path = MANIFESTS_DIR / (ptr.get("path") or f"{mid}.json")
        if not path.is_file():
            return None
        return IndexManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        logger.warning("Failed to load active manifest: %s", exc)
        return None


def list_manifests() -> list[dict[str, Any]]:
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    active = None
    if ACTIVE_POINTER.is_file():
        try:
            active = json.loads(ACTIVE_POINTER.read_text(encoding="utf-8")).get("manifest_id")
        except Exception:
            active = None
    out: list[dict[str, Any]] = []
    for path in sorted(MANIFESTS_DIR.glob("kb-*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["is_active"] = data.get("manifest_id") == active
            out.append(data)
        except Exception:
            continue
    return out
