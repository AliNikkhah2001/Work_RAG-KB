"""Historical KB listing / editing + data-repo zip persistence.

Combines three sources into one ``/ingestion/kbs`` API:

* immutable snapshots in ``versions/<label>/`` (see
  :mod:`kb_manager.versioning.snapshot`),
* live DB files ``data/kb_*.db``,
* persisted source zips ``data/kb_zips/*.zip``.

All JSON output preserves Persian text (``ensure_ascii=False`` on writes;
Starlette ``JSONResponse`` already emits UTF-8).
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from kb_manager.config import PROJECT_ROOT
from kb_manager.versioning import snapshot as snap_mod

router = APIRouter()

DATA_DIR = PROJECT_ROOT / "data"
KB_ZIPS_DIR = DATA_DIR / "kb_zips"


# ---------------------------------------------------------------------------
# Helpers (exposed for testing with tmp_path overrides)
# ---------------------------------------------------------------------------

def ensure_kb_zips_dir(zips_dir: Path | None = None) -> Path:
    """Create (if needed) and return the kb_zips directory."""
    target = Path(zips_dir) if zips_dir is not None else KB_ZIPS_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target


def _resolve_versions_root(versions_root: Path | str | None = None) -> Path:
    if versions_root is not None:
        return Path(versions_root)
    return Path(snap_mod.VERSIONS_ROOT)


def _resolve_data_dir(data_dir: Path | str | None = None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    return DATA_DIR


def _resolve_zips_dir(zips_dir: Path | str | None = None) -> Path:
    if zips_dir is not None:
        return Path(zips_dir)
    return KB_ZIPS_DIR


def sanitize_label(label: str) -> str:
    """Normalize + validate a KB label (path-traversal safe)."""
    if not isinstance(label, str):
        raise ValueError("Label must be a string")
    cleaned = label.strip().replace(" ", "_")
    if not cleaned:
        raise ValueError("Label required")
    if cleaned in (".", "..") or ".." in cleaned or "/" in cleaned or "\\" in cleaned:
        raise ValueError(f"Invalid label: {label!r}")
    # Resolve-against-root check (defence in depth).
    if (Path(cleaned).name != cleaned):
        raise ValueError(f"Invalid label: {label!r}")
    return cleaned


def count_db_docs_chunks(db_path: Path | str) -> dict[str, Any]:
    """Best-effort ``{docs, chunks}`` counts for a sqlite KB file.

    Returns ``{"docs": int|None, "chunks": int|None}``; ``None`` when the
    file is missing or the query fails (guarded by try/except).
    """
    try:
        p = Path(db_path)
        if not p.exists() or not p.is_file():
            return {"docs": None, "chunks": None}
        conn = sqlite3.connect(str(p))
        try:
            cur = conn.cursor()
            docs = chunks = None
            with contextlib.suppress(Exception):
                cur.execute("SELECT COUNT(*) FROM documents")
                row = cur.fetchone()
                docs = int(row[0]) if row else 0
            with contextlib.suppress(Exception):
                cur.execute("SELECT COUNT(*) FROM chunks")
                row = cur.fetchone()
                chunks = int(row[0]) if row else 0
            return {"docs": docs, "chunks": chunks}
        finally:
            conn.close()
    except Exception:
        return {"docs": None, "chunks": None}


def _file_stat(path: Path) -> dict[str, Any]:
    try:
        st = path.stat()
        mtime_iso = datetime.fromtimestamp(st.st_mtime, tz=UTC).isoformat()
        return {"size": st.st_size, "mtime": mtime_iso, "created_at": mtime_iso}
    except Exception:
        return {"size": None, "mtime": None, "created_at": ""}


def _list_snapshots_from(versions_root: Path) -> list[dict[str, Any]]:
    """Mirror of ``snapshot.list_snapshots()`` rooted at ``versions_root``."""
    if not versions_root.exists():
        return []
    snapshots: list[dict[str, Any]] = []
    for d in sorted(versions_root.iterdir()):
        if not d.is_dir():
            continue
        manifest_path = d / "manifest.json"
        info: dict[str, Any] = {"label": d.name, "created_at": "", "git_commit": "", "counts": {}}
        if manifest_path.exists():
            with contextlib.suppress(Exception):
                info.update(json.loads(manifest_path.read_text(encoding="utf-8")))
        snapshots.append(info)
    snapshots.sort(key=lambda s: s.get("created_at", ""), reverse=True)
    return snapshots


def list_kbs(
    versions_root: Path | str | None = None,
    data_dir: Path | str | None = None,
    zips_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Combine snapshots + ``kb_*.db`` files + ``kb_zips/*.zip`` into one list."""
    vroot = _resolve_versions_root(versions_root)
    ddir = _resolve_data_dir(data_dir)
    zdir = _resolve_zips_dir(zips_dir)
    with contextlib.suppress(Exception):
        zdir.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = []

    # 1) snapshots in versions/<label>/
    for snap in _list_snapshots_from(vroot):
        label = str(snap.get("label") or snap.get("version") or "")
        if not label:
            continue
        counts = snap.get("counts") or {}
        docs = counts.get("documents", counts.get("docs"))
        chunks = counts.get("chunks_total", counts.get("chunks"))
        source_db = snap.get("source_db") or ""
        db_candidate = (ddir / Path(str(source_db)).name) if source_db else None
        db_path = str(db_candidate) if db_candidate and db_candidate.exists() else None
        zip_candidate = zdir / f"{label}.zip"
        zip_path = str(zip_candidate) if zip_candidate.exists() else None
        items.append(
            {
                "id": label,
                "label": label,
                "type": "snapshot",
                "created_at": snap.get("created_at", ""),
                "notes": snap.get("notes", ""),
                "db_path": db_path,
                "zip_path": zip_path,
                "docs": docs,
                "chunks": chunks,
            }
        )

    # 2) live DB files data/kb_*.db
    if ddir.exists():
        for db_file in sorted(ddir.glob("kb_*.db")):
            if not db_file.is_file():
                continue
            stat = _file_stat(db_file)
            counts = count_db_docs_chunks(db_file)
            zip_candidate = zdir / f"{db_file.stem}.zip"
            items.append(
                {
                    "id": db_file.stem,
                    "label": db_file.stem,
                    "type": "db",
                    "created_at": stat["created_at"],
                    "notes": "",
                    "db_path": str(db_file),
                    "zip_path": str(zip_candidate) if zip_candidate.exists() else None,
                    "docs": counts["docs"],
                    "chunks": counts["chunks"],
                    "size": stat["size"],
                    "mtime": stat["mtime"],
                }
            )

    # 3) persisted zips data/kb_zips/*.zip
    if zdir.exists():
        for zf in sorted(zdir.glob("*.zip")):
            if not zf.is_file():
                continue
            stat = _file_stat(zf)
            db_candidate = ddir / f"{zf.stem}.db"
            # also match kb_*.db naming: stem already includes kb_ prefix usually
            items.append(
                {
                    "id": zf.stem,
                    "label": zf.stem,
                    "type": "zip",
                    "created_at": stat["created_at"],
                    "notes": "",
                    "db_path": str(db_candidate) if db_candidate.exists() else None,
                    "zip_path": str(zf),
                    "docs": None,
                    "chunks": None,
                    "size": stat["size"],
                    "mtime": stat["mtime"],
                }
            )

    return items


def get_kb_detail(
    label: str,
    versions_root: Path | str | None = None,
    data_dir: Path | str | None = None,
    zips_dir: Path | str | None = None,
) -> dict[str, Any] | None:
    """Detail for one snapshot label (manifest + files + db/zip linkage)."""
    clean = sanitize_label(label)
    vroot = _resolve_versions_root(versions_root)
    ddir = _resolve_data_dir(data_dir)
    zdir = _resolve_zips_dir(zips_dir)
    snap_dir = vroot / clean
    # Containment check.
    try:
        resolved = snap_dir.resolve()
        if resolved.parent != vroot.resolve():
            return None
    except Exception:
        return None
    if not snap_dir.exists() or not snap_dir.is_dir():
        return None

    manifest: dict[str, Any] = {}
    manifest_path = snap_dir / "manifest.json"
    if manifest_path.exists():
        with contextlib.suppress(Exception):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    files = sorted(p.name for p in snap_dir.glob("*") if p.exists())
    counts = manifest.get("counts", {}) if isinstance(manifest, dict) else {}
    docs = counts.get("documents", counts.get("docs"))
    chunks = counts.get("chunks_total", counts.get("chunks"))

    source_db = manifest.get("source_db", "") if isinstance(manifest, dict) else ""
    db_candidate = (ddir / Path(str(source_db)).name) if source_db else None
    db_path = str(db_candidate) if db_candidate and db_candidate.exists() else None
    # Fall back: a db file matching the label.
    if db_path is None:
        cand2 = ddir / f"{clean}.db"
        if cand2.exists():
            db_path = str(cand2)
    zip_candidate = zdir / f"{clean}.zip"
    zip_path = str(zip_candidate) if zip_candidate.exists() else None

    db_counts = count_db_docs_chunks(db_path) if db_path else {"docs": None, "chunks": None}

    return {
        "id": clean,
        "label": clean,
        "type": "snapshot",
        "created_at": manifest.get("created_at", "") if isinstance(manifest, dict) else "",
        "notes": manifest.get("notes", "") if isinstance(manifest, dict) else "",
        "manifest": manifest,
        "files": files,
        "db_path": db_path,
        "zip_path": zip_path,
        "docs": docs if docs is not None else db_counts["docs"],
        "chunks": chunks if chunks is not None else db_counts["chunks"],
    }


def update_kb_notes(
    label: str,
    notes: str,
    versions_root: Path | str | None = None,
) -> dict[str, Any]:
    """Update ``manifest.json`` notes for a snapshot; returns the manifest."""
    clean = sanitize_label(label)
    vroot = _resolve_versions_root(versions_root)
    snap_dir = vroot / clean
    try:
        resolved = snap_dir.resolve()
        if resolved.parent != vroot.resolve():
            raise ValueError(f"Invalid label: {label!r}")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Invalid label: {label!r}") from exc
    manifest_path = snap_dir / "manifest.json"
    if not snap_dir.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {clean}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["notes"] = notes or ""
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def repack_kb(
    label: str,
    versions_root: Path | str | None = None,
    data_dir: Path | str | None = None,
    zips_dir: Path | str | None = None,
) -> Path:
    """Zip ``versions/<label>/`` + associated DB into ``kb_zips/<label>.zip``."""
    clean = sanitize_label(label)
    vroot = _resolve_versions_root(versions_root)
    ddir = _resolve_data_dir(data_dir)
    zdir = ensure_kb_zips_dir(_resolve_zips_dir(zips_dir))
    snap_dir = vroot / clean
    try:
        resolved = snap_dir.resolve()
        if resolved.parent != vroot.resolve():
            raise ValueError(f"Invalid label: {label!r}")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Invalid label: {label!r}") from exc
    if not snap_dir.exists() or not snap_dir.is_dir():
        raise FileNotFoundError(f"Snapshot not found: {clean}")

    # Associated DB from manifest source_db (best-effort).
    db_to_pack: Path | None = None
    manifest_path = snap_dir / "manifest.json"
    if manifest_path.exists():
        with contextlib.suppress(Exception):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            src = manifest.get("source_db", "")
            if src:
                cand = ddir / Path(str(src)).name
                if cand.exists() and cand.is_file():
                    db_to_pack = cand
    if db_to_pack is None:
        cand2 = ddir / f"{clean}.db"
        if cand2.exists() and cand2.is_file():
            db_to_pack = cand2

    zip_path = zdir / f"{clean}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(snap_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(snap_dir).as_posix())
        if db_to_pack is not None:
            # Store DB at zip root under its own file name.
            zf.write(db_to_pack, arcname=db_to_pack.name)
    return zip_path


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class KbCreate(BaseModel):
    label: str
    db_path: str | None = None
    zip_path: str | None = None
    notes: str = ""


class KbUpdate(BaseModel):
    notes: str = ""


# ---------------------------------------------------------------------------
# Routes (mounted under /ingestion in app.py)
# ---------------------------------------------------------------------------

@router.get("/kbs")
async def list_kbs_route():
    """List snapshots + kb_*.db files + kb_zips/*.zip (Persian-safe JSON)."""
    items = list_kbs()
    return JSONResponse(items, media_type="application/json; charset=utf-8")


@router.post("/kbs")
async def create_kb_route(payload: KbCreate):
    """Create a snapshot via ``create_snapshot(label, notes, db_path)``."""
    try:
        label = sanitize_label(payload.label)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    notes = payload.notes or ""
    db_arg: str | None = None
    if payload.db_path:
        cand = Path(payload.db_path)
        candidates = [cand]
        # Allow paths relative to data dir / project root for convenience.
        if not cand.is_absolute():
            candidates.append(DATA_DIR / cand)
            candidates.append(PROJECT_ROOT / cand)
        found: Path | None = next((c for c in candidates if c.exists() and c.is_file()), None)
        if found is None:
            raise HTTPException(status_code=404, detail=f"db_path not found: {payload.db_path}")
        db_arg = str(found)

    zip_echo: str | None = None
    if payload.zip_path:
        zc = Path(payload.zip_path)
        zcands = [zc]
        if not zc.is_absolute():
            zcands.append(DATA_DIR / zc)
            zcands.append(PROJECT_ROOT / zc)
        zfound = next((c for c in zcands if c.exists()), None)
        if zfound is None:
            raise HTTPException(status_code=404, detail=f"zip_path not found: {payload.zip_path}")
        zip_echo = str(zfound)

    try:
        target = snap_mod.create_snapshot(label, notes=notes, db_path=db_arg) if db_arg else snap_mod.create_snapshot(label, notes=notes)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)[:500]) from exc

    ensure_kb_zips_dir()
    return JSONResponse(
        {
            "id": label,
            "label": label,
            "snapshot_path": str(target),
            "db_path": db_arg,
            "zip_path": zip_echo,
            "notes": notes,
        },
        media_type="application/json; charset=utf-8",
    )


@router.get("/kbs/{label}")
async def get_kb_route(label: str):
    try:
        clean = sanitize_label(label)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    detail = get_kb_detail(clean)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Snapshot not found: {clean}")
    return JSONResponse(detail, media_type="application/json; charset=utf-8")


@router.put("/kbs/{label}")
async def update_kb_route(label: str, payload: KbUpdate):
    try:
        clean = sanitize_label(label)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        manifest = update_kb_notes(clean, payload.notes or "")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(
        {"id": clean, "label": clean, "manifest": manifest},
        media_type="application/json; charset=utf-8",
    )


@router.post("/kbs/{label}/repack")
async def repack_kb_route(label: str):
    try:
        clean = sanitize_label(label)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        zip_path = repack_kb(clean)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)[:500]) from exc
    size = None
    with contextlib.suppress(Exception):
        size = zip_path.stat().st_size
    return JSONResponse(
        {"id": clean, "label": clean, "zip_path": str(zip_path), "size": size},
        media_type="application/json; charset=utf-8",
    )
