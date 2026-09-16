"""Tests for historical KB manager (kb_manager.web.routes.kb_history)."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from kb_manager.web.routes import kb_history


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(kb_history.router, prefix="/ingestion")
    return app


def _client() -> TestClient:
    return TestClient(_make_app())


def _seed_snapshot(versions_root: Path, label: str, notes: str = "یادداشت فارسی") -> Path:
    d = versions_root / label
    d.mkdir(parents=True, exist_ok=True)
    manifest = {
        "label": label,
        "created_at": "2026-09-14T00:00:00+00:00",
        "notes": notes,
        "counts": {"documents": 2, "chunks_total": 10},
        "source_db": "",
    }
    (d / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    (d / "kb_export.json").write_text("{}", encoding="utf-8")
    return d


def test_sanitize_label_rejects_traversal():
    import pytest

    with pytest.raises(ValueError):
        kb_history.sanitize_label("../evil")
    with pytest.raises(ValueError):
        kb_history.sanitize_label("a/b")
    assert kb_history.sanitize_label("v8 new") == "v8_new"


def test_list_kbs_empty_ok(tmp_path):
    vroot = tmp_path / "versions"
    ddir = tmp_path / "data"
    zdir = tmp_path / "zips"
    vroot.mkdir()
    ddir.mkdir()
    items = kb_history.list_kbs(versions_root=vroot, data_dir=ddir, zips_dir=zdir)
    assert items == []
    assert zdir.exists()  # auto-created


def test_snapshot_roundtrip_and_notes(tmp_path):
    vroot = tmp_path / "versions"
    vroot.mkdir()
    _seed_snapshot(vroot, "v_test")
    items = kb_history.list_kbs(versions_root=vroot, data_dir=tmp_path, zips_dir=tmp_path / "z")
    assert any(i["label"] == "v_test" and i["type"] == "snapshot" for i in items)
    detail = kb_history.get_kb_detail("v_test", versions_root=vroot, data_dir=tmp_path, zips_dir=tmp_path / "z")
    assert detail is not None and detail["notes"] == "یادداشت فارسی"
    assert "manifest.json" in detail["files"]
    manifest = kb_history.update_kb_notes("v_test", "ویرایش شده", versions_root=vroot)
    assert manifest["notes"] == "ویرایش شده"
    # Persian preserved verbatim (no ascii mangling)
    raw = (vroot / "v_test" / "manifest.json").read_text(encoding="utf-8")
    assert "ویرایش شده" in raw


def test_repack_creates_zip(tmp_path):
    vroot = tmp_path / "versions"
    ddir = tmp_path / "data"
    zdir = tmp_path / "zips"
    vroot.mkdir()
    ddir.mkdir()
    _seed_snapshot(vroot, "v_pack")
    zp = kb_history.repack_kb("v_pack", versions_root=vroot, data_dir=ddir, zips_dir=zdir)
    assert zp.exists() and zp.suffix == ".zip"
    import zipfile

    with zipfile.ZipFile(zp) as z:
        assert "manifest.json" in z.namelist()


def test_kbs_api_routes(tmp_path, monkeypatch):
    vroot = tmp_path / "versions"
    vroot.mkdir()
    _seed_snapshot(vroot, "v_api")
    monkeypatch.setattr(kb_history.snap_mod, "VERSIONS_ROOT", str(vroot))
    monkeypatch.setattr(kb_history, "DATA_DIR", tmp_path)
    monkeypatch.setattr(kb_history, "KB_ZIPS_DIR", tmp_path / "zips")
    client = _client()
    r = client.get("/ingestion/kbs")
    assert r.status_code == 200
    assert any(i["label"] == "v_api" for i in r.json())
    r = client.get("/ingestion/kbs/v_api")
    assert r.status_code == 200
    assert r.json()["label"] == "v_api"
    r = client.put("/ingestion/kbs/v_api", json={"notes": "به‌روزرسانی"})
    assert r.status_code == 200
    r = client.post("/ingestion/kbs/v_api/repack")
    assert r.status_code == 200
    assert Path(r.json()["zip_path"]).exists()
