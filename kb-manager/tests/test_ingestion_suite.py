"""Tests for the ingestion suite (kb_manager.web.routes.ingestion_suite)."""

from __future__ import annotations

import io
import json
import shutil
import zipfile
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from kb_manager.config import PROJECT_ROOT
from kb_manager.web.routes import ingestion_suite


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(ingestion_suite.router, prefix="/ingestion")
    return app


def _make_xlsx_bytes() -> bytes:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["question", "answer", "keyword"])
    ws.append(["سوال اول", "پاسخ اول", "کلید"])
    ws.append(["سوال دوم", "پاسخ دوم", "کلید۲"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _make_zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("data/نمونه.xlsx", _make_xlsx_bytes())
        z.writestr("data/یادداشت.txt", "سلام دنیا".encode("utf-8"))
    return buf.getvalue()


def _client() -> TestClient:
    return TestClient(_make_app())


def _create_session(client: TestClient) -> str:
    r = client.post(
        "/ingestion/sessions",
        files={"file": ("test-upload.zip", _make_zip_bytes(), "application/zip")},
    )
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]
    assert sid
    return sid


def _cleanup_session(sid: str) -> None:
    shutil.rmtree(PROJECT_ROOT / "data" / "uploads" / "ingestion" / sid, ignore_errors=True)
    for p in (PROJECT_ROOT / "data" / "kb_zips").glob(f"*-{sid}.zip"):
        p.unlink(missing_ok=True)


# ---------------------------------------------------------------------------

def test_session_create_and_list():
    client = _client()
    sid = _create_session(client)
    try:
        r = client.get("/ingestion/sessions")
        assert r.status_code == 200
        ids = [s["id"] for s in r.json()]
        assert sid in ids

        # session.json exists with expected keys
        cfg = json.loads(
            (PROJECT_ROOT / "data" / "uploads" / "ingestion" / sid / "session.json").read_text(encoding="utf-8")
        )
        assert cfg["id"] == sid
        assert set(("tags", "include", "columns", "sheet_cache")) <= set(cfg)
        # Persian filename preserved on extract
        assert (PROJECT_ROOT / "data" / "uploads" / "ingestion" / sid / "source" / "data" / "نمونه.xlsx").exists()
    finally:
        _cleanup_session(sid)


def test_tree_and_tree_ops():
    client = _client()
    sid = _create_session(client)
    try:
        r = client.get(f"/ingestion/sessions/{sid}/tree")
        assert r.status_code == 200
        root = r.json()
        assert root["type"] == "dir"
        assert any(c["name"] == "data" for c in root["children"])

        # mkdir
        r = client.post(f"/ingestion/sessions/{sid}/tree/mkdir", json={"path": "data/newdir"})
        assert r.status_code == 200, r.text

        # tag
        r = client.post(
            f"/ingestion/sessions/{sid}/tree/tag",
            json={"path": "data/یادداشت.txt", "tags": ["مهم"]},
        )
        assert r.status_code == 200

        # include (exclude a file)
        r = client.post(
            f"/ingestion/sessions/{sid}/tree/include",
            json={"path": "data/یادداشت.txt", "include": False},
        )
        assert r.status_code == 200

        # rename
        r = client.post(
            f"/ingestion/sessions/{sid}/tree/rename",
            json={"path": "data/newdir", "new_name": "renamed"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["path"] == "data/renamed"

        # verify tree reflects ops
        tree = client.get(f"/ingestion/sessions/{sid}/tree").json()

        def _find(node, path):
            if node["path"] == path:
                return node
            for c in node.get("children", []):
                hit = _find(c, path)
                if hit:
                    return hit
            return None

        assert _find(tree, "data/renamed") is not None
        note = _find(tree, "data/یادداشت.txt")
        assert note is not None
        assert note["include"] is False
        assert note["tags"] == ["مهم"]

        # delete
        r = client.post(
            f"/ingestion/sessions/{sid}/tree/delete", json={"path": "data/یادداشت.txt"}
        )
        assert r.status_code == 200
        tree = client.get(f"/ingestion/sessions/{sid}/tree").json()
        assert _find(tree, "data/یادداشت.txt") is None
    finally:
        _cleanup_session(sid)


def test_config_roundtrip():
    client = _client()
    sid = _create_session(client)
    try:
        payload = {
            "include": {"data/نمونه.xlsx": True},
            "columns": {"data/نمونه.xlsx": {"Sheet1": {"schema_override": "crm_qa", "target_columns": ["question", "answer"]}}},
            "tags": {"data/نمونه.xlsx": ["qa"]},
        }
        r = client.put(f"/ingestion/sessions/{sid}/config", json=payload)
        assert r.status_code == 200, r.text
        r = client.get(f"/ingestion/sessions/{sid}/config")
        assert r.status_code == 200
        body = r.json()
        assert body["columns"]["data/نمونه.xlsx"]["Sheet1"]["target_columns"] == ["question", "answer"]
        assert body["tags"] == {"data/نمونه.xlsx": ["qa"]}
    finally:
        _cleanup_session(sid)


def test_xlsx_preview_and_edit():
    client = _client()
    sid = _create_session(client)
    try:
        r = client.get(f"/ingestion/sessions/{sid}/file", params={"path": "data/نمونه.xlsx"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["type"] == "xlsx"
        sheet = next(s for s in body["sheets"] if s["name"] == "Sheet1")
        assert sheet["headers"] == ["question", "answer", "keyword"]
        assert len(sheet["rows_preview"]) == 2
        assert "سوال اول" in sheet["rows_preview"][0]

        # edit: rewrite headers+rows (Persian-safe)
        r = client.put(
            f"/ingestion/sessions/{sid}/file",
            params={"path": "data/نمونه.xlsx"},
            json={"sheets": {"Sheet1": {"headers": ["question", "answer"], "rows": [["پرسش جدید", "پاسخ جدید"]]}}},
        )
        assert r.status_code == 200, r.text

        r = client.get(f"/ingestion/sessions/{sid}/file", params={"path": "data/نمونه.xlsx"})
        sheet = next(s for s in r.json()["sheets"] if s["name"] == "Sheet1")
        assert sheet["headers"] == ["question", "answer"]
        assert sheet["rows_preview"][0] == ["پرسش جدید", "پاسخ جدید"]
    finally:
        _cleanup_session(sid)


def test_pdf_edit_not_supported():
    import fitz  # PyMuPDF

    client = _client()
    # build a session containing a pdf
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "hello pdf")
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.pdf", buf.getvalue())
    zbuf.seek(0)
    r = client.post(
        "/ingestion/sessions",
        files={"file": ("with-pdf.zip", zbuf.getvalue(), "application/zip")},
    )
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]
    try:
        r = client.put(
            "/ingestion/sessions/{}/file".format(sid),
            params={"path": "doc.pdf"},
            json={"text": "new"},
        )
        assert r.status_code == 400
        assert "PDF" in r.json()["detail"]
    finally:
        _cleanup_session(sid)


def test_save_zip():
    client = _client()
    sid = _create_session(client)
    try:
        r = client.post(f"/ingestion/sessions/{sid}/save-zip")
        assert r.status_code == 200, r.text
        body = r.json()
        zp = Path(body["zip_path"])
        assert zp.exists()
        assert body["size"] > 0
        with zipfile.ZipFile(str(zp)) as z:
            names = z.namelist()
        assert "data/یادداشت.txt" in names
    finally:
        _cleanup_session(sid)


def test_column_filter_helper():
    from kb_manager.web.routes.ingestion_suite import filter_sheet_data, get_sheet_config

    sheet = {"name": "S", "headers": ["a", "b", "c"], "rows": [["1", "2", "3"]], "schema": None}
    out = filter_sheet_data(sheet, target_columns=["c", "a"], schema_override="crm_qa")
    assert out["headers"] == ["c", "a"]
    assert out["rows"] == [["3", "1"]]
    assert out["schema"] == "crm_qa"
    # no-op when no filters
    out2 = filter_sheet_data(sheet)
    assert out2["headers"] == ["a", "b", "c"]

    cfg = {"columns": {"f.xlsx": {"S": {"target_columns": ["a"], "schema_override": None}}}}
    assert get_sheet_config(cfg, "f.xlsx", "S") == {"target_columns": ["a"], "schema_override": None}
    assert get_sheet_config(cfg, "f.xlsx", "Missing") == {}
    assert get_sheet_config({}, "f.xlsx", "S") == {}


def test_flat_xlsx_save_payload():
    client = _client()
    sid = _create_session(client)
    try:
        r = client.get(f"/ingestion/sessions/{sid}/file", params={"path": "data/نمونه.xlsx"})
        assert r.status_code == 200, r.text
        sheet = next(s for s in r.json()["sheets"] if s["name"] == "Sheet1")
        # flat single-sheet payload, as sent by the file editor
        r = client.put(
            f"/ingestion/sessions/{sid}/file",
            params={"path": "data/نمونه.xlsx"},
            json={"path": "data/نمونه.xlsx", "sheet": "Sheet1", "headers": sheet["headers"], "rows": sheet["rows_preview"]},
        )
        assert r.status_code == 200, r.text
        assert r.json()["ok"] is True
    finally:
        _cleanup_session(sid)


def test_chunking_view_keeps_partial_rows():
    client = _client()
    sid = _create_session(client)
    try:
        r = client.get(f"/ingestion/sessions/{sid}/file/chunking", params={"path": "data/نمونه.xlsx"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["sheets"], "expected at least one sheet"
        sheet = body["sheets"][0]
        assert "field_mappings" in sheet and sheet["field_mappings"]
        assert all("header" in m and "label" in m for m in sheet["field_mappings"])
        # partial rows must be kept: only fully-empty rows skipped
        assert all(not c["skipped"] for c in sheet["chunks"])
        assert sheet["skipped_rows"] == 0
    finally:
        _cleanup_session(sid)


def test_review_gate_flow():
    client = _client()
    sid = _create_session(client)
    try:
        r = client.get(f"/ingestion/sessions/{sid}/review")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["done"] is False
        assert body["total"] >= 2
        first = body["remaining"][0]
        # approve one file
        r = client.post(f"/ingestion/sessions/{sid}/review", json={"path": first, "approved": True})
        assert r.status_code == 200, r.text
        assert r.json()["reviewed_count"] == 1
        # approve the rest -> done
        r = client.get(f"/ingestion/sessions/{sid}/review")
        for rel in r.json()["remaining"]:
            rr = client.post(f"/ingestion/sessions/{sid}/review", json={"path": rel, "approved": True})
            assert rr.status_code == 200, rr.text
        r = client.get(f"/ingestion/sessions/{sid}/review")
        assert r.json()["done"] is True
        # unapprove reopens the gate
        r = client.post(f"/ingestion/sessions/{sid}/review", json={"path": first, "approved": False})
        assert r.status_code == 200, r.text
        assert r.json()["done"] is False
    finally:
        _cleanup_session(sid)
