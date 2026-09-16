"""Tests for kb_manager.query_enhance + POST /search/enhance."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.testclient import TestClient

from kb_manager.query_enhance import enhance_query, maybe_llm_rewrite
from kb_manager.web.routes.search import router


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/search")
    return TestClient(app)


def test_expansion_applied():
    res = enhance_query("میشه وام بگیرم؟")
    assert res["enhanced"] != "میشه وام بگیرم؟"
    assert "می‌شود" in res["enhanced"]
    assert isinstance(res["beams"], list)


def test_beams_deduped_and_capped_at_5():
    res = enhance_query("امتیاز اعتباری من چقدر است؟")
    beams = res["beams"]
    assert len(beams) <= 5
    assert len(beams) == len(set(beams))
    assert len(beams) >= 1


def test_original_first():
    q = "  وام   بانکی  "
    res = enhance_query(q)
    assert res["beams"][0] == q


def test_empty_query_handled():
    assert enhance_query("") == {"enhanced": "", "beams": []}
    assert enhance_query("   ") == {"enhanced": "", "beams": []}


def test_enhance_endpoint_returns_200_with_beams():
    client = _client()
    resp = client.post("/search/enhance", json={"query": "میشه وام بگیرم؟"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "میشه وام بگیرم؟"
    assert "enhanced" in body and "beams" in body
    assert isinstance(body["beams"], list)
    assert body["beams"][0] == "میشه وام بگیرم؟"


def test_maybe_llm_rewrite_mock_gated():
    os.environ.pop("KB_ALLOW_MOCK", None)
    assert maybe_llm_rewrite("سلام") is None
    os.environ["KB_ALLOW_MOCK"] = "true"
    try:
        out = maybe_llm_rewrite("سلام")
        assert out is not None and "سلام" in out
        assert maybe_llm_rewrite("سلام", allow_mock_only=False) is None
    finally:
        os.environ.pop("KB_ALLOW_MOCK", None)
