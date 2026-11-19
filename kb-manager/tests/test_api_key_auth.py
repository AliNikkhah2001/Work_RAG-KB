"""KB Manager API-key gate.

The gate is opt-in: unset KB_API_KEY keeps every endpoint open (current
behaviour), and setting it requires the key on everything except probes.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kb_manager.web.auth import ApiKeyMiddleware

KEY = "test-key-123"


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(ApiKeyMiddleware)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/registry/api/active")
    async def active():
        return {"pins": {}}

    @app.post("/search/api")
    async def search():
        return {"final_results": []}

    return app


@pytest.fixture
def disabled(monkeypatch):
    monkeypatch.delenv("KB_API_KEY", raising=False)
    return TestClient(_app())


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("KB_API_KEY", KEY)
    return TestClient(_app())


# --- disabled: nothing changes for existing callers -------------------------

def test_disabled_keeps_endpoints_open(disabled):
    assert disabled.get("/registry/api/active").status_code == 200
    assert disabled.post("/search/api").status_code == 200


def test_disabled_ignores_wrong_key(disabled):
    r = disabled.get("/registry/api/active", headers={"X-API-Key": "nope"})
    assert r.status_code == 200


# --- enabled: the key is required ------------------------------------------

def test_enabled_rejects_missing_key(enabled):
    r = enabled.get("/registry/api/active")
    assert r.status_code == 401
    assert "API key" in r.json()["detail"]
    assert r.headers["WWW-Authenticate"] == "Bearer"


def test_enabled_rejects_wrong_key(enabled):
    assert enabled.get("/registry/api/active", headers={"X-API-Key": "wrong"}).status_code == 401


def test_enabled_accepts_x_api_key(enabled):
    assert enabled.get("/registry/api/active", headers={"X-API-Key": KEY}).status_code == 200


def test_enabled_accepts_bearer(enabled):
    r = enabled.get("/registry/api/active", headers={"Authorization": f"Bearer {KEY}"})
    assert r.status_code == 200


def test_enabled_accepts_query_param(enabled):
    assert enabled.get(f"/registry/api/active?api_key={KEY}").status_code == 200


def test_enabled_protects_post(enabled):
    assert enabled.post("/search/api").status_code == 401
    assert enabled.post("/search/api", headers={"X-API-Key": KEY}).status_code == 200


def test_health_and_ready_stay_public_when_enabled(enabled):
    assert enabled.get("/health").status_code == 200
