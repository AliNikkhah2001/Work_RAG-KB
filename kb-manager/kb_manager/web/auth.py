"""Optional API-key gate for the KB Manager.

Disabled unless ``KB_API_KEY`` is set, so every existing caller (the orchestrator,
the admin UI, local scripts) keeps working unchanged. When a key *is* set, every
request must present it and the health/metrics paths stay open so probes keep
working.

Accepted credentials, any one of:
    - ``X-API-Key: <key>``
    - ``Authorization: Bearer <key>``
    - ``?api_key=<key>``  (escape hatch for the browser UI and curl links)

This is a shared secret, not per-user auth. It has no rotation, expiry or scopes —
it closes the "anyone on the network can read and write the corpus" hole.
"""

from __future__ import annotations

import hmac
import logging
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

log = logging.getLogger(__name__)

# Reachable without a key so container/orchestrator probes never fail on auth.
_PUBLIC_PATHS = frozenset({"/health", "/api/health", "/ready", "/metrics", "/metrics/healthz"})
_PUBLIC_PREFIXES = ("/static/",)


def api_key() -> str:
    """The configured key, or an empty string when auth is disabled."""
    return (os.getenv("KB_API_KEY") or "").strip()


def _presented_key(request) -> str:
    header = (request.headers.get("x-api-key") or "").strip()
    if header:
        return header
    auth = (request.headers.get("authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (request.query_params.get("api_key") or "").strip()


def _is_public(path: str) -> bool:
    if path in _PUBLIC_PATHS:
        return True
    return any(path.startswith(p) for p in _PUBLIC_PREFIXES)


class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        expected = api_key()
        if not expected:
            return await call_next(request)
        path = request.url.path
        if _is_public(path):
            return await call_next(request)
        provided = _presented_key(request)
        # compare_digest: constant-time, so a wrong key leaks nothing by timing.
        if provided and hmac.compare_digest(provided, expected):
            return await call_next(request)
        log.warning("rejected %s %s: missing or invalid API key", request.method, path)
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing API key"},
            headers={"WWW-Authenticate": "Bearer"},
        )
