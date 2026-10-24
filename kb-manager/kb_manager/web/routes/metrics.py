"""Prometheus metrics for KB search observability.

Exposes ``GET /metrics`` in Prometheus text exposition format so the already
polling Prometheus server stops receiving 404s. Uses a *private*
``CollectorRegistry`` (not the prometheus_client global default) so importing
this module twice — e.g. under pytest with re-imported route modules — can
never raise "duplicate timeseries" errors.

The search pipeline calls :func:`record_stage_ms` (and the other ``record_*``
helpers) as it progresses; every helper is O(1), safe to call before any
route registration, and must never raise — observability must not break the
request it is observing.
"""

from __future__ import annotations

import statistics
from collections import deque
from typing import Final

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

router = APIRouter(tags=["metrics"])

# prometheus_client 0.26 advertises "version=1.0.0" in CONTENT_TYPE_LATEST,
# but our Prometheus server is configured for the 0.0.4 exposition dialect —
# both are wire-compatible, so we pin the header the scraper expects.
PROMETHEUS_CONTENT_TYPE: Final[str] = "text/plain; version=0.0.4; charset=utf-8"

# Private registry: decouples us from the global default registry and makes
# the module safe to re-import in tests.
registry = CollectorRegistry()

# Latency buckets spanning the observed range of KB search stages: sub-10ms
# BM25/RRF work, 10-100ms dense/cache hits, up to multi-second rerank/cold
# first-load paths (rerank alone measured at ~6.5s in production).
STAGE_BUCKETS: Final[tuple[float, ...]] = (
    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 5.0,
)

STAGES: Final[tuple[str, ...]] = (
    "bm25", "dense", "rrf", "rerank", "hyde", "build_context", "total",
)

OUTCOMES: Final[tuple[str, ...]] = ("success", "error")

CACHE_KINDS: Final[tuple[str, ...]] = ("redis", "memory", "semantic", "none")

# Millisecond ring buffers powering /metrics/healthz p50s — maxlen keeps each
# append O(1) and bounds memory, and 100 samples is plenty for a live p50.
_RING_SIZE: Final[int] = 100

stage_histogram = Histogram(
    "kb_search_stage_seconds",
    "Latency of individual KB search pipeline stages, in seconds",
    labelnames=("stage",),
    buckets=STAGE_BUCKETS,
    registry=registry,
)

request_counter = Counter(
    "kb_search_requests_total",
    "Total KB search requests by outcome",
    labelnames=("outcome",),
    registry=registry,
)

result_counter = Counter(
    "kb_search_result_count",
    "Cumulative sum of results returned across KB searches (summary counter, not a per-request gauge)",
    registry=registry,
)

cache_hit_counter = Counter(
    "kb_search_cache_hits_total",
    "KB search cache hits by cache layer",
    labelnames=("cache",),
    registry=registry,
)

# Pre-create every known label child at import time so the hot path
# (record_stage_ms inside a live search) is a plain dict lookup with zero
# lock contention on label creation.
for _stage in STAGES:
    stage_histogram.labels(_stage)
for _outcome in OUTCOMES:
    request_counter.labels(_outcome)
for _cache in CACHE_KINDS:
    cache_hit_counter.labels(_cache)

_stage_samples: dict[str, deque[float]] = {stage: deque(maxlen=_RING_SIZE) for stage in STAGES}


def record_stage_ms(stage: str, ms: float) -> None:
    """Record one stage latency in milliseconds.

    Called from the search pipeline per query; must stay O(1) and never raise
    so metrics can be layered onto the request path without risk. Unknown
    stage names are tolerated (a new label child is created lazily).
    """
    try:
        if stage not in _stage_samples:
            _stage_samples[stage] = deque(maxlen=_RING_SIZE)
        _stage_samples[stage].append(float(ms))
        stage_histogram.labels(stage).observe(ms / 1000.0)
    except Exception:
        pass


def record_request_outcome(outcome: str) -> None:
    """Count one finished search request (``outcome``: success|error). Never raises."""
    try:
        request_counter.labels(outcome).inc()
    except Exception:
        pass


def record_result_count(count: int) -> None:
    """Add the number of results a search returned to the summary counter. Never raises."""
    try:
        result_counter.inc(max(int(count), 0))
    except Exception:
        pass


def record_cache_hit(cache: str) -> None:
    """Count one cache event (``cache``: redis|memory|semantic|none). Never raises."""
    try:
        cache_hit_counter.labels(cache).inc()
    except Exception:
        pass


@router.get("/metrics")
async def prometheus_metrics() -> Response:
    """Render all KB metrics in Prometheus text exposition format 0.0.4.

    Prometheus already scrapes this path on :8000; the content type must be
    the exact exposition header or the scrape target errors out.
    """
    return Response(content=generate_latest(registry), media_type=PROMETHEUS_CONTENT_TYPE)


@router.get("/metrics/healthz")
async def metrics_healthz() -> JSONResponse:
    """Cheap in-process latency health: per-stage p50 over the last 100 samples.

    Computed with ``statistics.median`` over the module-local rings, so no
    Prometheus query language is needed for a quick human check.
    """
    p50s: dict[str, float] = {}
    for stage, samples in _stage_samples.items():
        if samples:
            p50s[stage] = round(statistics.median(samples), 3)
    return JSONResponse({"stage_p50_ms": p50s, "ring_size": _RING_SIZE})
