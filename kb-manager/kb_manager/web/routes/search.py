"""Search / query testing routes with step-by-step transparency.

This module now uses the unified HybridRetriever orchestrator which combines:
- BM25 lexical search
- Dense semantic search (with contextual embeddings)
- HyDE (Hypothetical Document Embeddings)
- Multi-Query rewriting (6 variants)
- RRF / Weighted RRF fusion
- Cross-encoder reranking
"""

from __future__ import annotations

import asyncio
import traceback
from typing import TYPE_CHECKING

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

from kb_manager.config import config
from kb_manager.retrieval.orchestrator import (
    HybridRetriever,
    get_retriever,
    invalidate_retriever,
)
from kb_manager.web.app import templates

if TYPE_CHECKING:
    pass

router = APIRouter()


class SearchRequest(BaseModel):
    """Search request payload."""
    query: str
    top_k: int = 10
    strategy: str = "auto"  # auto|bm25|dense|hybrid|hyde|multi_query|full
    filters: dict = {}


class SearchResponse(BaseModel):
    """Search response with full step breakdown."""
    query: str
    normalized_query: str
    tokens: list[str]
    total_chunks_indexed: int
    detected_query_type: str
    strategy_used: str
    hyde_document: str = ""
    sub_queries: list[dict] = []
    bm25_results: list[dict]
    dense_results: list[dict]
    merged_candidates: list[dict]
    final_results: list[dict]
    elapsed_ms: float
    timing_breakdown: dict[str, float] = {}


@router.get("")
async def search_page(request: Request):
    """Search/test query page."""
    strategies = list(config.retrieval.strategies.keys())
    return templates.TemplateResponse(
        request,
        "search.html",
        {"strategies": strategies},
    )


@router.post("/api")
async def search_api(request: Request):
    """Advanced search API endpoint with strategy selection."""
    try:
        body = await request.json()
    except Exception:
        return {"error": "Invalid JSON body"}

    if not isinstance(body, dict):
        return {"error": "Request body must be JSON object"}

    query = body.get("query", "").strip()
    top_k = min(body.get("top_k", 10), 50)
    strategy = body.get("strategy", "auto")
    filters = body.get("filters", {})

    # Validate strategy
    valid_strategies = list(config.retrieval.strategies.keys()) + ["auto"]
    if strategy not in valid_strategies:
        return {"error": f"Invalid strategy. Choose from: {valid_strategies}"}

    if not query:
        return {"error": "Empty query"}

    try:
        retriever = await get_retriever()
        steps = await retriever.search(
            query=query,
            strategy=strategy,
            top_k=top_k,
            filters=filters,
        )

        # Build response with strategy info
        response = {
            "query": steps.query,
            "normalized_query": steps.normalized_query,
            "tokens": steps.tokens,
            "total_chunks_indexed": steps.total_chunks_indexed,
            "detected_query_type": "general",  # TODO: expose from retriever
            "strategy_used": strategy,
            "hyde_document": "",  # TODO: expose from retriever
            "sub_queries": [],
            "bm25_results": [r.model_dump() for r in steps.bm25_results],
            "dense_results": [r.model_dump() for r in steps.dense_results],
            "merged_candidates": [r.model_dump() for r in steps.merged_candidates],
            "final_results": [r.model_dump() for r in steps.final_results],
            "elapsed_ms": steps.elapsed_ms,
            "timing_breakdown": {
                "total": steps.elapsed_ms,
            },
        }
        return response
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@router.post("/api/simple")
async def search_simple(request: Request):
    """Simple search endpoint for backward compatibility."""
    try:
        body = await request.json()
    except Exception:
        return {"error": "Invalid JSON body"}

    query = body.get("query", "").strip() if isinstance(body, dict) else ""
    top_k = min(body.get("top_k", 10), 50) if isinstance(body, dict) else 10

    if not query:
        return {"error": "Empty query"}

    try:
        retriever = await get_retriever()
        steps = await retriever.search(query=query, top_k=top_k)
        return {"results": [r.model_dump() for r in steps.final_results], "elapsed_ms": steps.elapsed_ms}
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@router.post("/invalidate-cache")
async def invalidate_cache():
    """Invalidate the search index cache (call after ingestion)."""
    invalidate_retriever()
    return {"status": "cache invalidated"}


@router.get("/strategies")
async def list_strategies():
    """List available search strategies."""
    return {
        "strategies": list(config.retrieval.strategies.keys()),
        "default": "auto",
        "adaptive_enabled": config.retrieval.adaptive_enabled,
    }


@router.get("/query-types")
async def list_query_types():
    """List supported query types for adaptive weighting."""
    return {
        "types": list(config.retrieval.adaptive_weights.keys()),
        "weights": config.retrieval.adaptive_weights,
    }