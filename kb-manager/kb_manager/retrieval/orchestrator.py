"""Unified retrieval orchestrator: BM25 + Dense + HyDE + Multi-Query + RRF + Rerank."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from kb_manager.config import config, PROJECT_ROOT
from kb_manager.dense import DenseSemanticIndex, load_or_build
from kb_manager.reranker import CrossEncoderReranker, get_reranker
from kb_manager.retrieval.lexicon import (
    BM25,
    PERSIAN_TRANSLATE_TABLE as _PERSIAN_TRANSLATE_TABLE,
    STOPWORDS as _STOPWORDS,
    tokenize as _tokenize,
)
from kb_manager.query_reform import (
    HyDEGenerator,
    MultiQueryGenerator,
    reciprocal_rank_fusion,
    weighted_rrf,
    detect_query_type,
)

import re
from pydantic import BaseModel
from sqlalchemy import select
from kb_manager.models.database import Chunk, Document

logger = logging.getLogger(__name__)

_DENSE_CACHE_PATH = PROJECT_ROOT / "data" / "dense_embeddings.npz"
_DENSE_MODEL = config.embedding.model_name
_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
_RERANKER_TOP_K = 50


# ---------------------------------------------------------------------------
# Result models (shared with web layer)
# ---------------------------------------------------------------------------

class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    heading_path: str
    content_preview: str
    bm25_score: float = 0.0
    semantic_score: float = 0.0
    dense_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0
    ordinal: int = 0


class SearchSteps(BaseModel):
    query: str
    normalized_query: str
    tokens: list[str]
    total_chunks_indexed: int
    detected_query_type: str = "general"
    strategy_used: str = "auto"
    hyde_document: str = ""
    sub_queries: list[dict] = []
    timing_breakdown: dict[str, float] = {}
    bm25_results: list[SearchResult]
    dense_results: list[SearchResult]
    merged_candidates: list[SearchResult]
    final_results: list[SearchResult]
    elapsed_ms: float


@dataclass
class RetrievalConfig:
    """Runtime retrieval configuration."""
    strategy: str = "auto"              # auto|bm25|dense|hybrid|hyde|multi_query|full
    top_k: int = 5
    bm25_weight: float = 0.3
    dense_weight: float = 0.7
    rrf_k: int = 60
    bm25_candidates: int = 50
    dense_candidates: int = 50
    rerank_candidates: int = 20
    rerank_enabled: bool = True
    hyde_enabled: bool = True
    multi_query_enabled: bool = True
    adaptive_enabled: bool = True
    filters: dict = field(default_factory=dict)


@dataclass
class RetrievalTiming:
    """Timing breakdown for each stage."""
    total_ms: float = 0.0
    preprocess_ms: float = 0.0
    bm25_ms: float = 0.0
    dense_ms: float = 0.0
    hyde_ms: float = 0.0
    multi_query_ms: float = 0.0
    rrf_ms: float = 0.0
    rerank_ms: float = 0.0


class HybridRetriever:
    """
    Production hybrid retriever combining multiple strategies.
    
    Pipeline:
    1. Query preprocessing (Persian normalization)
    2. Query type detection (adaptive weights)
    3. Optional HyDE (hypothetical document embedding)
    4. Optional Multi-Query rewriting (6 variants)
    5. BM25 lexical search
    6. Dense semantic search
    7. RRF fusion
    8. Cross-encoder reranking
    9. Return top-k with full breakdown
    """

    def __init__(self):
        self._index_built = False
        self._dense_built = False
        self._chunk_data = []
        self._bm25 = None
        self._dense = None
        self._reranker = None
        self._hyde_gen = None
        self._multi_query_gen = None
        self._llm_client = None

    async def _ensure_lexical(self):
        """Build BM25 index + load chunk data (cheap, no ML deps)."""
        if self._index_built:
            return

        # Lazy import to avoid circular dependency with web.app
        from kb_manager.web.app import db

        # Load chunks from DB
        async with db.session() as session:
            result = await session.execute(select(Chunk))
            all_chunks = result.scalars().all()

            doc_ids = list({c.document_id for c in all_chunks})
            docs_result = await session.execute(
                select(Document).where(Document.id.in_(doc_ids))
            )
            doc_map = {d.id: d for d in docs_result.scalars().all()}

        # Prepare data for BM25 and Dense
        docs_for_bm25 = []
        chunk_data = []
        dense_titles = []
        dense_headings = []
        dense_chunk_types = []

        for c in all_chunks:
            doc = doc_map.get(c.document_id)
            title = doc.title if doc else "Unknown"
            docs_for_bm25.append((c.id, c.content))
            chunk_data.append((c.id, c.document_id, title, c.heading_path, c.content, c.chunk_type))
            dense_titles.append(title)
            dense_headings.append(c.heading_path)
            dense_chunk_types.append(c.chunk_type)

        # Build BM25
        self._bm25 = BM25()
        self._bm25.index(docs_for_bm25)

        # Stash dense metadata for lazy build
        self._dense_meta = (dense_titles, dense_headings, dense_chunk_types)
        self._chunk_data = chunk_data
        self._index_built = True

    async def _ensure_dense(self):
        """Build/load the dense semantic index lazily (requires sentence-transformers)."""
        await self._ensure_lexical()
        if self._dense_built:
            return

        dense_titles, dense_headings, dense_chunk_types = self._dense_meta
        dense_texts = [cd[4] for cd in self._chunk_data]
        dense_ids = [cd[0] for cd in self._chunk_data]
        self._dense = load_or_build(
            _DENSE_CACHE_PATH,
            dense_ids,
            dense_texts,
            titles=dense_titles,
            headings=dense_headings,
            chunk_types=dense_chunk_types,
            model_name=_DENSE_MODEL,
        )
        self._dense_built = True
        self._index_built = True

    def _get_llm(self):
        """Get LLM client from config (lazy)."""
        if self._llm_client is None:
            from kb_manager.llm import create_llm_client_from_config
            self._llm_client = create_llm_client_from_config(config)
        return self._llm_client

    def _preprocess_query(self, query: str) -> str:
        """Apply Persian normalization to query."""
        return query.strip().translate(_PERSIAN_TRANSLATE_TABLE)

    def _get_adaptive_weights(self, query_type: str) -> tuple[float, float, bool, bool, bool]:
        """Get adaptive weights based on query type."""
        adaptive = config.retrieval.adaptive_weights.get(query_type, {})
        return (
            adaptive.get("bm25_weight", config.retrieval.bm25_weight),
            adaptive.get("dense_weight", config.retrieval.dense_weight),
            adaptive.get("hyde_enabled", config.retrieval.hyde_enabled),
            adaptive.get("multi_query_enabled", config.retrieval.multi_query_enabled),
            adaptive.get("rerank_enabled", config.reranker.enabled),
        )

    async def search(
        self,
        query: str,
        strategy: str = "auto",
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> SearchSteps:
        """Main search entry point with full step breakdown."""
        start_total = time.monotonic()
        await self._ensure_lexical()

        timing = RetrievalTiming()
        filters = filters or {}
        original_query = query

        # Step 1: Preprocess
        t0 = time.monotonic()
        query = self._preprocess_query(query)
        timing.preprocess_ms = (time.monotonic() - t0) * 1000

        # Step 2: Query type detection (for adaptive weights)
        query_type = "general"
        if config.retrieval.adaptive_enabled:
            query_type = detect_query_type(query)

        # Get adaptive weights
        bm25_w, dense_w, hyde_on, mq_on, rerank_on = self._get_adaptive_weights(query_type)

        # Override with strategy preset
        strat = config.retrieval.strategies.get(strategy, {})
        if not strat.get("adaptive"):
            bm25_w = strat.get("bm25_weight", bm25_w)
            dense_w = strat.get("dense_weight", dense_w)
            hyde_on = strat.get("hyde_enabled", hyde_on)
            mq_on = strat.get("multi_query_enabled", mq_on)
            rerank_on = strat.get("rerank_enabled", rerank_on)

        # Dense-dependent legs require the dense index
        needs_dense = dense_w > 0 or hyde_on or mq_on
        if needs_dense:
            await self._ensure_dense()

        # Lazy init of query reform components (only when used)
        if hyde_on and self._hyde_gen is None:
            from kb_manager.query_reform import HyDEGenerator
            self._hyde_gen = HyDEGenerator(
                llm_client=self._get_llm(),
                prompt_template=config.hyde.prompt_template,
                max_length=config.hyde.max_length,
            )
        if mq_on and self._multi_query_gen is None:
            from kb_manager.query_reform import MultiQueryGenerator
            self._multi_query_gen = MultiQueryGenerator(
                llm_client=self._get_llm(),
                prompt_template=config.multi_query.prompt_template,
                num_queries=config.multi_query.num_queries,
                beam_size=config.multi_query.beam_size,
            )

        # Step 3: HyDE (optional)
        hyde_doc = ""
        hyde_results = []
        if hyde_on:
            t0 = time.monotonic()
            hyde_doc_obj = self._hyde_gen.generate(query)
            hyde_doc = hyde_doc_obj.content
            if hyde_doc:
                hyde_results = self._hyde_gen.embed_and_search(
                    query, self._dense, top_k=config.retrieval.rerank_candidates
                )
            timing.hyde_ms = (time.monotonic() - t0) * 1000

        # Step 4: Multi-Query (optional)
        multi_queries = []
        mq_results = []
        if mq_on:
            t0 = time.monotonic()
            multi_queries = self._multi_query_gen.generate(query)
            # Search each variant
            for mq in multi_queries:
                variant_results = self._dense.search(mq.query, top_k=config.retrieval.dense_candidates)
                mq_results.append((mq.query, mq.query_type, variant_results))
            timing.multi_query_ms = (time.monotonic() - t0) * 1000

        # Step 5: BM25 Search
        t0 = time.monotonic()
        bm25_raw = self._bm25.search(query, top_k=config.retrieval.bm25_candidates)
        timing.bm25_ms = (time.monotonic() - t0) * 1000

        # Step 6: Dense Search
        dense_raw = []
        if needs_dense:
            t0 = time.monotonic()
            dense_raw = self._dense.search(query, top_k=config.retrieval.dense_candidates)
            timing.dense_ms = (time.monotonic() - t0) * 1000

        # Step 7: RRF Fusion
        t0 = time.monotonic()
        # Build ranked lists
        ranked_lists = []

        # BM25 leg (skip when pure-dense strategy)
        if bm25_w > 0:
            ranked_lists.append([(cid, score) for cid, score in bm25_raw if score > 0])

        # Dense leg
        if needs_dense and dense_w > 0:
            ranked_lists.append(dense_raw)

        # HyDE leg (if enabled)
        if hyde_on and hyde_results:
            ranked_lists.append(hyde_results)

        # Multi-Query legs (if enabled)
        if mq_on:
            for mq_query, mq_type, variant_results in mq_results:
                ranked_lists.append(variant_results)

        # Apply adaptive weights for weighted RRF
        if config.multi_query.fusion_method == "weighted_rrf":
            weights = config.multi_query.weights
            fused = weighted_rrf(ranked_lists, weights, k=config.retrieval.rrf_k, top_k=config.retrieval.rerank_candidates)
        else:
            fused = reciprocal_rank_fusion(ranked_lists, k=config.retrieval.rrf_k, top_k=config.retrieval.rerank_candidates)

        timing.rrf_ms = (time.monotonic() - t0) * 1000

        # Step 8: Cross-encoder Reranking
        candidates = []
        bm25_id_map = {cd[0]: cd for cd in self._chunk_data}
        bm25_scores = dict(bm25_raw)
        dense_scores = dict(dense_raw)

        for chunk_id, fused_score in fused:
            cd = bm25_id_map.get(chunk_id)
            if cd is None:
                continue
            candidates.append(SearchResult(
                chunk_id=chunk_id,
                doc_id=cd[1],
                doc_title=cd[2],
                heading_path=cd[3],
                content_preview=cd[4][:300],
                bm25_score=round(bm25_scores.get(chunk_id, 0), 4),
                dense_score=round(dense_scores.get(chunk_id, 0), 4),
                hybrid_score=round(fused_score, 6),
                ordinal=cd[5] if isinstance(cd[5], int) else 0,
            ))

        if rerank_on and len(candidates) > 5:
            t0 = time.monotonic()
            if self._reranker is None:
                self._reranker = get_reranker(model_name=_RERANKER_MODEL)
            reranked_dicts = self._reranker.rerank(
                query,
                [c.model_dump() for c in candidates],
                top_k=max(top_k, config.retrieval.rerank_candidates),
                score_key="hybrid_score",
            )
            candidates = [SearchResult(**d) for d in reranked_dicts]
            timing.rerank_ms = (time.monotonic() - t0) * 1000

        # Step 9: Final top-k
        final = candidates[:top_k]
        timing.total_ms = (time.monotonic() - start_total) * 1000

        return SearchSteps(
            query=original_query,
            normalized_query=query,
            tokens=_tokenize(query),
            total_chunks_indexed=len(self._chunk_data),
            detected_query_type=query_type,
            strategy_used=strategy,
            hyde_document=hyde_doc,
            sub_queries=[
                {"query": mq.query, "type": mq.query_type} for mq in multi_queries
            ],
            timing_breakdown={
                "total": round(timing.total_ms, 1),
                "preprocess_ms": round(timing.preprocess_ms, 1),
                "bm25_ms": round(timing.bm25_ms, 1),
                "dense_ms": round(timing.dense_ms, 1),
                "hyde_ms": round(timing.hyde_ms, 1),
                "multi_query_ms": round(timing.multi_query_ms, 1),
                "rrf_ms": round(timing.rrf_ms, 1),
                "rerank_ms": round(timing.rerank_ms, 1),
            },
            bm25_results=[
                c if isinstance(c, SearchResult) else SearchResult(**c)
                for c in sorted(candidates, key=lambda x: x.bm25_score, reverse=True)[:top_k]
            ],
            dense_results=[
                c if isinstance(c, SearchResult) else SearchResult(**c)
                for c in sorted(candidates, key=lambda x: x.dense_score, reverse=True)[:top_k]
            ],
            merged_candidates=candidates[:top_k],
            final_results=[c if isinstance(c, SearchResult) else SearchResult(**c) for c in final],
            elapsed_ms=round(timing.total_ms, 1),
        )


# Global instance (singleton pattern for caching)
_retriever_instance: Optional[HybridRetriever] = None


async def get_retriever() -> HybridRetriever:
    """Get or create the global retriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance


def invalidate_retriever():
    """Invalidate the cached retriever (call after ingestion)."""
    global _retriever_instance
    _retriever_instance = None