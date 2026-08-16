"""Search / query testing routes with step-by-step transparency."""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from typing import TYPE_CHECKING

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel
from sqlalchemy import select

if TYPE_CHECKING:
    pass

router = APIRouter()


# ---------------------------------------------------------------------------
# Simple Persian-aware tokenizer
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "از در به و با برای که این آن را شد است هستند بودند می باشد می شود "
    "می گردد می کند هر دو آیا یا اگر ولی تا باشد بر اساس طبق طریق "
    "نیز همچنین نیز درباره بین توسط مانند مثل طی خود کنید گردد "
    "باید یک یکی شود گردد را ندارد نمی کنند می شوند می باشند "
    "the a an is are was were be been am does do did have has had "
    "in on at to for of and or but not no so if it its this that "
    "can will would should could may might shall".split()
)


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words, removing stopwords."""
    text = text.lower()
    tokens = re.findall(r"[a-zA-Z\u0600-\u06FF\u0750-\u077F\u200C\u200D\d]+", text)
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# BM25 implementation
# ---------------------------------------------------------------------------

class BM25:
    """Okapi BM25 ranking."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_dl = 0.0
        self.doc_freqs: dict[str, int] = {}
        self.doc_lens: list[int] = []
        self.doc_ids: list[str] = []
        self.tokens_per_doc: list[list[str]] = []

    def index(self, documents: list[tuple[str, str]]) -> None:
        """Index (doc_id, content) pairs."""
        self.doc_ids = []
        self.tokens_per_doc = []
        self.doc_lens = []
        all_df: dict[str, int] = {}

        for doc_id, content in documents:
            tokens = _tokenize(content)
            self.doc_ids.append(doc_id)
            self.tokens_per_doc.append(tokens)
            self.doc_lens.append(len(tokens))
            seen: set[str] = set()
            for t in tokens:
                if t not in seen:
                    all_df[t] = all_df.get(t, 0) + 1
                    seen.add(t)

        self.doc_count = len(self.doc_ids)
        self.avg_dl = sum(self.doc_lens) / max(self.doc_count, 1)
        self.doc_freqs = all_df

    def _idf(self, term: str) -> float:
        df = self.doc_freqs.get(term, 0)
        return math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        tokens = self.tokens_per_doc[doc_idx]
        dl = self.doc_lens[doc_idx]
        tf_map: dict[str, int] = Counter(tokens)
        score = 0.0
        for qt in query_tokens:
            tf = tf_map.get(qt, 0)
            idf = self._idf(qt)
            num = tf * (self.k1 + 1)
            den = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avg_dl, 1))
            score += idf * num / den
        return score

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        q_tokens = _tokenize(query)
        if not q_tokens or self.doc_count == 0:
            return []
        scored = [(self.doc_ids[i], self.score(q_tokens, i)) for i in range(self.doc_count)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Simple cosine similarity (TF-IDF)
# ---------------------------------------------------------------------------

def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf_map: Counter = Counter(tokens)
    total = max(len(tokens), 1)
    return {t: (c / total) * idf.get(t, 1.0) for t, c in tf_map.items()}


def _cosine_sim(v1: dict[str, float], v2: dict[str, float]) -> float:
    if not v1 or not v2:
        return 0.0
    common = set(v1) & set(v2)
    dot = sum(v1[k] * v2[k] for k in common)
    mag1 = math.sqrt(sum(v ** 2 for v in v1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in v2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


# ---------------------------------------------------------------------------
# Search API
# ---------------------------------------------------------------------------

class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    heading_path: str
    content_preview: str
    bm25_score: float = 0.0
    semantic_score: float = 0.0
    hybrid_score: float = 0.0
    ordinal: int = 0


class SearchSteps(BaseModel):
    query: str
    normalized_query: str
    tokens: list[str]
    total_chunks_indexed: int
    bm25_results: list[SearchResult]
    semantic_results: list[SearchResult]
    merged_candidates: list[SearchResult]
    final_results: list[SearchResult]
    elapsed_ms: float


async def _build_index() -> tuple[list[tuple[str, str, str, str, str, str]], BM25]:
    """Load all chunks + docs and build BM25 index."""
    from kb_manager.models.database import Chunk, Document
    from kb_manager.web.app import db

    async with db.session() as session:
        result = await session.execute(select(Chunk))
        all_chunks = result.scalars().all()

        doc_ids = list({c.document_id for c in all_chunks})
        docs_result = await session.execute(
            select(Document).where(Document.id.in_(doc_ids))
        )
        doc_map = {d.id: d for d in docs_result.scalars().all()}

    docs_for_bm25 = []
    chunk_data = []
    for c in all_chunks:
        doc = doc_map.get(c.document_id)
        title = doc.title if doc else "Unknown"
        docs_for_bm25.append((c.id, c.content))
        chunk_data.append((c.id, c.document_id, title, c.heading_path, c.content, c.chunk_type))

    bm25 = BM25()
    bm25.index(docs_for_bm25)
    return chunk_data, bm25


async def _compute_idf(chunk_data: list) -> dict[str, float]:
    """Compute IDF across all chunks."""
    doc_count = len(chunk_data)
    df: dict[str, int] = {}
    for _, _, _, _, content, _ in chunk_data:
        tokens = set(_tokenize(content))
        for t in tokens:
            df[t] = df.get(t, 0) + 1
    return {t: math.log((doc_count - c + 0.5) / (c + 0.5) + 1.0) for t, c in df.items()}


async def search_knowledge_base(query: str, top_k: int = 10) -> SearchSteps:
    """Run full search pipeline with step tracking."""
    start = time.monotonic()
    normalized = query.strip()
    tokens = _tokenize(normalized)

    chunk_data, bm25 = await _build_index()

    # --- Step 1: BM25 ---
    bm25_raw = bm25.search(normalized, top_k=top_k * 3)
    bm25_id_map = {cd[0]: cd for cd in chunk_data}
    bm25_results = []
    bm25_scores = {}
    for chunk_id, score in bm25_raw:
        if score <= 0:
            continue
        cd = bm25_id_map.get(chunk_id)
        if cd is None:
            continue
        bm25_scores[chunk_id] = score
        bm25_results.append(SearchResult(
            chunk_id=chunk_id,
            doc_id=cd[1],
            doc_title=cd[2],
            heading_path=cd[3],
            content_preview=cd[4][:300],
            bm25_score=round(score, 4),
            ordinal=cd[5] if isinstance(cd[5], int) else 0,
        ))

    # --- Step 2: Semantic (TF-IDF cosine) ---
    idf = await _compute_idf(chunk_data)
    query_vec = _tfidf_vector(tokens, idf)

    semantic_scores_list: list[tuple[str, float]] = []
    for cd in chunk_data:
        chunk_tokens = _tokenize(cd[4])
        chunk_vec = _tfidf_vector(chunk_tokens, idf)
        sim = _cosine_sim(query_vec, chunk_vec)
        if sim > 0:
            semantic_scores_list.append((cd[0], sim))

    semantic_scores_list.sort(key=lambda x: x[1], reverse=True)
    semantic_scores_map = {cid: s for cid, s in semantic_scores_list[:top_k * 3]}
    semantic_results = []
    for chunk_id, score in semantic_scores_list[:top_k * 3]:
        cd = bm25_id_map.get(chunk_id)
        if cd is None:
            continue
        semantic_results.append(SearchResult(
            chunk_id=chunk_id,
            doc_id=cd[1],
            doc_title=cd[2],
            heading_path=cd[3],
            content_preview=cd[4][:300],
            semantic_score=round(score, 4),
            ordinal=cd[5] if isinstance(cd[5], int) else 0,
        ))

    # --- Step 3: Merge (RRF - Reciprocal Rank Fusion) ---
    all_chunk_ids = set(bm25_scores.keys()) | set(sem_scores for sem_scores, _ in semantic_scores_list[:top_k * 3])
    merged: dict[str, SearchResult] = {}
    k = 60  # RRF constant

    # BM25 ranks
    for rank, (chunk_id, _) in enumerate(bm25_raw):
        if chunk_id not in bm25_id_map:
            continue
        cd = bm25_id_map[chunk_id]
        rrf_score = 1.0 / (k + rank + 1)
        if chunk_id not in merged:
            merged[chunk_id] = SearchResult(
                chunk_id=chunk_id,
                doc_id=cd[1],
                doc_title=cd[2],
                heading_path=cd[3],
                content_preview=cd[4][:300],
                bm25_score=round(bm25_scores.get(chunk_id, 0), 4),
                semantic_score=round(semantic_scores_map.get(chunk_id, 0), 4),
                hybrid_score=round(rrf_score, 6),
                ordinal=cd[5] if isinstance(cd[5], int) else 0,
            )
        else:
            merged[chunk_id].hybrid_score += rrf_score
            merged[chunk_id].bm25_score = round(bm25_scores.get(chunk_id, 0), 4)

    for rank, (chunk_id, _) in enumerate(semantic_scores_list):
        if chunk_id not in bm25_id_map:
            continue
        cd = bm25_id_map[chunk_id]
        rrf_score = 1.0 / (k + rank + 1)
        if chunk_id not in merged:
            merged[chunk_id] = SearchResult(
                chunk_id=chunk_id,
                doc_id=cd[1],
                doc_title=cd[2],
                heading_path=cd[3],
                content_preview=cd[4][:300],
                bm25_score=round(bm25_scores.get(chunk_id, 0), 4),
                semantic_score=round(semantic_scores_map.get(chunk_id, 0), 4),
                hybrid_score=round(rrf_score, 6),
                ordinal=cd[5] if isinstance(cd[5], int) else 0,
            )
        else:
            merged[chunk_id].hybrid_score += rrf_score
            merged[chunk_id].semantic_score = round(semantic_scores_map.get(chunk_id, 0), 4)

    candidates = sorted(merged.values(), key=lambda x: x.hybrid_score, reverse=True)
    for i, c in enumerate(candidates):
        c.hybrid_score = round(c.hybrid_score, 6)

    # --- Step 4: Final top-k ---
    final = candidates[:top_k]
    elapsed = (time.monotonic() - start) * 1000

    return SearchSteps(
        query=query,
        normalized_query=normalized,
        tokens=tokens,
        total_chunks_indexed=len(chunk_data),
        bm25_results=bm25_results[:top_k],
        semantic_results=semantic_results[:top_k],
        merged_candidates=candidates[:top_k],
        final_results=final,
        elapsed_ms=round(elapsed, 1),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def search_knowledge_base_sync(query: str, top_k: int = 10) -> SearchSteps:
    """Synchronous wrapper for search_knowledge_base."""
    import asyncio
    new_loop = asyncio.new_event_loop()
    try:
        return new_loop.run_until_complete(search_knowledge_base(query, top_k))
    finally:
        new_loop.close()

@router.get("")
async def search_page(request: Request):
    """Search/test query page."""
    from kb_manager.web.app import templates

    return templates.TemplateResponse(request, "search.html", {})


@router.post("/api")
async def search_api(request: Request):
    """Search API endpoint - returns JSON with step-by-step results."""
    import asyncio
    import traceback

    try:
        body = await request.json()
    except Exception:
        return {"error": "Invalid JSON body"}

    query = body.get("query", "").strip() if isinstance(body, dict) else ""
    top_k = min(body.get("top_k", 10), 50) if isinstance(body, dict) else 10

    if not query:
        return {"error": "Empty query"}

    try:
        steps = await asyncio.to_thread(search_knowledge_base_sync, query, top_k)
        return steps.model_dump()
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}
