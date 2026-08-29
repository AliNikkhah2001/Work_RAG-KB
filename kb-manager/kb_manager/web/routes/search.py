"""Search / query testing routes with step-by-step transparency."""

from __future__ import annotations

import asyncio
import math
import os
import re
import time
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel
from sqlalchemy import select

from kb_manager.config import PROJECT_ROOT
from kb_manager.dense import DenseSemanticIndex, load_or_build
from kb_manager.hyde import HyDEGenerator
from kb_manager.reranker import CrossEncoderReranker, get_reranker

if TYPE_CHECKING:
    pass

router = APIRouter()

_DENSE_CACHE_PATH = PROJECT_ROOT / "data" / "dense_embeddings.npz"
_DENSE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
_RERANKER_TOP_K = 50  # Number of candidates to rerank

# HyDE configuration (disabled by default; set KB_HYDE_ENABLED=true to enable)
_HYDE_ENABLED = os.getenv("KB_HYDE_ENABLED", "false").lower() == "true"
_HYDE_LLM_MODEL = os.getenv("KB_HYDE_LLM", "gpt-4o-mini")
_HYDE_LLM_API_KEY = os.getenv("KB_HYDE_API_KEY", os.getenv("OPENAI_API_KEY", ""))
_HYDE_LLM_BASE_URL = os.getenv("KB_HYDE_BASE_URL", os.getenv("OPENAI_BASE_URL", ""))

# Persian character normalization map
_PERSIAN_CHAR_MAP = {
    # Arabic yeh -> Persian yeh
    "\u064a": "\u06cc",  # ي -> ی
    "\u0649": "\u06cc",  # ى -> ی
    # Arabic kaf -> Persian kaf
    "\u0643": "\u06a9",  # ك -> ک
    # Arabic teh marbuta -> Persian heh
    "\u0629": "\u0647",  # ة -> ه
    # Alef variants -> basic alef
    "\u0671": "\u0627",  # ٱ -> ا
    "\u0623": "\u0627",  # أ -> ا
    "\u0625": "\u0627",  # إ -> ا
    "\u0622": "\u0622",  # آ -> آ (keep)
    # ZWNJ handling - replace with space for tokenization
    "\u200c": " ",  # ZWNJ -> space
    # Arabic-Indic digits -> ASCII
    "\u0660": "0", "\u0661": "1", "\u0662": "2", "\u0663": "3", "\u0664": "4",
    "\u0665": "5", "\u0666": "6", "\u0667": "7", "\u0668": "8", "\u0669": "9",
    # Extended Arabic-Indic digits
    "\u06f0": "0", "\u06f1": "1", "\u06f2": "2", "\u06f3": "3", "\u06f4": "4",
    "\u06f5": "5", "\u06f6": "6", "\u06f7": "7", "\u06f8": "8", "\u06f9": "9",
}

_PERSIAN_TRANSLATE_TABLE = str.maketrans(_PERSIAN_CHAR_MAP)


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
    """Tokenize text into lowercase words + Persian char 3-grams, removing stopwords."""
    # Apply Persian character normalization
    text = text.lower().translate(_PERSIAN_TRANSLATE_TABLE)

    # Extract word tokens (Persian, Arabic, Latin, digits)
    word_tokens = re.findall(r"[a-zA-Z\u0600-\u06FF\u0750-\u077F\u200C\u200D\d]+", text)
    word_tokens = [t for t in word_tokens if t not in _STOPWORDS and len(t) > 1]

    # Generate Persian character 3-grams for typo/orthographic robustness
    # Only for tokens containing Persian/Arabic characters
    char_ngrams = []
    for token in word_tokens:
        if any('\u0600' <= ch <= '\u06FF' for ch in token):
            # Generate 3-grams
            for i in range(len(token) - 2):
                char_ngrams.append(token[i:i+3])

    # Combine word tokens + char n-grams
    return word_tokens + char_ngrams


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
    dense_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0
    ordinal: int = 0


class SearchSteps(BaseModel):
    query: str
    normalized_query: str
    tokens: list[str]
    total_chunks_indexed: int
    bm25_results: list[SearchResult]
    semantic_results: list[SearchResult]
    dense_results: list[SearchResult]
    merged_candidates: list[SearchResult]
    final_results: list[SearchResult]
    elapsed_ms: float


async def _build_index() -> tuple[list[tuple[str, str, str, str, str, str, int]], BM25, DenseSemanticIndex, CrossEncoderReranker, HyDEGenerator | None]:
    """Load all chunks + docs and build BM25 + dense indexes + reranker + HyDE."""
    from kb_manager.models.database import Chunk, Document
    from kb_manager.web.deps import db

    async with db.session() as session:
        result = await session.execute(select(Chunk))
        all_chunks = result.scalars().all()

        doc_ids = list({c.document_id for c in all_chunks})
        docs_result = await session.execute(
            select(Document).where(Document.id.in_(doc_ids))
        )
        doc_map = {d.id: d for d in docs_result.scalars().all()}

    docs_for_bm25 = []
    chunk_data: list[tuple[str, str, str, str, str, str, int]] = []
    dense_titles = []
    dense_headings = []
    dense_chunk_types = []
    for c in all_chunks:
        doc = doc_map.get(c.document_id)
        title = doc.title if doc else "Unknown"
        # Boost keywords in BM25 by repeating them in the content
        # Keywords get 3x weight by being added 3 times
        keyword_text = " ".join(c.keywords) if c.keywords else ""
        boosted_content = c.content + " " + keyword_text + " " + keyword_text + " " + keyword_text
        docs_for_bm25.append((c.id, boosted_content))
        chunk_data.append((c.id, c.document_id, title, c.heading_path, c.content, c.chunk_type, c.ordinal))
        dense_titles.append(title)
        dense_headings.append(c.heading_path)
        dense_chunk_types.append(c.chunk_type)

    bm25 = BM25()
    bm25.index(docs_for_bm25)

    dense_texts = [cd[4] for cd in chunk_data]
    dense_ids = [cd[0] for cd in chunk_data]
    dense = load_or_build(
        _DENSE_CACHE_PATH,
        dense_ids,
        dense_texts,
        titles=dense_titles,
        headings=dense_headings,
        chunk_types=dense_chunk_types,
        model_name=_DENSE_MODEL,
    )

    reranker = get_reranker(model_name=_RERANKER_MODEL)

    # HyDE: optional LLM-based hypothetical document generation
    hyde = None
    if _HYDE_ENABLED and _HYDE_LLM_API_KEY:
        hyde = HyDEGenerator(
            llm_model=_HYDE_LLM_MODEL,
            llm_api_key=_HYDE_LLM_API_KEY,
            llm_base_url=_HYDE_LLM_BASE_URL,
            embedding_model=_DENSE_MODEL,
        )

    return chunk_data, bm25, dense, reranker, hyde


_index_cache: tuple[list[tuple[str, str, str, str, str, str, int]], BM25, DenseSemanticIndex, CrossEncoderReranker, HyDEGenerator | None] | None = None
_index_cache_count: int = 0
_index_cache_fp: str | None = None
_index_lock = asyncio.Lock()


def _invalidate_index_cache() -> None:
    """Drop the cached index so the next search rebuilds it."""
    global _index_cache, _index_cache_count, _index_cache_fp
    _index_cache = None
    _index_cache_count = 0
    _index_cache_fp = None


async def _get_index() -> tuple[list[tuple[str, str, str, str, str, str, int]], BM25, DenseSemanticIndex, CrossEncoderReranker, HyDEGenerator | None]:
    """Return the cached index, building it on first use (thread-safe, F5 fix)."""
    global _index_cache, _index_cache_count, _index_cache_fp

    from sqlalchemy import func, select

    from kb_manager.models.database import Chunk, Document
    from kb_manager.web.deps import db
    from kb_manager.dense import DenseSemanticIndex

    async with db.session() as session:
        chunk_count = (await session.execute(select(func.count(Chunk.id)))).scalar_one()

    # Fast path: count mismatch → rebuild; count match but need fingerprint check for same-count content change (F5)
    if _index_cache is not None and _index_cache_count == chunk_count and _index_cache_fp is not None:
        # Compute current fingerprint via lightweight DB scan to detect stale cache
        async with db.session() as session:
            result = await session.execute(select(Chunk))
            all_chunks = result.scalars().all()
            if len(all_chunks) == chunk_count:
                doc_ids = list({c.document_id for c in all_chunks})
                docs_result = await session.execute(select(Document).where(Document.id.in_(doc_ids)))
                doc_map = {d.id: d for d in docs_result.scalars().all()}
                texts = [c.content for c in all_chunks]
                titles = [doc_map.get(c.document_id).title if doc_map.get(c.document_id) else "" for c in all_chunks]
                headings = [c.heading_path for c in all_chunks]
                ctypes = [c.chunk_type for c in all_chunks]
                cur_fp = DenseSemanticIndex.fingerprint(texts, titles, headings, ctypes, _DENSE_MODEL, True)
                if cur_fp == _index_cache_fp:
                    return _index_cache
                # fingerprint mismatch → stale, fall through to rebuild
            # else count actually changed → rebuild

    async with _index_lock:
        # Double-check after acquiring lock
        if _index_cache is not None and _index_cache_count == chunk_count and _index_cache_fp is not None:
            # Re-check fingerprint under lock to avoid race
            async with db.session() as session:
                result = await session.execute(select(Chunk))
                all_chunks = result.scalars().all()
                if len(all_chunks) == chunk_count:
                    doc_ids = list({c.document_id for c in all_chunks})
                    docs_result = await session.execute(select(Document).where(Document.id.in_(doc_ids)))
                    doc_map = {d.id: d for d in docs_result.scalars().all()}
                    texts = [c.content for c in all_chunks]
                    titles = [doc_map.get(c.document_id).title if doc_map.get(c.document_id) else "" for c in all_chunks]
                    headings = [c.heading_path for c in all_chunks]
                    ctypes = [c.chunk_type for c in all_chunks]
                    cur_fp = DenseSemanticIndex.fingerprint(texts, titles, headings, ctypes, _DENSE_MODEL, True)
                    if cur_fp == _index_cache_fp:
                        return _index_cache
        chunk_data, bm25, dense, reranker, hyde = await _build_index()
        # Compute and store fingerprint for future same-count checks
        texts = [cd[4] for cd in chunk_data]
        titles = [cd[2] for cd in chunk_data]
        headings = [cd[3] for cd in chunk_data]
        ctypes = [cd[5] for cd in chunk_data]
        cur_fp = DenseSemanticIndex.fingerprint(texts, titles, headings, ctypes, _DENSE_MODEL, True)
        _index_cache = (chunk_data, bm25, dense, reranker, hyde)
        _index_cache_count = chunk_count
        _index_cache_fp = cur_fp
        return _index_cache


def _compute_idf(chunk_data: list) -> dict[str, float]:
    """Compute IDF across all chunks."""
    doc_count = len(chunk_data)
    df: dict[str, int] = {}
    for _, _, _, _, content, _, _ in chunk_data:
        tokens = set(_tokenize(content))
        for t in tokens:
            df[t] = df.get(t, 0) + 1
    return {t: math.log((doc_count - c + 0.5) / (c + 0.5) + 1.0) for t, c in df.items()}


async def search_knowledge_base(query: str, top_k: int = 10) -> SearchSteps:
    """Run full search pipeline with step tracking.

    Pipeline: BM25 + Dense + [HyDE] → RRF → Cross-encoder reranker
    HyDE is optional: only runs when KB_HYDE_ENABLED=true and API key is set.
    """
    start = time.monotonic()
    normalized = query.strip()
    tokens = _tokenize(normalized)

    chunk_data, bm25, dense, reranker, hyde = await _get_index()

    # --- Step 2: BM25 ---
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
            ordinal=cd[6] if isinstance(cd[6], int) else 0,
        ))

    # --- Step 3: Dense semantic (embedding cosine) ---
    dense_raw = dense.search(normalized, top_k=top_k * 3)
    dense_scores = dict(dense_raw)
    dense_results = []
    for chunk_id, score in dense_raw:
        cd = bm25_id_map.get(chunk_id)
        if cd is None:
            continue
        dense_results.append(SearchResult(
            chunk_id=chunk_id,
            doc_id=cd[1],
            doc_title=cd[2],
            heading_path=cd[3],
            content_preview=cd[4][:300],
            dense_score=round(score, 4),
            semantic_score=round(score, 4),  # repurpose semantic_score for dense
            ordinal=cd[6] if isinstance(cd[6], int) else 0,
        ))

    # --- Step 3b: HyDE (optional) ---
    hyde_scores: dict[str, float] = {}
    if hyde is not None and hyde.is_available:
        hyde_vec = hyde.get_hyde_vector(normalized)
        if hyde_vec is not None and dense._matrix is not None:
            # Cosine similarity: dense matrix is already L2-normalized
            sims = dense._matrix @ hyde_vec
            hyde_order = np.argsort(-sims)[:top_k * 3]
            for idx in hyde_order:
                chunk_id = dense._ids[idx]
                if chunk_id in bm25_id_map:
                    hyde_scores[chunk_id] = float(sims[idx])

    # --- Step 4: Merge (RRF - Reciprocal Rank Fusion over 2-3 legs) ---
    ranked_lists: list[list[tuple[str, float]]] = [
        [(cid, s) for cid, s in bm25_raw if cid in bm25_id_map],
        list(dense_raw),
    ]
    # Add HyDE leg if available
    if hyde_scores:
        ranked_lists.append(list(hyde_scores.items()))

    merged: dict[str, SearchResult] = {}
    k = 60  # RRF constant
    for ranked in ranked_lists:
        for rank, (chunk_id, _) in enumerate(ranked):
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
                    semantic_score=round(dense_scores.get(chunk_id, 0), 4),
                    dense_score=round(dense_scores.get(chunk_id, 0), 4),
                    hybrid_score=round(rrf_score, 6),
                    ordinal=cd[6] if isinstance(cd[6], int) else 0,
                )
            else:
                merged[chunk_id].hybrid_score += rrf_score
                merged[chunk_id].bm25_score = round(bm25_scores.get(chunk_id, 0), 4)
                merged[chunk_id].semantic_score = round(dense_scores.get(chunk_id, 0), 4)
                merged[chunk_id].dense_score = round(dense_scores.get(chunk_id, 0), 4)

    candidates = sorted(merged.values(), key=lambda x: x.hybrid_score, reverse=True)
    for i, c in enumerate(candidates):
        c.hybrid_score = round(c.hybrid_score, 6)

    # --- Step 6: Cross-encoder Reranking ---
    rerank_input = candidates[:_RERANKER_TOP_K]
    reranked = reranker.rerank(normalized, [c.model_dump() for c in rerank_input], top_k=top_k)
    
    # Convert reranked dicts back to SearchResult objects
    reranked_results = [SearchResult(**r) for r in reranked]

    # --- Step 7: Final top-k ---
    final = reranked_results[:top_k]
    elapsed = (time.monotonic() - start) * 1000

    return SearchSteps(
        query=query,
        normalized_query=normalized,
        tokens=tokens,
        total_chunks_indexed=len(chunk_data),
        bm25_results=bm25_results[:top_k],
        semantic_results=dense_results[:top_k],
        dense_results=dense_results[:top_k],
        merged_candidates=candidates[:top_k],
        final_results=final,
        elapsed_ms=round(elapsed, 1),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _run_sync(coro):
    """Run coroutine from sync context without breaking a running loop (F2/F29 fix)."""
    import concurrent.futures

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(asyncio.run, coro)
        return fut.result()


def search_knowledge_base_sync(query: str, top_k: int = 10) -> SearchSteps:
    """Synchronous wrapper — safe from both sync and async callers."""
    return _run_sync(search_knowledge_base(query, top_k))

@router.get("")
async def search_page(request: Request):
    """Search/test query page."""
    from kb_manager.web.deps import templates

    return templates.TemplateResponse(request, "search.html", {})


@router.post("/api")
async def search_api(request: Request):
    """Search API endpoint - returns JSON with step-by-step results."""
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
        # F2/F29 fix: directly await async pipeline — no to_thread/_sync_loop indirection
        steps = await search_knowledge_base(query, top_k)
        return steps.model_dump()
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}
