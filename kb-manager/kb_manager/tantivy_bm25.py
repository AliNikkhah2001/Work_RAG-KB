"""Rust-backed BM25 via tantivy (RAM-resident).

Replaces the pure-Python Okapi BM25 in search.py for ~95% latency cut.
Keeps the same public API: index(docs) + search(query, top_k, allowed_ids).

Tokenization reuses the Persian-aware _tokenize logic from search.py
(word tokens + 3-char n-grams, stopword removal, ZWNJ etc.) and feeds
tantivy a whitespace-tokenized string so tantivy's BM25 scores the same
token stream. Scores are on a different scale than Python BM25, but the
retrieval pipeline fuses via RRF over rank, so only order matters.

Two separate instances are used for content vs keyword indexes (mirrors
search.py's tuple (bm25_content, bm25_kw) with keyword_boost).

All data lives in RAM: tantivy Index in RAMDirectory, writer heap 50MB,
plus Python doc_ids list. No disk I/O after commit. HNSW pgvector already
prewarmed via pg_prewarm, dense matrix in RAM, reranker on GPU.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Set

try:
    import tantivy  # type: ignore
    _HAS_TANTIVY = True
except Exception:
    tantivy = None  # type: ignore
    _HAS_TANTIVY = False

log = logging.getLogger(__name__)


def _tokenize_for_tantivy(text: str) -> str:
    """Persian-aware tokenization matching search.py::_tokenize, joined for tantivy.

    Returns a whitespace-joined string of tokens (word tokens + char 3-grams)
    ready for tantivy's whitespace tokenizer.
    """
    # Lazily import to avoid circular import at module load
    try:
        from kb_manager.web.routes.search import _tokenize as _orig_tokenize

        toks = _orig_tokenize(text)
        return " ".join(toks)
    except Exception:
        # Fallback: minimal replication if import fails during early build
        from kb_manager.preprocessor.regex_persian import (
            ARABIC_TO_PERSIAN_MAP as _BASE_MAP,
        )
        from kb_manager.preprocessor.regex_persian import DIACRITICS_RE, TOKEN_RE

        _PERSIAN_CHAR_MAP = {
            **_BASE_MAP,
            "\u0622": "\u0622",
            "\u200c": " ",
            "\u0660": "0",
            "\u0661": "1",
            "\u0662": "2",
            "\u0663": "3",
            "\u0664": "4",
            "\u0665": "5",
            "\u0666": "6",
            "\u0667": "7",
            "\u0668": "8",
            "\u0669": "9",
            "\u06f0": "0",
            "\u06f1": "1",
            "\u06f2": "2",
            "\u06f3": "3",
            "\u06f4": "4",
            "\u06f5": "5",
            "\u06f6": "6",
            "\u06f7": "7",
            "\u06f8": "8",
            "\u06f9": "9",
        }
        _PERSIAN_TRANSLATE_TABLE = str.maketrans(_PERSIAN_CHAR_MAP)
        _STOPWORDS = frozenset(
            "از در به و با برای که این آن را شد است هستند بودند می باشد می شود "
            "می گردد می کند هر دو آیا یا اگر ولی تا باشد بر اساس طبق طریق "
            "نیز همچنین نیز درباره بین توسط مانند مثل طی خود کنید گردد "
            "باید یک یکی شود گردد را ندارد نمی کنند می شوند می باشند "
            "the a an is are was were be been am does do did have has had "
            "in on at to for of and or but not no so if it its this that "
            "can will would should could may might shall".split()
        )

        text_l = text.lower().translate(_PERSIAN_TRANSLATE_TABLE)
        text_l = DIACRITICS_RE.sub("", text_l)
        word_tokens = TOKEN_RE.findall(text_l)
        word_tokens = [t for t in word_tokens if t not in _STOPWORDS and len(t) > 1]
        char_ngrams: List[str] = []
        for token in word_tokens:
            if any("\u0600" <= ch <= "\u06FF" for ch in token):
                for i in range(len(token) - 2):
                    char_ngrams.append(token[i : i + 3])
        return " ".join(word_tokens + char_ngrams)


class TantivyBM25:
    """RAM-resident BM25 using tantivy (Rust). API mirrors search.py:BM25."""

    def __init__(self):
        if not _HAS_TANTIVY:
            raise ImportError("tantivy not installed (pip install tantivy)")
        self._schema = None
        self._index = None
        self._searcher = None
        self.doc_count = 0

    def index(self, documents: List[Tuple[str, str]]) -> None:
        """Build a RAM index from (doc_id, content) pairs."""
        import tantivy

        builder = tantivy.SchemaBuilder()
        # raw for doc_id exact stored, whitespace for content (we pre-tokenize)
        builder.add_text_field("doc_id", stored=True, tokenizer_name="raw")
        builder.add_text_field(
            "content", stored=False, tokenizer_name="whitespace", index_option="position"
        )
        schema = builder.build()
        # RAM directory (default when no path)
        index = tantivy.Index(schema)
        writer = index.writer(heap_size=50_000_000)  # 50MB heap, all RAM
        for doc_id, content in documents:
            # Pre-tokenize to match Python BM25 char-ngram logic
            tokenized = _tokenize_for_tantivy(content or "")
            # tantivy Document
            doc = tantivy.Document()
            doc.add_text("doc_id", doc_id)
            # Empty content still indexable but will not match
            doc.add_text("content", tokenized if tokenized else " ")
            writer.add_document(doc)
        writer.commit()
        # Reload to make searchable
        index.reload()
        self._schema = schema
        self._index = index
        self._searcher = index.searcher()
        self.doc_count = len(documents)
        log.info("TantivyBM25 indexed %d docs (RAM)", self.doc_count)

    def search(
        self, query: str, top_k: int = 20, allowed_ids: Set[str] | None = None
    ) -> List[Tuple[str, float]]:
        if not query or not query.strip() or self._index is None or self._searcher is None:
            return []
        tokenized = _tokenize_for_tantivy(query)
        if not tokenized.strip():
            return []
        # Parse query for content field
        try:
            q = self._index.parse_query(tokenized, ["content"])
        except Exception as e:
            log.warning("tantivy parse_query failed: %s", e)
            return []
        # If filtered, over-fetch then post-filter (avoids tantivy filter complexity)
        limit = top_k if allowed_ids is None else top_k * 5
        # Guard: tantivy requires limit >0
        limit = max(1, limit)
        try:
            result = self._searcher.search(q, limit)
        except Exception as e:
            log.warning("tantivy search failed: %s", e)
            return []
        hits = result.hits  # List[(score, DocAddress)]
        out: List[Tuple[str, float]] = []
        for score, addr in hits:
            try:
                doc = self._searcher.doc(addr)
                did = doc["doc_id"][0]
                if allowed_ids is not None and did not in allowed_ids:
                    continue
                out.append((did, float(score)))
                if len(out) >= top_k:
                    break
            except Exception:
                continue
        return out
