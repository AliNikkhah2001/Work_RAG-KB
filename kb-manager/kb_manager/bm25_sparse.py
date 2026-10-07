"""
bm25_sparse.py - Pure-Python Persian text normalizer, tokenizer, and BM25 sparse vector generator.

Converts raw Persian text into Qdrant SparseVector representations (indices: List[int], values: List[float]).
Uses deterministic 31-bit MD5 hashing to ensure chunk insertion and query sparse vectors map
to identical integer indices across processes without external dependencies.
"""

import hashlib
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple, Union

from qdrant_client import models
from qdrant_client.http.models import SparseVector

# Persian character normalization map
_PERSIAN_CHAR_MAP: Dict[str, str] = {
    # Arabic Yeh variants -> Persian Yeh
    "\u064a": "\u06cc",  # ي -> ی
    "\u0649": "\u06cc",  # ى -> ی
    "\u0626": "\u06cc",  # ئ -> ی
    "\u06d2": "\u06cc",  # ے -> ی
    # Arabic Kaf variants -> Persian Kaf
    "\u0643": "\u06a9",  # ك -> ک
    "\u06aa": "\u06a9",  # ڪ -> ک
    # Teh Marbuta / Heh variants -> Persian Heh
    "\u0629": "\u0647",  # ة -> ه
    "\u06c1": "\u0647",  # ہ -> ه
    # Alef variants -> Bare Alef
    "\u0622": "\u0627",  # آ -> ا
    "\u0623": "\u0627",  # أ -> ا
    "\u0625": "\u0627",  # إ -> ا
    "\u0671": "\u0627",  # ٱ -> ا
    # Waw with Hamza -> Waw
    "\u0624": "\u0648",  # ؤ -> و
    # Tatweel / Kashida -> strip
    "\u0640": "",
    # Zero-width / non-breaking spaces -> standard space
    "\u200c": " ",  # ZWNJ
    "\u200d": " ",  # ZWJ
    "\u200b": " ",  # Zero-width space
    "\ufeff": " ",  # BOM / ZWNBSP
    "\u00a0": " ",  # Non-breaking space
}

# Add Persian digits (۰-۹: \u06f0-\u06f9) -> standard digits
for _i in range(10):
    _PERSIAN_CHAR_MAP[chr(0x06F0 + _i)] = str(_i)
    # Arabic-Indic digits (٠-٩: \u0660-\u0669) -> standard digits
    _PERSIAN_CHAR_MAP[chr(0x0660 + _i)] = str(_i)

_TRANS_TABLE = str.maketrans(_PERSIAN_CHAR_MAP)
_DIACRITICS_REGEX = re.compile(r"[\u064b-\u065f\u0670]")
_PUNCTUATION_REGEX = re.compile(r"[^\w\s]")
_WHITESPACE_REGEX = re.compile(r"\s+")

# 31-bit Mersenne prime for safe uint32 Qdrant index bounds: 0 <= idx < 2**31 - 1
HASH_MODULO: int = 2147483647


def normalize_persian(text: str) -> str:
    """
    Pure-Python Persian text normalizer.
    Normalizes character variants, removes diacritics, maps digits,
    and strips punctuation while converting ZWNJ into whitespace.
    """
    if not text or not isinstance(text, str):
        return ""
    # Map character variants and digits
    norm = text.translate(_TRANS_TABLE)
    # Remove Arabic / Persian diacritics (Fatha, Damma, Kasra, Tanwin, Sukun, Shadda, etc.)
    norm = _DIACRITICS_REGEX.sub("", norm)
    # Replace punctuation and symbols with space
    norm = _PUNCTUATION_REGEX.sub(" ", norm)
    # Treat underscore as word separator
    norm = norm.replace("_", " ")
    # Lowercase Latin characters
    norm = norm.lower()
    # Normalize multiple whitespaces
    norm = _WHITESPACE_REGEX.sub(" ", norm).strip()
    return norm


def tokenize_persian(text: str) -> List[str]:
    """
    Tokenize normalized Persian text into word tokens.
    """
    normalized = normalize_persian(text)
    if not normalized:
        return []
    return [token for token in normalized.split() if token]


def token_to_index(token: str, modulo: int = HASH_MODULO) -> int:
    """
    Deterministically hash a token into a non-negative integer (0 <= index < modulo).
    Uses MD5 to guarantee consistent cross-process, cross-platform hash values.
    """
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big") % modulo


class PersianBM25Encoder:
    """
    BM25 Sparse Vector Encoder for Persian text.

    Calculates document term weights and query IDF weights, formatting them
    as integer indices and float weights (SparseVector) for Qdrant.
    """

    def __init__(
        self,
        corpus: Optional[List[str]] = None,
        avgdl: float = 100.0,
        k1: float = 1.5,
        b: float = 0.75,
        default_idf: float = 1.0,
        vocab: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize the Persian BM25 encoder.

        :param corpus: Optional initial document collection to fit IDF and avgdl statistics.
        :param avgdl: Average document length fallback (default 100.0).
        :param k1: BM25 term frequency saturation parameter (default 1.5).
        :param b: BM25 document length penalty parameter (default 0.75).
        :param default_idf: Default IDF weight when corpus is empty or unfitted (default 1.0).
        :param vocab: Optional explicit token-to-index mapping dictionary.
        """
        self.avgdl = float(avgdl)
        self.k1 = float(k1)
        self.b = float(b)
        self.default_idf = float(default_idf)
        self.vocab: Dict[str, int] = dict(vocab) if vocab else {}
        self.doc_count: int = 0
        self.doc_freqs: Dict[str, int] = Counter()

        if corpus:
            self.fit(corpus)

    def normalize(self, text: str) -> str:
        """Normalize Persian text."""
        return normalize_persian(text)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize Persian text into normalized tokens."""
        return tokenize_persian(text)

    def get_token_index(self, token: str) -> int:
        """
        Map a token to a deterministic integer index.
        Checks explicit vocabulary if provided, otherwise hashes deterministically.
        """
        if self.vocab and token in self.vocab:
            return self.vocab[token]
        return token_to_index(token)

    def fit(self, documents: List[str]) -> "PersianBM25Encoder":
        """
        Fit BM25 statistics (doc_count, doc_freqs, avgdl) from a collection of documents.
        """
        self.doc_count = len(documents)
        self.doc_freqs = Counter()
        total_tokens = 0

        for doc in documents:
            tokens = self.tokenize(doc)
            total_tokens += len(tokens)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1

        if self.doc_count > 0:
            self.avgdl = max(1.0, total_tokens / self.doc_count)

        return self

    def add_documents(self, documents: List[str]) -> None:
        """
        Incrementally add documents to update BM25 corpus statistics.
        """
        total_tokens = self.avgdl * self.doc_count
        new_count = len(documents)

        for doc in documents:
            tokens = self.tokenize(doc)
            total_tokens += len(tokens)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.doc_count += new_count
        if self.doc_count > 0:
            self.avgdl = max(1.0, total_tokens / self.doc_count)

    def _calc_idf(self, token: str) -> float:
        """
        Calculate Robertson-Spärck Jones IDF for a token.
        Always non-negative using ln(1 + (N - df + 0.5) / (df + 0.5)).
        """
        if self.doc_count == 0:
            return self.default_idf

        df = self.doc_freqs.get(token, 0)
        numerator = self.doc_count - df + 0.5
        denominator = df + 0.5
        return math.log(1.0 + (numerator / denominator))

    def encode_document(self, text: str) -> Tuple[List[int], List[float]]:
        """
        Encode a document chunk into BM25 sparse representation.

        Formula for term weight:
            v_D(t) = (TF(t, D) * (k1 + 1)) / (TF(t, D) + k1 * (1 - b + b * (|D| / avgdl)))

        Returns:
            Tuple of (indices: List[int], values: List[float])
            Indices are non-negative, unique, and strictly sorted in ascending order.
        """
        tokens = self.tokenize(text)
        if not tokens:
            return ([], [])

        doc_len = len(tokens)
        tf_counts = Counter(tokens)
        index_weights: Dict[int, float] = {}

        # BM25 document length normalization denominator term
        len_norm = 1.0 - self.b + self.b * (doc_len / self.avgdl)

        for token, tf in tf_counts.items():
            weight = (tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm)
            idx = self.get_token_index(token)
            # Accumulate in case of rare hash collision
            index_weights[idx] = index_weights.get(idx, 0.0) + weight

        # Sort by index ascending (required by Qdrant SparseVector)
        sorted_items = sorted(index_weights.items(), key=lambda item: item[0])
        indices = [idx for idx, _ in sorted_items]
        values = [round(float(val), 6) for _, val in sorted_items]

        return (indices, values)

    def encode_query(self, query: str) -> Tuple[List[int], List[float]]:
        """
        Encode a query into BM25 sparse representation.

        Formula for query weight:
            v_Q(t) = IDF(t) * TF(t, Q)

        Returns:
            Tuple of (indices: List[int], values: List[float])
            Indices are non-negative, unique, and strictly sorted in ascending order.
        """
        tokens = self.tokenize(query)
        if not tokens:
            return ([], [])

        q_tf_counts = Counter(tokens)
        index_weights: Dict[int, float] = {}

        for token, tf in q_tf_counts.items():
            idf = self._calc_idf(token)
            weight = idf * tf
            idx = self.get_token_index(token)
            # Accumulate in case of rare hash collision
            index_weights[idx] = index_weights.get(idx, 0.0) + weight

        # Sort by index ascending (required by Qdrant SparseVector)
        sorted_items = sorted(index_weights.items(), key=lambda item: item[0])
        indices = [idx for idx, _ in sorted_items]
        values = [round(float(val), 6) for _, val in sorted_items]

        return (indices, values)

    @staticmethod
    def build_sparse_vector(indices: List[int], values: List[float]) -> SparseVector:
        """
        Construct a Qdrant SparseVector model from indices and values.
        """
        return SparseVector(indices=indices, values=values)

    def encode_documents_batch(
        self, texts: List[str]
    ) -> List[Tuple[List[int], List[float]]]:
        """
        Encode a batch of document texts into BM25 sparse representations.
        """
        return [self.encode_document(text) for text in texts]
