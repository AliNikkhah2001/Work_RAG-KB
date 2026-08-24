"""Shared lexical retrieval primitives: Persian tokenizer, normalization, BM25.

This module is intentionally free of web/orchestrator imports so both
``kb_manager.web.routes.search`` and ``kb_manager.retrieval.orchestrator``
can use it without circular imports.
"""

from __future__ import annotations

import math
import re
from collections import Counter

# Persian character normalization map
PERSIAN_CHAR_MAP = {
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
    # ZWNJ handling - replace with space for tokenization
    "\u200c": " ",
    # Arabic-Indic digits -> ASCII
    "\u0660": "0", "\u0661": "1", "\u0662": "2", "\u0663": "3", "\u0664": "4",
    "\u0665": "5", "\u0666": "6", "\u0667": "7", "\u0668": "8", "\u0669": "9",
    # Extended Arabic-Indic digits
    "\u06f0": "0", "\u06f1": "1", "\u06f2": "2", "\u06f3": "3", "\u06f4": "4",
    "\u06f5": "5", "\u06f6": "6", "\u06f7": "7", "\u06f8": "8", "\u06f9": "9",
}

PERSIAN_TRANSLATE_TABLE = str.maketrans(PERSIAN_CHAR_MAP)

STOPWORDS = frozenset(
    "از در به و با برای که این آن را شد است هستند بودند می باشد می شود "
    "می گردد می کند هر دو آیا یا اگر ولی تا باشد بر اساس طبق طریق "
    "نیز همچنین نیز درباره بین توسط مانند مثل طی خود کنید گردد "
    "باید یک یکی شود گردد را ندارد نمی کنند می شوند می باشند "
    "the a an is are was were be been am does do did have has had "
    "in on at to for of and or but not no so if it its this that "
    "can will would should could may might shall".split()
)


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words + Persian char 3-grams, removing stopwords."""
    text = text.lower().translate(PERSIAN_TRANSLATE_TABLE)

    word_tokens = re.findall(r"[a-zA-Z\u0600-\u06FF\u0750-\u077F\u200C\u200D\d]+", text)
    word_tokens = [t for t in word_tokens if t not in STOPWORDS and len(t) > 1]

    char_ngrams = []
    for token in word_tokens:
        if any("\u0600" <= ch <= "\u06FF" for ch in token):
            for i in range(len(token) - 2):
                char_ngrams.append(token[i : i + 3])

    return word_tokens + char_ngrams


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
            tokens = tokenize(content)
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
        q_tokens = tokenize(query)
        if not q_tokens or self.doc_count == 0:
            return []
        scored = [(self.doc_ids[i], self.score(q_tokens, i)) for i in range(self.doc_count)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
