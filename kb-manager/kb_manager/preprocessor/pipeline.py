"""Full preprocessing pipeline: clean → normalise → extract keywords."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from kb_manager.preprocessor.clean import clean_text
from kb_manager.preprocessor.persian import PersianPreprocessor

# ---------------------------------------------------------------------------
# Persian stop-words (very small set – enough for keyword extraction)
# ---------------------------------------------------------------------------

_PERSIAN_STOP_WORDS: frozenset[str] = frozenset(
    {
        "و",
        "در",
        "به",
        "از",
        "که",
        "این",
        "را",
        "با",
        "است",
        "برای",
        "آن",
        "یک",
        "خود",
        "تا",
        "کرد",
        "بر",
        "هم",
        "نیز",
        "گفت",
        "می",
        "شد",
        "باید",
        "دو",
        "یا",
        "هر",
        "او",
        "ما",
        "شما",
        "آنها",
        "من",
        "بی",
        "پس",
        "چه",
        "اگر",
        "اما",
        "یعنی",
        "شود",
        "شده",
        "بود",
        "شدند",
        "دارد",
        "می‌شود",
        "می‌کند",
        "داشت",
        "ندارد",
        "شوند",
        "بودن",
        "کردن",
        "شدن",
        "داشتن",
        "باشد",
        "باشند",
        "نیست",
        "هست",
        "هستند",
        "نیستند",
        "نمی",
        "بیشتر",
        "کمتر",
        "خیلی",
        "البته",
        "یکی",
        "دیگر",
        "همه",
        "چند",
        "چون",
        "لی",
        "هنوز",
        "هیچ",
        "بین",
        "زیر",
        "روی",
        "قبل",
        "بعد",
        "همین",
        "همان",
        "آنان",
        "اینک",
        "پیش",
        "فقط",
        "بدون",
        "علاوه",
        "طی",
        "ضمن",
        "نوع",
        " manner",
        "مورد",
        "اول",
        "دوم",
        "سوم",
    }
)

# English stop-words (common)
_ENGLISH_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the",
        "is",
        "at",
        "which",
        "on",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "with",
        "to",
        "for",
        "of",
        "not",
        "no",
        "can",
        "had",
        "has",
        "have",
        "it",
        "its",
        "be",
        "as",
        "are",
        "was",
        "were",
        "been",
        "that",
        "this",
        "these",
        "those",
        "from",
        "by",
        "if",
        "than",
        "then",
        "so",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "need",
        "also",
        "very",
        "just",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "same",
        "other",
        "being",
        "there",
        "here",
        "where",
        "when",
        "how",
        "what",
        "who",
        "whom",
        "why",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "some",
        "any",
        "such",
        "only",
        "own",
        "too",
    }
)

_STOP_WORDS = _PERSIAN_STOP_WORDS | _ENGLISH_STOP_WORDS

# ---------------------------------------------------------------------------
# Keyword extraction helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[\w\u0600-\u06ff\u0750-\u077f]+", re.UNICODE)
_TECH_TOKEN = re.compile(r"^[A-Za-z0-9_\-\.]{3,}$")


def _tokenise(text: str) -> list[str]:
    """Split *text* into word tokens (Persian + English)."""
    return _WORD_RE.findall(text.lower())


def _extract_keywords(text: str, top_n: int = 15) -> list[str]:
    """Extract the *top_n* most frequent non-stop-word tokens.

    Tokens that look like technical codes (alphanumeric + hyphens) are
    always kept regardless of frequency.
    """
    tokens = _tokenise(text)
    freq: dict[str, int] = {}
    for tok in tokens:
        if tok in _STOP_WORDS or len(tok) < 2:
            continue
        freq[tok] = freq.get(tok, 0) + 1

    # Technical tokens get a big bonus so they surface even with low counts
    scored: list[tuple[str, float]] = []
    for tok, count in freq.items():
        bonus = 5.0 if _TECH_TOKEN.match(tok) else 0.0
        scored.append((tok, count + bonus))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [tok for tok, _ in scored[:top_n]]


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------


def _encoding_quality(text: str) -> float:
    """Heuristic: how much of *text* is valid / well-encoded.

    Returns a value in [0, 1] where 1 means no encoding artefacts were
    found.
    """
    if not text:
        return 0.0
    bad = 0
    for ch in text:
        cp = ord(ch)
        # Replacement char, surrogates, U+FFFD, control chars (except \n \t)
        if cp == 0xFFFD or 0xD800 <= cp <= 0xDFFF or (cp < 0x20 and ch not in "\n\t\r"):
            bad += 1
    return 1.0 - (bad / len(text))


def _persian_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are in the Persian/Arabic range."""
    alpha = [ch for ch in text if ch.isalpha()]
    if not alpha:
        return 0.0
    persian = sum(1 for ch in alpha if "\u0600" <= ch <= "\u06ff" or "\u0750" <= ch <= "\u077f")
    return persian / len(alpha)


def _quality_score(text: str, cleaned: str) -> float:
    """Combined quality score in [0, 1].

    Components:
    - encoding quality (40 %)
    - Persian ratio (30 %)
    - non-empty check (20 %)
    - length sanity (10 %)
    """
    enc = _encoding_quality(cleaned)
    pers = _persian_ratio(cleaned)
    non_empty = 1.0 if len(cleaned.strip()) > 0 else 0.0
    length = min(len(cleaned) / 100.0, 1.0)  # saturate at 100 chars

    return round(0.4 * enc + 0.3 * pers + 0.2 * non_empty + 0.1 * length, 4)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PreprocessingResult:
    """Output of :func:`preprocess`."""

    original_text: str
    cleaned_text: str
    normalised_text: str
    keywords: list[str] = field(default_factory=list)
    sentence_count: int = 0
    quality_score: float = 0.0

    def __repr__(self) -> str:
        kw_preview = ", ".join(self.keywords[:5])
        return (
            f"PreprocessingResult("
            f"sentences={self.sentence_count}, "
            f"keywords=[{kw_preview}, ...], "
            f"quality={self.quality_score:.2%})"
        )


# ---------------------------------------------------------------------------
# Sentence tokeniser (simple regex – works without Hazm)
# ---------------------------------------------------------------------------

_SENT_RE = re.compile(r"[.!؟]+\s+|\Z")


def _count_sentences(text: str) -> int:
    """Count sentences using a simple heuristic."""
    matches = _SENT_RE.findall(text)
    return max(len(matches), 1) if text.strip() else 0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class PreprocessingPipeline:
    """Chain: clean → Persian normalise → keyword extraction.

    Parameters
    ----------
    persian_preprocessor:
        Optional custom :class:`PersianPreprocessor`.  A default instance
        is created when *None*.
    spell_check:
        Forwarded to :class:`PersianPreprocessor` when no custom one is
        provided.
    max_keywords:
        Maximum number of keywords to extract.
    """

    def __init__(
        self,
        persian_preprocessor: PersianPreprocessor | None = None,
        *,
        spell_check: bool = True,
        max_keywords: int = 15,
    ) -> None:
        self._pp = persian_preprocessor or PersianPreprocessor(spell_check=spell_check)
        self._max_keywords = max_keywords

    def run(self, text: str) -> PreprocessingResult:
        """Execute the pipeline on *text* and return structured results."""
        original = text

        # Stage 1 – generic cleaning
        cleaned = clean_text(text)

        # Stage 2 – Persian normalisation
        normalised = self._pp.normalise(cleaned)

        # Stage 3 – keyword extraction
        keywords = _extract_keywords(normalised, top_n=self._max_keywords)

        # Metadata
        sentence_count = _count_sentences(normalised)
        quality = _quality_score(original, cleaned)

        return PreprocessingResult(
            original_text=original,
            cleaned_text=cleaned,
            normalised_text=normalised,
            keywords=keywords,
            sentence_count=sentence_count,
            quality_score=quality,
        )

    def __call__(self, text: str) -> PreprocessingResult:
        return self.run(text)
