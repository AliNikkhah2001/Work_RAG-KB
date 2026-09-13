"""Persian-specific text normalization for knowledge base documents."""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Optional heavy dependencies – degrade gracefully if missing
# ---------------------------------------------------------------------------

_HAZM_AVAILABLE = False
try:
    from hazm import Normalizer as HazmNormalizer  # type: ignore[import-untyped]

    _HAZM_AVAILABLE = True
except ImportError:
    pass

_SHEKAR_AVAILABLE = False
try:
    from shekar import Normalizer as ShekarNormalizer  # type: ignore[import-untyped]

    _SHEKAR_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Unicode mapping tables  (Arabic → Persian)
# ---------------------------------------------------------------------------

_ARABIC_TO_PERSIAN: dict[str, str] = {
    "\u0622": "\u0627",  # آ → ا
    "\u0623": "\u0627",  # أ → ا
    "\u0625": "\u0627",  # إ → ا
    "\u0629": "\u0647",  # ة → ه
    "\u0649": "\u06cc",  # ى → ی
    "\u0671": "\u0627",  # ٱ → ا (Alef Wasla)
}

# Characters that look the same in some fonts but have different code-points
# NOTE: canonical source is regex_persian.ARABIC_TO_PERSIAN_MAP; kept here for backward compat
_PERSIAN_CHARS_ONLY: dict[str, str] = {
    "\u0643": "\u06a9",  # ك → ک
    "\u0621": "\u0627",  # ء → ا
    "\u0626": "\u06cc",  # ئ → ی
}


def _apply_unicode_map(text: str, table: dict[str, str]) -> str:
    """Apply a character replacement table to *text*."""
    for src, dst in table.items():
        text = text.replace(src, dst)
    return text


# ---------------------------------------------------------------------------
# ZWNJ helpers
# ---------------------------------------------------------------------------

_ZWNJ = "\u200c"

# Persian suffixes that should be preceded by ZWNJ
# NOTE: single-letter suffixes (ی, م, ت, ش) are intentionally excluded —
# they are too ambiguous and corrupt stems like "نام"→"نا‌م", "رتبه"→"ر‌تبه",
# "کمیل"→"کم‌یل". Only multi-char suffixes are safe for auto ZWNJ.
_SUFFIXES_WITH_ZWNJ: tuple[str, ...] = (
    "ها",
    "های",
    "ترین",
    "تر",
    "گر",
    "گری",
    "ای",
    "هایی",
)

# Patterns where ZWNJ should be *removed* (e.g. inside English words)
_ZWNJ_IN_ENGLISH = re.compile(r"([a-zA-Z0-9])\u200c([a-zA-Z0-9])")

# Persian letter range (used for word-boundary checks)
_PERSIAN_LETTER = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")


def _fix_zwnj(text: str) -> str:
    """Insert/remove ZWNJ in common Persian patterns.

    Inserts ZWNJ *before* a suffix only when suffix is at word end
    (not followed by another Persian letter) and preceded by a Persian
    letter.  This avoids corrupting middle of words like "رتبه" where
    "ت" is part of the stem, not a suffix.
    """
    # Remove ZWNJ that leaked into ASCII runs
    text = _ZWNJ_IN_ENGLISH.sub(r"\1\2", text)

    # NOTE 2026-09-03: auto ZWNJ insertion before suffixes (ها/تر/...) is disabled
    # by default — it corrupted stems like "دکتر"→"دک‌تر", "هیئت"→"هییت",
    # "نام"→"نا‌م". Enable only with explicit allowlist and word-end check.
    # for suffix in _SUFFIXES_WITH_ZWNJ:
    #     pattern = re.compile(
    #         f"([\u0600-\u06FF\u0750-\u077F])({re.escape(suffix)})(?![\u0600-\u06FF\u0750-\u077F\u200c])",
    #     )
    #     text = pattern.sub(rf"\1{_ZWNJ}\2", text)

    # Collapse double ZWNJ + trim
    text = text.replace(f"{_ZWNJ}{_ZWNJ}", _ZWNJ)

    return text


# ---------------------------------------------------------------------------
# Number normalisation – keep Arabic/Persian digits as-is for technical terms
# ---------------------------------------------------------------------------


def _normalise_numbers(text: str) -> str:
    """Unify digits + percent (parsitext port: 3-way digit unification)."""
    from kb_manager.preprocessor.regex_persian import normalize_digits

    # parsitext: keep display form but unify for indexing; here we unify to ASCII
    # so search BM25 matches ۱۲۳, ١٢٣, 123 equally. Original display preserved upstream.
    text = normalize_digits(text, to="ascii")
    text = text.replace("\u066a", "%")
    return text


def _strip_diacritics(text: str) -> str:
    """Remove Arabic harakat (parsitext diacritics removal)."""
    from kb_manager.preprocessor.regex_persian import strip_diacritics

    return strip_diacritics(text)


def _reduce_repetition(text: str) -> str:
    """Collapse 3+ repetitions to 2 (parsitext). Preserves digit runs."""
    from kb_manager.preprocessor.regex_persian import reduce_repetition

    return reduce_repetition(text)


# ---------------------------------------------------------------------------
# Punctuation normalisation
# ---------------------------------------------------------------------------

_PUNCT_MAP: dict[str, str] = {
    "\u060c": "\u060c",  # ، (already correct)
    "\u061b": "\u060c",  # ؛ → ،
    "\u066b": "\u066b",  # ٫  (Persian decimal separator – keep)
    "\u200b": "",  # zero-width space → remove
    "\ufeff": "",  # BOM → remove
    "\u202b": "",  # RTL embedding → remove
    "\u202c": "",  # pop directional → remove
    "\u202d": "",  # LRO → remove
    "\u202e": "",  # RLO → remove
}


def _fix_punctuation(text: str) -> str:
    """Normalise punctuation and remove invisible formatting chars."""
    for src, dst in _PUNCT_MAP.items():
        text = text.replace(src, dst)
    return text


# ---------------------------------------------------------------------------
# Extra-space removal
# ---------------------------------------------------------------------------

_MULTI_SPACE = re.compile(r"[ \t\u00a0]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def _remove_extra_spaces(text: str) -> str:
    """Collapse runs of whitespace and excessive newlines."""
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Light spell-check (articles only, not technical codes) — REMOVED
# The _COMMON_TYPOS dict contained only identity mappings (no actual corrections).
# _light_spell_check was a no-op. If spell-check is needed, implement properly
# with a real dictionary or integrate with Hazm/Shekar spell-checker.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class PersianPreprocessor:
    """Persian (Farsi) text normaliser.

    The normaliser tries to use **Hazm** or **Shekar** when available and
    falls back to a pure-regex pipeline otherwise.  In all cases the
    ``normalise()`` method returns usable text.

    Parameters
    ----------
    use_hazm:
        Prefer Hazm for tokenisation and normalisation (requires ``hazm``).
    use_shekar:
        Prefer Shekar for normalisation (requires ``shekar``).
    spell_check:
        Run a very light spell-check pass on non-technical words.
    """

    def __init__(
        self,
        *,
        use_hazm: bool = True,
        use_shekar: bool = True,
    ) -> None:
        self._use_hazm = use_hazm and _HAZM_AVAILABLE
        self._use_shekar = use_shekar and _SHEKAR_AVAILABLE

        self._hazm_norm: object | None = None
        if self._use_hazm:
            try:
                self._hazm_norm = HazmNormalizer()  # type: ignore[union-attr]
            except Exception:
                self._use_hazm = False

        self._shekar_norm: object | None = None
        if self._use_shekar:
            try:
                self._shekar_norm = ShekarNormalizer()  # type: ignore[union-attr]
            except Exception:
                self._use_shekar = False

    # ------------------------------------------------------------------

    def _hazm_normalise(self, text: str) -> str:
        """Normalise using Hazm's Normalizer."""
        try:
            return self._hazm_norm.normalize(text)  # type: ignore[union-attr]
        except Exception:
            return text

    def _shekar_normalise(self, text: str) -> str:
        """Normalise using Shekar's Normalizer."""
        try:
            return self._shekar_norm.normalize(text)  # type: ignore[union-attr]
        except Exception:
            return text

    # ------------------------------------------------------------------

    def normalise(self, text: str) -> str:
        """Run the full Persian normalisation pipeline on *text*.

        Order (parsitext-aligned):
        1. Unicode NFC
        2. Arabic → Persian mapping
        3. Shekar/Hazm
        4. ZWNJ fix
        5. Diacritics strip (NEW)
        6. Repetition reduction (NEW)
        7. Number unification (CHANGED: 3-way)
        8. Punctuation fix
        9. Extra-space removal
        """
        if not text:
            return text

        text = unicodedata.normalize("NFC", text)
        text = _apply_unicode_map(text, _ARABIC_TO_PERSIAN)
        text = _apply_unicode_map(text, _PERSIAN_CHARS_ONLY)

        if self._use_shekar:
            text = self._shekar_normalise(text)
        elif self._use_hazm:
            text = self._hazm_normalise(text)

        text = _fix_zwnj(text)
        text = _strip_diacritics(text)
        text = _reduce_repetition(text)
        text = _normalise_numbers(text)
        text = _fix_punctuation(text)
        text = _remove_extra_spaces(text)
        return text

    # Convenience alias
    def __call__(self, text: str) -> str:
        return self.normalise(text)
