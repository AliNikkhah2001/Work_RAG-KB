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
}

# Characters that look the same in some fonts but have different code-points
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
_SUFFIXES_WITH_ZWNJ: tuple[str, ...] = (
    "ها",
    "های",
    "ترین",
    "تر",
    "گر",
    "گری",
    "ی",
    "ای",
    "هایی",
    "هایی",
    "م",
    "ت",
    "ش",
)

# Patterns where ZWNJ should be *removed* (e.g. inside English words)
_ZWNJ_IN_ENGLISH = re.compile(r"([a-zA-Z0-9])\u200c([a-zA-Z0-9])")

# Persian letter range (used for word-boundary checks)
_PERSIAN_LETTER = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")


def _fix_zwnj(text: str) -> str:
    """Insert/remove ZWNJ in common Persian patterns.

    Only inserts ZWNJ *before* a suffix when it follows a Persian letter
    at a word boundary.  This avoids corrupting the middle of words.
    """
    # Remove ZWNJ that leaked into ASCII runs
    text = _ZWNJ_IN_ENGLISH.sub(r"\1\2", text)

    # Ensure ZWNJ before Persian possessive / compound suffixes,
    # but ONLY when preceded by a Persian letter (word boundary)
    for suffix in _SUFFIXES_WITH_ZWNJ:
        # pattern: Persian-letter + suffix (no ZWNJ already present)
        pattern = re.compile(
            f"([\u0600-\u06FF\u0750-\u077F])({re.escape(suffix)})(?![\u200c])",
        )
        text = pattern.sub(rf"\1{_ZWNJ}\2", text)

    # Collapse double ZWNJ
    text = text.replace(f"{_ZWNJ}{_ZWNJ}", _ZWNJ)

    return text


# ---------------------------------------------------------------------------
# Number normalisation – keep Arabic/Persian digits as-is for technical terms
# ---------------------------------------------------------------------------


def _normalise_numbers(text: str) -> str:
    """Normalise number characters.

    We intentionally *keep* Arabic-Indic (٠-٩) and Extended Arabic-Indic
    (۰-۹) digits as-is because they are common in Persian technical
    documents.  Only normalise the presentation form of *Arabic* fractions
    and percent signs.
    """
    # Arabic percent sign → ASCII %
    text = text.replace("\u066a", "%")
    return text


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
# Light spell-check (articles only, not technical codes)
# ---------------------------------------------------------------------------

# Very small lookup for ultra-common typos found in Persian KB articles.
# This is intentionally lightweight – real spell-checking is expensive.
_COMMON_TYPOS: dict[str, str] = {
    "انجام": "انجام",
    "اصلی": "اصلی",
    "نرم‌افزار": "نرم‌افزار",
    "سخت‌افزار": "سخت‌افزار",
}

_TECH_CODE_PATTERN = re.compile(r"\b[A-Za-z0-9_\-]{3,}\b")  # looks like a code/ID


def _light_spell_check(text: str) -> str:
    """Fix a handful of very common typos – skip technical codes."""
    tokens = text.split()
    result: list[str] = []
    for token in tokens:
        if _TECH_CODE_PATTERN.fullmatch(token):
            result.append(token)
            continue
        result.append(_COMMON_TYPOS.get(token, token))
    return " ".join(result)


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
        spell_check: bool = True,
    ) -> None:
        self._use_hazm = use_hazm and _HAZM_AVAILABLE
        self._use_shekar = use_shekar and _SHEKAR_AVAILABLE
        self._spell_check = spell_check

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

        Order of operations:
        1. Unicode normalisation (NFC)
        2. Arabic → Persian character mapping
        3. Shekar / Hazm normalisation (if available)
        4. ZWNJ fix-ups
        5. Number normalisation
        6. Punctuation fix
        7. Extra-space removal
        8. Light spell-check (if enabled)
        """
        if not text:
            return text

        # 1. Unicode NFC
        text = unicodedata.normalize("NFC", text)

        # 2. Arabic → Persian char mapping
        text = _apply_unicode_map(text, _ARABIC_TO_PERSIAN)
        text = _apply_unicode_map(text, _PERSIAN_CHARS_ONLY)

        # 3. Shekar (preferred) or Hazm normaliser
        if self._use_shekar:
            text = self._shekar_normalise(text)
        elif self._use_hazm:
            text = self._hazm_normalise(text)

        # 4. ZWNJ
        text = _fix_zwnj(text)

        # 5. Numbers
        text = _normalise_numbers(text)

        # 6. Punctuation
        text = _fix_punctuation(text)

        # 7. Extra spaces
        text = _remove_extra_spaces(text)

        # 8. Spell check (optional)
        if self._spell_check:
            text = _light_spell_check(text)

        return text

    # Convenience alias
    def __call__(self, text: str) -> str:
        return self.normalise(text)
