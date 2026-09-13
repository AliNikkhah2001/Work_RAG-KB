"""Central Persian regex codepoints — port of Regex.Persian.Language + parsitext ranges.

Source:
- https://github.com/mirhmousavi/Regex.Persian.Language
- https://crates.io/crates/parsitext

Provides single source of truth for character classes used across
preprocessor/persian.py, search.py tokenizer, parsers/xlsx_parser.py.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Codepoint ranges from Regex.Persian.Language
# ---------------------------------------------------------------------------

# Spaces: all Unicode spaces Persian actually uses, including ZWNJ variants
# U+0020, U+2000-U+200F, U+2028-U+202F
SPACE_CODEPOINTS = r"\u0020\u2000-\u200F\u2028-\u202F"

# Persian alphabet (33 letters + hamzah variants)
# U+0621-U+0628, U+062A-U+063A, U+0641-U+0642, U+0644-U+0648,
# U+064E-U+0651, U+0655, U+067E, U+0686, U+0698, U+06A9-U+06AF, U+06BE, U+06CC
PERSIAN_ALPHA_CODEPOINTS = r"\u0621-\u0628\u062A-\u063A\u0641-\u0642\u0644-\u0648\u064E-\u0651\u0655\u067E\u0686\u0698\u06A9-\u06AF\u06BE\u06CC"

# Extended canonical Persian block (includes ZWNJ-aware)
PERSIAN_LETTER_RANGE = r"\u0600-\u06FF\u0750-\u077F"

# Persian (Extended Arabic-Indic) digits ۰-۹
PERSIAN_NUM_CODEPOINTS = r"\u06F0-\u06F9"

# Arabic-Indic digits ٠-٩
ARABIC_NUM_CODEPOINTS = r"\u0660-\u0669"

# Persian/Arabic punctuation: ، ؛ ؟ ـ ٪ ٫ ٬
PERSIAN_PUNCT_CODEPOINTS = r"\u060C\u061B\u061F\u0640\u066A\u066B\u066C"

# Additional Arabic chars that leak into Persian texts
# ة ك ى ي ً ٍ ە
ADDITIONAL_ARABIC_CODEPOINTS = r"\u0629\u0643\u0649-\u064B\u064D\u06D5"

# ---------------------------------------------------------------------------
# Composed patterns (parsitext-inspired)
# ---------------------------------------------------------------------------

# Full Persian word: alpha + optional ZWNJ-joined suffixes
PERSIAN_WORD_RE = re.compile(rf"[{PERSIAN_ALPHA_CODEPOINTS}]+(?:\u200c[{PERSIAN_ALPHA_CODEPOINTS}]+)*")

# Token pattern used for BM25 / search indexing
TOKEN_RE = re.compile(rf"[a-zA-Z{PERSIAN_LETTER_RANGE}\d]+")

# Persian phrase validator (alpha + spaces)
PERSIAN_PHRASE_RE = re.compile(rf"^[{PERSIAN_ALPHA_CODEPOINTS}{SPACE_CODEPOINTS}]+$")

# Number helpers
PERSIAN_DIGIT_RE = re.compile(rf"[{PERSIAN_NUM_CODEPOINTS}]")
ARABIC_DIGIT_RE = re.compile(rf"[{ARABIC_NUM_CODEPOINTS}]")

# Mojibake detection: literal '?' between Arabic-script letters
_MOJIBAKE_RE = re.compile(rf"(?<=[{PERSIAN_ALPHA_CODEPOINTS}])\?(?=[{PERSIAN_ALPHA_CODEPOINTS}])")

# Repetition: 3+ same char (parsitext: collapse to 2, preserves IDs)
_REPETITION_RE = re.compile(r"(.)\1{2,}")

# Diacritics (harakat): U+064B-U+0652, U+0654-U+0656, U+0670
DIACRITICS_RE = re.compile(r"[\u064B-\u0652\u0654-\u0656\u0670]")

# Invisible formatting chars to strip
INVISIBLE_RE = re.compile(r"[\u200b\ufeff\u202a-\u202e\u2066-\u2069]")


# ---------------------------------------------------------------------------
# Mapping tables (shared)
# ---------------------------------------------------------------------------

# Arabic variant → Persian canonical (covers Regex.Persian.Language additional_arabic)
ARABIC_TO_PERSIAN_MAP: dict[str, str] = {
    "\u0629": "\u0647",  # ة → ه
    "\u0643": "\u06a9",  # ك → ک
    "\u0649": "\u06cc",  # ى → ی
    "\u064a": "\u06cc",  # ي → ی
    "\u06d5": "\u0647",  # ە → ه
    # Alef variants
    "\u0622": "\u0627",  # آ → ا (parsitext keeps آ, but normalize for search)
    "\u0623": "\u0627",  # أ → ا
    "\u0625": "\u0627",  # إ → ا
    "\u0671": "\u0627",  # ٱ → ا
    "\u0621": "\u0627",  # ء → ا
    "\u0626": "\u06cc",  # ئ → ی
}

# Unified digit maps
PERSIAN_DIGIT_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
ARABIC_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
DIGIT_UNIFY_MAP = {**PERSIAN_DIGIT_MAP, **ARABIC_DIGIT_MAP}  # type: ignore[arg-type]

# Soundex / phonetic groups (parsitext phonetic::soundex)
# Collapse homophones for fuzzy search: ص=س=ث, ز=ذ=ض=ظ, ت=ط, etc.
PHONETIC_GROUPS: dict[str, str] = {
    "ص": "س", "ث": "س",
    "ز": "ز", "ذ": "ز", "ض": "ز", "ظ": "ز",
    "ت": "ت", "ط": "ت",
    "ح": "ه", "ه": "ه", "ة": "ه",
    "ق": "ق", "غ": "ق",
    "ع": "ا",
}
PHONETIC_MAP = str.maketrans(PHONETIC_GROUPS)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def is_persian(text: str) -> bool:
    """True if text contains any Persian letter."""
    return bool(re.search(rf"[{PERSIAN_ALPHA_CODEPOINTS}]", text))


def is_pure_persian(text: str) -> bool:
    """True if text contains only Persian alpha + spaces."""
    return bool(PERSIAN_PHRASE_RE.match(text)) if text else False


def contains_persian(text: str) -> bool:
    return is_persian(text)


def has_mojibake(text: str) -> bool:
    """Detect literal '?' between Persian letters (classic mojibake signature)."""
    return bool(_MOJIBAKE_RE.search(text))


def build_persian_regex(
    include_spaces: bool = True,
    include_numbers: bool = False,
    include_punct: bool = False,
) -> str:
    """Build a regex char-class string from desired components."""
    parts = [PERSIAN_ALPHA_CODEPOINTS]
    if include_spaces:
        parts.append(SPACE_CODEPOINTS)
    if include_numbers:
        parts.append(PERSIAN_NUM_CODEPOINTS)
        parts.append(ARABIC_NUM_CODEPOINTS)
    if include_punct:
        parts.append(PERSIAN_PUNCT_CODEPOINTS)
    return "".join(parts)


def normalize_digits(text: str, to: str = "ascii") -> str:
    """Unify digits. to='ascii'→0-9, 'persian'→۰-۹, 'arabic'→٠-٩."""
    if to == "ascii":
        return text.translate(DIGIT_UNIFY_MAP)  # type: ignore[arg-type]
    if to == "persian":
        tmp = text.translate(ARABIC_DIGIT_MAP)  # type: ignore[arg-type]
        # ascii → persian
        return tmp.translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))
    if to == "arabic":
        tmp = text.translate(PERSIAN_DIGIT_MAP)  # type: ignore[arg-type]
        return tmp.translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))
    return text


def strip_diacritics(text: str) -> str:
    """Remove Arabic harakat (parsitext diacritics removal)."""
    return DIACRITICS_RE.sub("", text)


def reduce_repetition(text: str) -> str:
    """Collapse 3+ repeated chars to 2 (parsitext: خیییلی→خیلی, preserves IDs)."""
    # Don't collapse digits (IDs like 1111)
    def _repl(m: re.Match[str]) -> str:
        ch = m.group(1)
        if ch.isdigit() or ch in "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩":
            return m.group(0)
        return ch * 2

    return _REPETITION_RE.sub(_repl, text)


def to_soundex(text: str) -> str:
    """Phonetic soundex: collapse homophone groups for fuzzy matching."""
    return text.translate(PHONETIC_MAP)


def sanitize_invisible(text: str) -> str:
    return INVISIBLE_RE.sub("", text)
