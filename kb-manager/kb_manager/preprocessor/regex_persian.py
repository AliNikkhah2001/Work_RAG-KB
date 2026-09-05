"""Compatibility shim for old regex_persian imports"""
try:
    from kb_manager.preprocessor.persian import (
        has_mojibake,
        ARABIC_TO_PERSIAN_MAP,
        DIACRITICS_RE,
        TOKEN_RE,
    )
except ImportError:
    # Fallback: define minimal versions
    import re
    def has_mojibake(text: str) -> bool:
        return "\ufffd" in text or ("?" in text and any(0x0600 <= ord(c) <= 0x06FF for c in text))

    ARABIC_TO_PERSIAN_MAP = {
        "\u0643": "\u06a9",
        "\u0649": "\u06cc",
        "\u0629": "\u0647",
    }
    DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670]")
    TOKEN_RE = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)

# Re-export for old code
__all__ = ["has_mojibake", "ARABIC_TO_PERSIAN_MAP", "DIACRITICS_RE", "TOKEN_RE"]
