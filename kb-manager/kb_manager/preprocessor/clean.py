"""Generic text cleaning utilities for knowledge base documents."""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# HTML tags (opening, closing, self-closing, and comments)
_HTML_TAG = re.compile(r"<[^>]+>")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

# URLs – http(s), ftp, and protocol-relative
_URL = re.compile(
    r"https?://[^\s<>\"']+|ftp://[^\s<>\"']+|//[^\s<>\"']+",
    re.IGNORECASE,
)

# Emails
_EMAIL = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")

# Temp-file markers (Office lock files, etc.)
_TEMP_MARKER = re.compile(r"~\$[^\s]*")

# BOM and invisible formatting characters (ZWNJ \u200c is intentionally kept —
# it is a vital part of Persian script used to separate compound words)
_BOM_CHARS = re.compile(
    r"[\ufeff\u200b\u200d\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069]"
)

# Multiple consecutive whitespace (but not newlines)
_MULTI_SPACE = re.compile(r"[^\S\n]{2,}")

# Multiple blank lines
_MULTI_NEWLINE = re.compile(r"\n{3,}")

# Surrogate pairs that can appear in badly-encoded files
_SURROGATE = re.compile(r"[\ud800-\udfff]")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def remove_html(text: str) -> str:
    """Strip HTML tags and comments from *text*.

    Returns the inner text only; no entity decoding is performed beyond
    what Python's built-in html module provides.
    """
    text = _HTML_COMMENT.sub("", text)
    text = _HTML_TAG.sub(" ", text)
    return text


def remove_urls(text: str, placeholder: str = "[URL]") -> str:
    """Replace URLs with *placeholder*."""
    return _URL.sub(placeholder, text)


def remove_emails(text: str, placeholder: str = "[EMAIL]") -> str:
    """Replace email addresses with *placeholder*."""
    return _EMAIL.sub(placeholder, text)


def remove_temp_markers(text: str) -> str:
    """Remove temp-file markers like ``~$Document1.docx``."""
    return _TEMP_MARKER.sub("", text)


def strip_bom(text: str) -> str:
    """Remove BOM characters and invisible Unicode formatting marks."""
    return _BOM_CHARS.sub("", text)


def collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace and excessive newlines.

    A single blank line (``\\n\\n``) is preserved as a paragraph separator.
    """
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def remove_surrogates(text: str) -> str:
    """Strip surrogate code-points that cause encoding errors."""
    return _SURROGATE.sub("", text)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Run the full generic cleaning pipeline on *text*.

    Steps (in order):
    1. Remove surrogate characters
    2. Strip BOM / invisible formatting
    3. Remove HTML tags and comments
    4. Replace URLs with ``[URL]`` placeholder
    5. Replace emails with ``[EMAIL]`` placeholder
    6. Remove temp-file markers (``~$…``)
    7. Collapse multiple whitespace / blank lines

    Returns the cleaned string.
    """
    if not text:
        return text

    text = remove_surrogates(text)
    text = strip_bom(text)
    text = remove_html(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_temp_markers(text)
    text = collapse_whitespace(text)

    return text
