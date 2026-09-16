"""Query enhancement: rule-based Persian expansion + light rewrite beams.

Reuses the colloquial-to-formal map from ``kb_manager.query_expansion``
(``SYNONYM_MAP_FINAL``, fallback ``SYNONYM_MAP``) — the map is imported,
never duplicated here.

LLM multi-query lives in ``kb_manager.query_reform.MultiQueryGenerator``
and is intentionally NOT duplicated here. This module is fully offline:
no network calls. ``maybe_llm_rewrite`` is a mock-only gate.
"""

from __future__ import annotations

import os
import re

try:  # prefer corpus-grounded overlay when present
    from kb_manager.query_expansion import SYNONYM_MAP_FINAL as _EXPANSION_MAP
except ImportError:  # pragma: no cover - fallback for minimal installs
    from kb_manager.query_expansion import SYNONYM_MAP as _EXPANSION_MAP  # type: ignore[no-redef]

try:
    from kb_manager.preprocessor.regex_persian import (
        ARABIC_TO_PERSIAN_MAP as _AR_MAP,
    )
except Exception:  # pragma: no cover
    _AR_MAP = {"\u064a": "\u06cc", "\u0643": "\u06a9"}

_AR_TRANSLATE = str.maketrans(_AR_MAP)

_PUNCT_RE = re.compile(r"[؟?!.،,؛;:()\[\]\"'«»…\-–—/\\|_+=*<>@#$%^&~`]+")
_WS_RE = re.compile(r"\s+")

_MOCK_ENV_VALUES = ("1", "true", "yes", "on")


def _apply_rule_expansion(query: str) -> str:
    """Apply colloquial→formal replacements using the shared expansion map.

    Single-token keys are replaced on token boundaries; multi-word keys
    (containing a space) are replaced longest-first via substring match.
    The first synonym entry is used as the canonical formal form.
    """
    toks = query.split()
    mapped: list[str] = []
    for tok in toks:
        syns = _EXPANSION_MAP.get(tok, [])
        if syns and syns[0] and syns[0] != tok:
            mapped.append(syns[0])
        else:
            mapped.append(tok)
    text = " ".join(mapped)

    multi_keys = [k for k in _EXPANSION_MAP.keys() if " " in k and k.strip()]
    multi_keys.sort(key=len, reverse=True)
    for key in multi_keys:
        if key in text:
            syns = _EXPANSION_MAP.get(key, [])
            if not syns or not syns[0] or syns[0] == key:
                continue
            text = text.replace(key, syns[0])

    return _WS_RE.sub(" ", text).strip()


def _normalize_whitespace(query: str) -> str:
    return _WS_RE.sub(" ", query).strip()


def _strip_punctuation(query: str) -> str:
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", query)).strip()


def _normalize_orthography(query: str) -> str:
    text = query.translate(_AR_TRANSLATE)
    text = text.replace("‌", " ")  # ZWNJ -> space for tokenization variant
    return _WS_RE.sub(" ", text).strip()


def enhance_query(query: str) -> dict:
    """Rule-based Persian query enhancement.

    Returns ``{"enhanced": str, "beams": list[str]}`` where ``beams`` holds
    up to 5 variants (original first, then expanded + 3 light rewrites),
    deduplicated while preserving order.
    """
    if not query or not query.strip():
        return {"enhanced": "", "beams": []}

    original = query
    expanded = _apply_rule_expansion(query)

    candidates = [
        original,
        expanded,
        _normalize_whitespace(query),
        _strip_punctuation(query),
        _normalize_orthography(query),
    ]

    seen: set[str] = set()
    beams: list[str] = []
    for cand in candidates:
        if cand and cand not in seen:
            seen.add(cand)
            beams.append(cand)
        if len(beams) >= 5:
            break

    return {"enhanced": expanded, "beams": beams[:5]}


def maybe_llm_rewrite(query: str, allow_mock_only: bool = True) -> str | None:
    """Mock-only LLM rewrite gate. Never performs network calls.

    Returns ``None`` unless the ``KB_ALLOW_MOCK=true`` env flag is set
    (and ``allow_mock_only`` is True), in which case a deterministic mock
    variant is returned. The real LLM path lives in
    ``kb_manager.query_reform.MultiQueryGenerator`` and requires an
    explicit client — it is never invoked from here.
    """
    if not allow_mock_only:
        return None
    if not query or not query.strip():
        return None
    flag = os.getenv("KB_ALLOW_MOCK", "false").strip().lower()
    if flag not in _MOCK_ENV_VALUES:
        return None
    return f"{query.strip()} [mock-rewrite]"
