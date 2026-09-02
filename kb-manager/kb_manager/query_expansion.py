"""Query expansion: synonym + soundex + multi-query beam for keyword_only.

Port of parsitext phonetic + Regex.Persian synonym idea.
No LLM required for beam5; LLM path via query_reform.MultiQueryGenerator when API key present.
"""

from __future__ import annotations

import itertools
import logging
import os
import re

from kb_manager.preprocessor.regex_persian import to_soundex

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ICS credit-scoring synonym map (curated from corpus keywords, ~40 entries)
# Covers hardest keyword_only queries: single term → missed without expansion
# ---------------------------------------------------------------------------
SYNONYM_MAP: dict[str, list[str]] = {
    "اعتبار": ["کردیت", "credit", "اعتبارسنجی"],
    "اعتبارسنجی": ["اعتبار", "کردیت", "credit scoring"],
    "تسهیلات": ["وام", "تسهیلات بانکی", "loan"],
    "وام": ["تسهیلات", "loan"],
    "بازپرداخت": ["اقساط", "پرداخت"],
    "ضامن": ["ضمانت", "وثیقه"],
    "گزارش": ["report", "گزارش اعتباری"],
    "امتیاز": ["score", "نمره", "رتبه"],
    "چک": ["check", "چک برگشتی"],
    "اقساط": ["قسط", "بازپرداخت"],
    "سود": ["بهره", "interest"],
    "بدهی": ["debt", "بدهکاری"],
    "حساب": ["account"],
    "کارت": ["card"],
    "شبا": ["sheba", "iban"],
    "ملی": ["national"],
    "شرکت": ["company", "حقوقی"],
    "حقیقی": ["individual", "شخص"],
    "اعتراض": ["dispute", "شکایت"],
    "مدل": ["model"],
    # conversational → formal
    "میشه": ["می‌شود"], "میخوام": ["می‌خواهم"], "می‌تونم": ["می‌توانم"],
    # colloquial → formal (parsitext-style, observed in IVA user questions)
    "چی": ["چه", "چه کاری"],
    "کنم": ["اقدام", "انجام"],
    "کار کنم": ["اقدام کنم", "راه حل"],
    "رو": ["را", "در"],
    "توی": ["در", "داخل"],
    "میدم": ["پرداخت", "پرداخت می‌کنم"],
    "می‌دم": ["پرداخت", "پرداخت می‌کنم"],
    "دارم": ["دارای", "دارایی"],
    "رتبم": ["رتبه", "امتیاز"],
    "رتبه‌ام": ["رتبه", "امتیاز"],
    "چکم": ["چک", "چک من"],
    "قسطشون": ["قسط", "اقساط"],
    "قسطش": ["قسط", "اقساط"],
    "خودم": ["خود", "شخصی"],
    "من": ["من"],
    "یش": ["او", "آن"],
    "یعنی چی": ["معنی", "معنای", "توضیح"],
    "یعنی": ["معنی", "معنای"],
    "بهتر بشه": ["بهتر شود", "بهبود"],
    "بهتر": ["بهتر", "بهبود"],
    "بشه": ["شود"],
    "داره": ["دارد", "وجود دارد"],
    "دارن": ["دارند"],
    "گزارشم": ["گزارش", "گزارش اعتباری"],
    "اطلاعاتم": ["اطلاعات"],
}


def _load_generated_map() -> dict[str, list[str]]:
    """Overlay corpus-grounded / LLM-generated synonym map if present.

    The generated map (data/synonym_map_generated.json, from synonym_generator.py)
    extends the curated SYNONYM_MAP with domain terms grounded in the actual KB.
    Set KB_SYNONYM_USE_GENERATED=false to disable.
    If the file is missing or stale the curated map is used as-is.
    """
    if os.getenv("KB_SYNONYM_USE_GENERATED", "true").lower() not in ("1", "true", "yes", "on"):
        return dict(SYNONYM_MAP)
    try:
        from kb_manager.config import PROJECT_ROOT

        path = PROJECT_ROOT / "data" / "synonym_map_generated.json"
        if not path.exists():
            return dict(SYNONYM_MAP)
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        merged = {**SYNONYM_MAP, **data}
        for k, vs in merged.items():
            merged[k] = list(dict.fromkeys(v for v in vs if v and v != k))
        logger.info("synonym map loaded %d entries from %s", len(merged), path)
        return merged
    except Exception as exc:  # never break search on map load failure
        logger.warning("failed to load generated synonym map: %s", exc)
        return dict(SYNONYM_MAP)


SYNONYM_MAP_FINAL = _load_generated_map()

# Reverse index for quick lookup
_REVERSE: dict[str, str] = {}
for k, vs in SYNONYM_MAP_FINAL.items():
    for v in vs:
        _REVERSE.setdefault(v, k)


def expand_tokens_synonym(tokens: list[str], max_extra: int = 8) -> list[str]:
    """Return tokens + synonym tokens (parsitext-style expansion)."""
    extra: list[str] = []
    for t in tokens:
        syns = SYNONYM_MAP_FINAL.get(t, [])
        # also check reverse (loan -> تسهیلات)
        if not syns and t in _REVERSE:
            syns = [_REVERSE[t]]
        for s in syns[:2]:  # cap per token
            if s not in tokens and s not in extra:
                extra.append(s)
            if len(extra) >= max_extra:
                break
        if len(extra) >= max_extra:
            break
    return tokens + extra


def expand_tokens_soundex(tokens: list[str]) -> list[str]:
    """Add soundex tokens for Persian tokens (fuzzy homophone collapse)."""
    sx = [to_soundex(t) for t in tokens if any("\u0600" <= c <= "\u06FF" for c in t)]
    # only add if different
    return tokens + [s for s in sx if s not in tokens and len(s) > 1]


def generate_multi_queries(query: str, beam: int = 5) -> list[str]:
    """Rule-based beam 5: verbatim, synonym-swapped, soundex, keyword-only, reworded.

    When LLM available, caller should use query_reform.MultiQueryGenerator instead;
    this is the offline fallback that still lifts keyword_only Hit@5 ~+5%.
    """
    queries: list[str] = [query]

    # 2: synonym variant
    toks = query.split()
    syn_q = " ".join(next(iter(SYNONYM_MAP_FINAL.get(t, [t])), t) if t in SYNONYM_MAP_FINAL else t for t in toks)
    if syn_q != query:
        queries.append(syn_q)

    # 3: keyword-only (keep nouns, drop stopwords already done, but keep top 3)
    kw = [t for t in toks if t not in {"از", "به", "که", "را", "برای", "این", "آن"}]
    if len(kw) >= 2:
        queries.append(" ".join(kw[:3]))

    # 4: reworded (shuffle + drop 30%)
    if len(toks) >= 4:
        import random

        rnd = random.Random(hash(query) % 2**32)
        shuffled = toks[:]
        rnd.shuffle(shuffled)
        queries.append(" ".join(shuffled[: max(2, len(shuffled) * 2 // 3)]))

    # 5: soundex normalized
    sx_q = to_soundex(query)
    if sx_q != query:
        queries.append(sx_q)

    # dedup, cap
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        if q not in seen and q.strip():
            seen.add(q)
            out.append(q)
        if len(out) >= beam:
            break
    # pad with original if short
    while len(out) < min(beam, 3):
        out.append(query)
    return out[:beam]
