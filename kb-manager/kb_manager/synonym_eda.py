"""EDA for synonym-map generation: build a compact domain profile from KB chunks.

Pipeline:
    KB chunks -> normalize (strip ZWNJ/diacritics, arabic->persian) -> tokenize
    -> term frequency / co-occurrence / keyword stats -> compact profile JSON
    -> fed to the LLM workbench (synonym_workbench.py) as grounding context.

The corpus stores ZWNJ between nearly every letter, so we strip it BEFORE
tokenizing so words like ``س|م|ت`` recover as ``سمت``. Without this step TF
would be garbage.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from kb_manager.preprocessor.regex_persian import (
    ARABIC_TO_PERSIAN_MAP,
    DIACRITICS_RE,
    PERSIAN_LETTER_RANGE,
    TOKEN_RE,
)

PROFILE_PATH = Path(__file__).resolve().parent.parent / "data" / "domain_profile.json"

_ZWNJ = "\u200c"

# Corpus-wide stopword-ish tokens (function words, verbs like "است / می شود")
_FUNCTION_WORDS = frozenset(
    """
    از در به و با برای که این آن را است هستند بودند می باشد می شود می گردد
    می کند هر دو آیا یا اگر ولی تا باشد بر اساس طبق طریق نیز همچنین همچنین
    نیز درباره بین توسط مانند مثل طی خود کنید گردد باید یک یکی شود رأی تنها
    هم نیست بود شده دارد داشت شوند باید یا آیا چنان چه کدام چرا چگونه
    the a an is are was were be been am does do did have has had in on at
    to for of and or but not no so if it its this that can will would should
    could may might shall hingga
    """.split()
)

_FORBIDDEN = re.compile(rf"[^\w\u200c{PERSIAN_LETTER_RANGE}\u0600-\u06ff]+", re.UNICODE)


@dataclass
class EDAResult:
    n_chunks: int = 0
    n_tokens: int = 0
    vocab: int = 0
    term_freq: Counter = field(default_factory=Counter)
    term_docfreq: Counter = field(default_factory=Counter)
    cooccur: Counter = field(default_factory=Counter)
    doc_distinct: Counter = field(default_factory=Counter)
    keyword_counter: Counter = field(default_factory=Counter)


def _normalize_for_eda(text: str) -> str:
    """Strip ZWNJ + diacritics + map Arabic->Persian so terms are searchable tokens."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace(_ZWNJ, "")
    for src, dst in ARABIC_TO_PERSIAN_MAP.items():
        text = text.replace(src, dst)
    text = DIACRITICS_RE.sub("", text)
    text = _FORBIDDEN.sub(" ", text)
    return text


def _tokens(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text) if len(t) > 1 and t.lower() not in _FUNCTION_WORDS]


def run_eda_from_chunks(chunks: list[str]) -> EDAResult:
    """Compute TF / document-frequency / co-occurrence from raw chunk texts."""
    r = EDAResult(n_chunks=len(chunks))
    window = 6
    for raw in chunks:
        norm = _normalize_for_eda(raw)
        toks = _tokens(norm)
        r.n_tokens += len(toks)
        uniq = set(toks)
        r.vocab += len(uniq)
        r.term_freq.update(toks)
        r.term_docfreq.update(uniq)
        for i, t in enumerate(toks):
            for j in range(max(0, i - window), min(len(toks), i + window + 1)):
                if i != j:
                    r.cooccur[tuple(sorted((t, toks[j])))] += 1
    return r


def aggregate_keywords(keyword_lists: list[list[str]]) -> Counter:
    c: Counter = Counter()
    for kws in keyword_lists:
        c.update(kws)
    return c


def _top(c: Counter, n: int) -> list[str]:
    return [t for t, _ in c.most_common(n)]


def _top_pairs(c: Counter, n: int) -> list[list[str]]:
    return [list(p) for p, _ in c.most_common(n)]


# ---------------------------------------------------------------------------
# Profile builders
# ---------------------------------------------------------------------------

def build_domain_profile(
    r: EDAResult,
    keyword_counts: Counter | None = None,
    *,
    top_terms: int = 300,
    top_pairs: int = 60,
    min_freq: int = 3,
) -> dict:
    """Assemble the compact JSON that grounds the LLM prompt."""
    filtered = Counter({t: f for t, f in r.term_freq.items() if f >= min_freq})
    pairs_filtered = Counter({p: c for p, c in r.cooccur.items() if c >= 2})

    profile = {
        "domain": "Iranian credit scoring / banking knowledge base (Persian)",
        "corpus": {
            "chunks": r.n_chunks,
            "total_tokens": r.n_tokens,
            "unique_tokens": r.vocab,
        },
        "top_terms": _top(filtered, top_terms),
        "term_frequency": {t: f for t, f in filtered.most_common(top_terms // 2)},
        "cooccurrence_pairs": _top_pairs(pairs_filtered, top_pairs),
        "document_frequency": {t: f for t, f in r.term_docfreq.most_common(top_terms // 2)},
        "keywords_from_chunks": _top(keyword_counts, top_terms // 2) if keyword_counts else [],
    }
    return profile


def save_profile(profile: dict, path: Path | None = None) -> Path:
    path = path or PROFILE_PATH
    path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_profile(path: Path | None = None) -> dict:
    path = path or PROFILE_PATH
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import asyncio

    from sqlalchemy import select

    from kb_manager.config import PROJECT_ROOT
    from kb_manager.models.database import Chunk
    from kb_manager.web.deps import db

    parser = argparse.ArgumentParser(description="Build domain profile for synonym-map LLM generation")
    parser.add_argument("--top-terms", type=int, default=300)
    parser.add_argument("--top-pairs", type=int, default=60)
    parser.add_argument("--min-freq", type=int, default=3)
    parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "data" / "domain_profile.json"))
    args = parser.parse_args()

    async def _load() -> tuple[list[str], list[list[str]]]:
        async with db.session() as session:
            result = await session.execute(select(Chunk.content, Chunk.keywords))
            rows = result.all()
        return [r[0] or "" for r in rows], [r[1] or [] for r in rows]

    texts, kw_lists = asyncio.run(_load())
    eda = run_eda_from_chunks(texts)
    kw_counts = aggregate_keywords(kw_lists)
    profile = build_domain_profile(
        eda,
        kw_counts,
        top_terms=args.top_terms,
        top_pairs=args.top_pairs,
        min_freq=args.min_freq,
    )
    out = save_profile(profile, Path(args.out))
    print(f"chunks={eda.n_chunks} tokens={eda.n_tokens} unique≈{eda.vocab}")
    print(f"top 20 terms: {profile['top_terms'][:20]}")
    print(f"top 10 pairs: {profile['cooccurrence_pairs'][:10]}")
    print(f"saved profile -> {out}")


if __name__ == "__main__":
    main()