"""Generate an expanded synonym map for query expansion.

Two sources feed SYNONYM_MAP:

1. **Corpus-grounded seed** (``generate_seed_map``): deterministic, no LLM.
   Uses the EDA domain profile (top frequencies + co-occurrence) plus a
   curated domain lexicon for the ICS credit-scoring corpus (Persian + the
   English feature tokens like ``score``/``bin`` that actually appear in
   chunks).  This is the offline fallback that always runs.

2. **LLM expansion** (``generate_map_with_llm``): feeds the compact domain
   profile + prompt to a SOTA model (OpenAI/Ollama/vLLM via ``kb_manager.llm``)
   and parses the returned JSON.  Only runs when a key / server is configured.
   The LLM receives actual corpus terms as grounding so it cannot hallucinate
   out-of-domain synonyms.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from kb_manager.config import PROJECT_ROOT
from kb_manager.query_expansion import SYNONYM_MAP
from kb_manager.synonym_eda import PROFILE_PATH, load_profile

GENERATED_PATH = PROJECT_ROOT / "data" / "synonym_map_generated.json"

_PERSIAN = re.compile(r"[\u0600-\u06FF]")


def _is_persian(t: str) -> bool:
    return bool(_PERSIAN.search(t))


# ---------------------------------------------------------------------------
# Corpus-grounded seed (offline, deterministic)
# ---------------------------------------------------------------------------

# Curated cross-lingual + domain lexicon for the ICS credit-scoring corpus.
# Persian <-> English feature tokens plus formal/colloquial Persian variants.
SEED_LEXICON: dict[str, list[str]] = {
    # core credit terms (English tokens actually present in chunks)
    "امتیاز": ["score", "نقطه", "نمره اعتباری"],
    "اعتباری": ["credit", "گزارش اعتباری"],
    "اعتبار": ["credit", "کردیت", "اعتبارسنجی"],
    "چک": ["cheque", "check", "چک برگشتی"],
    "برگشتی": ["چک برگشتی", "returned"],
    "اعتبارسنجی": ["credit scoring", "credit check", "ارزیابی اعتباری"],
    "مالی": ["financial", "مالیاتی"],
    "کاهش": ["decrease", "کم", "افت", "کاهش امتیاز"],
    "شرکت": ["company", "حقوقی", "سازمان"],
    "گزارش": ["report", "گزارش اعتباری", "گزارش امتیاز"],
    "پرداخت": ["payment", "تسویه", "پرداخت بدهی"],
    "بازپرداخت": ["repayment", "اقساط", "پرداخت"],
    "ریسک": ["risk", "خطر", "ریسک اعتباری"],
    "سابقه": ["history", "سوابق", "پیشینه"],
    "مدل": ["model", "روش"],
    "منفی": ["negative", "منفی بودن"],
    "تاخیر": ["delayed", "دیرکرد", "تاخیر پرداخت"],
    "وضعیت": ["status", "حالت"],
    "افزایش": ["increase", "زیاد", "رشد"],
    "فرد": ["person", "حقیقی", "شخص"],
    "تاثیر": ["impact", "اثر"],
    "اطلاعات": ["information", "داده", "دادهها"],
    "ضامن": ["guarantor", "ضمانت", "وثیقه"],
    "تسویه": ["settlement", "پرداخت", "تسویه حساب"],
    "معوق": ["overdue", "معوقه", "تاخیر"],
    "وام": ["loan", "تسهیلات", "وام بانکی"],
    "تسهیلات": ["facility", "وام", "تسهیلات بانکی"],
    "حساب": ["account", "حساب بانکی"],
    "بدهی": ["debt", "بدهکاری", "بدهی معوق"],
    "شخص": ["person", "فرد", "حقیقی"],
    "حقیقی": ["individual", "شخص", "فرد"],
    "حقوقی": ["legal", "شرکت", "سازمان"],
    # English feature tokens present in chunks
    "score": ["امتیاز", "نمره"],
    "bin": ["بانک شناسه", "کد"],
    "impact": ["تاثیر", "اثر"],
    "feature": ["ویژگی", "مشخصه"],
    "model": ["مدل", "مدل اعتبارسنجی"],
    "reason": ["دلیل", "علت"],
    # conversational -> formal
    "میشه": ["می شود"],
    "میخوام": ["می خواهم"],
    "میتونه": ["می تواند"],
    "تس": [] ,
}

# Persian tokens that should NOT be lemmatized/aliased (stopword-ish noise)
_SEED_SKIP = {
    "شما", "سال", "ماه", "داشته", "شود", "باشید", "کند", "کرد", "کرده", "دارد",
    "بوده", "همه", "بسیار", "ایا", "هایی", "وی", "ها",
}


def generate_seed_map() -> dict[str, list[str]]:
    """Blend curated seed lexicon + existing SYNONYM_MAP.

    Only includes keys that are either present in the domain profile
    (high-frequency corpus tokens) or already curated.  This avoids adding
    cold/unused terms that would waste the expansion budget.
    """
    profile = load_profile() if PROFILE_PATH.exists() else {"top_terms": []}
    top_terms = set(profile.get("top_terms", []))

    merged: dict[str, list[str]] = {k: list(v) for k, v in SYNONYM_MAP.items()}
    for key, vals in SEED_LEXICON.items():
        key_ok = key in top_terms or any(v in top_terms for v in vals)
        if key_ok:
            merged.setdefault(key, []).extend(v for v in vals if v and v not in merged.get(key, []))

    # ensure reverse direction for curated pairs (e.g. English -> Persian)
    reverse: dict[str, list[str]] = {}
    for k, vs in merged.items():
        if not vs:
            reverse.setdefault(k, [])
            continue
        for v in vs:
            if _is_persian(k) != _is_persian(v) and v not in reverse:
                reverse.setdefault(v, []).append(k)
    for rk, rv in reverse.items():
        if rk not in top_terms and rk not in merged:
            continue
        merged.setdefault(rk, []).extend(x for x in rv if x not in merged.get(rk, []))

    # drop empty / pathological entries
    merged = {k: vs for k, vs in merged.items() if vs and k not in _SEED_SKIP}
    return merged


# ---------------------------------------------------------------------------
# LLM expansion (optional; needs key or local server)
# ---------------------------------------------------------------------------

LLM_PROMPT_TEMPLATE = """You are expanding a Persian banking / credit-scoring synonym map for bilingual (fa/en) BM25 query expansion.

GROUND TRUTH FROM THE ACTUAL KNOWLEDGE BASE (domain profile):
{profile_json}

TASK
----
Return a JSON object with one key "synonym_map": a dict mapping each key term
to a list of 2-6 search-worthy aliases. For every Persian domain term, provide
its English equivalents used in banking AND other Persian spellings/variants
(formal, colloquial, chunk wording). For English feature tokens (score, bin,
model, impact, feature, reason...), map back to the Persian terms users type.

RULES
- ONLY use terms that plausibly appear in this credit-scoring domain. No
  generic filler synonyms, no definitions, no translations of adjectives.
- Prefer terms from the GROUND TRUTH list; you may add established banking
  Persian synonyms (اش. formal/colloquial variants) for the top-200 terms.
- Multi-word aliases allowed (e.g. "گزارش اعتباری").
- Include the conversational->formal subset implicitly (میشه -> می شود).
- Output MUST be valid JSON: {{"synonym_map": {{...}}}}. No markdown, no comments.

Respond with only the JSON object.
"""


def _parse_llm_json(text: str) -> dict:
    """Extract the synonym_map object from an LLM response (tolerate fences)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            data = json.loads(m.group(0))
        else:
            raise
    return data.get("synonym_map", data)


def generate_map_with_llm(
    model: str = "",
    backend: str = "",
    base_url: str = "",
    api_key: str = "",
) -> dict[str, list[str]]:
    """Call a SOTA LLM through kb_manager.llm to expand the synonym map."""
    from kb_manager.llm import create_llm_client

    client = create_llm_client(
        backend=backend or os.getenv("KB_LLM_BACKEND", "openai"),
        model=model or os.getenv(
            "KB_LLM_MODEL", os.getenv("KB_HYDE_LLM", "gpt-4o-mini")
        ),
        base_url=base_url or os.getenv("KB_LLM_BASE_URL", os.getenv("OPENAI_BASE_URL")),
        api_key=api_key or os.getenv("KB_HYDE_API_KEY", os.getenv("OPENAI_API_KEY")),
    )
    profile = load_profile()
    prompt = LLM_PROMPT_TEMPLATE.format(
        profile_json=json.dumps(profile, ensure_ascii=False, indent=2)[:24000]
    )
    resp = client.generate(prompt, max_tokens=12000, temperature=0.1)
    llm_map = _parse_llm_json(resp.text)

    # merge with curated seed (curated wins for keys it defines)
    final = {**llm_map, **generate_seed_map()}
    for k, vs in final.items():
        final[k] = list(dict.fromkeys(v for v in vs if v and v != k))
    return final


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_generated_map(mapping: dict[str, list[str]], path: Path | None = None) -> Path:
    path = path or GENERATED_PATH
    path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_generated_map(path: Path | None = None) -> dict[str, list[str]] | None:
    path = path or GENERATED_PATH
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate expanded synonym map")
    parser.add_argument("--llm", action="store_true", help="Use LLM expansion (requires API key)")
    parser.add_argument("--backend", default="", help="openai|ollama|vllm")
    parser.add_argument("--model", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if args.llm:
        mapping = generate_map_with_llm(model=args.model, backend=args.backend)
        print(f"LLM map: {len(mapping)} keys")
    else:
        mapping = generate_seed_map()
        print(f"Seed map: {len(mapping)} keys (offline, corpus-grounded)")

    out = save_generated_map(mapping, Path(args.out) if args.out else None)
    print(f"saved -> {out}")
    for k, vs in sorted(mapping.items())[:60]:
        print(f"  {k}: {vs}")


if __name__ == "__main__":
    main()