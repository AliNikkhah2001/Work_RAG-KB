"""Generate leveled (easy/medium/hard) + multi-format test questions from KB QA data.

For each sampled ground-truth question we produce six variants whose
difficulty is determined by lexical similarity to the original wording:

    verbatim       ~ near-verbatim copy           (easy)
    paraphrase     ~ synonym swaps + reorder      (medium)
    typo           ~ realistic spelling errors    (medium)
    reworded       ~ heavy reword + keyword drop  (hard)
    keyword_only   ~ only the keyword list        (hard)
    conversational ~ informal spoken rephrase     (hard)

Output is written as an evaluation-style JSON dataset that can be consumed
by the benchmark runner, plus a readable report file.
"""

from __future__ import annotations

import io
import json
import random
import re
import sqlite3
from pathlib import Path

from kb_manager.evaluation.query_formats import apply_format

DB_PATH = str(Path(__file__).resolve().parent / "data" / "kb_test.db")
OUT_JSON = str(Path(__file__).resolve().parent / "data" / "test_questions.json")
OUT_REPORT = str(Path(__file__).resolve().parent / "data" / "test_questions_report.txt")


# ---------------------------------------------------------------------------
# Persian tokenizer + similarity
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    ["از", "در", "به", "و", "با", "برای", "که", "این", "آن", "را", "شد", "است", "هستند", "بودند", "می", "باشد", "می", "شود", "می", "گردد", "می", "کند", "هر", "دو", "آیا", "یا", "اگر", "ولی", "تا", "باشد", "بر", "اساس", "طبق", "طریق", "نیز", "همچنین", "درباره", "بین", "توسط", "مانند", "مثل", "طی", "خود", "کنید", "گردد", "باید", "یک", "یکی", "شود", "نیست", "نمی", "کنند", "می", "شوند", "می", "باشند", "لطفا", "لطفاً", "آیا", "چه", "چیزی", "خودمان", "چگونه"]
)

# Per-format target similarity bands for the difficulty guard.
_FORMAT_BANDS = {
    "verbatim": (0.8, 1.0),
    "paraphrase": (0.45, 0.79),
    "typo": (0.4, 0.85),
    "reworded": (0.05, 0.44),
    "keyword_only": (0.0, 0.3),
    "conversational": (0.05, 0.45),
}


def tokenize(text: str) -> list[str]:
    """Simple Persian-aware word tokens (keeps key terms)."""
    text = text.replace("\u064a", "\u06cc").replace("\u0643", "\u06a9")
    text = text.replace("\u200c", " ")  # ZWNJ -> space
    words = re.findall(r"[\u0600-\u06FF]+", text)
    # Exclude pure punctuation / diacritic-only tokens (؟ ، ؛ ً ٍ َ ُ ِ etc.)
    return [
        w for w in words
        if any(ch in "ابتثجحخدذرزسشصضطظعغفقكلمنهويءأآئؤة" for ch in w)
    ]


def jaccard(a: list[str], b: list[str]) -> float:
    """Token-set Jaccard similarity."""
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


# ---------------------------------------------------------------------------
# Persian synonym / paraphrase tables (domain-aware)
# ---------------------------------------------------------------------------

_SYNONYMS: dict[str, list[str]] = {
    "قرارداد": ["عقد", "قرارداد", "توافق‌نامه"],
    "آغاز": ["شروع", "آغاز", "آغازین"],
    "فرآیند": ["روند", "فرآیند", "مشخصه"],
    "ثبت": ["قید", "ثبت", "درج"],
    "نامه": ["مکاتبه", "نامه", "اصطبل"],
    "امضاء": ["امضا", "امضا", "تایید"],
    "ارسال": ["فرستادن", "ارسال", "دریافت"],
    "مدارک": ["اسناد", "مدارک", "مستندات"],
    "فعالیت": ["حوزه کار", "فعالیت", "حوزه فعالیت"],
    "سرویس": ["خدمت", "سرویس", "وب‌سرویس"],
    "دسترسی": ["دسترسی", "ورود", "دریافت مجوز"],
    "تعهدات": ["تعقیب", "تعهدات", "مسئولیت‌ها"],
    "مدیرعامل": ["مدیرعامل", "مدیر اجرایی", "مسئول شرکت"],
    "بررسی": ["بازبینی", "بررسی", "تأیید"],
    "تایید": ["تأیید", "تصدیق", "پذیرش"],
    "پرداخت": ["پرداخت", "تسویه", "وصول"],
    "گزارش": ["خروجی", "گزارش", "گزارش اعتباری"],
    "مشتری": ["کاربر", "مشتری", "متقاضی"],
}

_ASK_PREFIXES = [
    "لطفاً بگویید ",
    "ممکن است توضیح دهید که ",
    "آیا می‌دانید که ",
    "می‌خواهم بدانم ",
    "در این باره بگویید که ",
]

_ASK_SUFFIXES = [
    "؟",
    " . لطفاً دقیق توضیح دهید.",
    " . اگر جزئیات دارید در میان بگذارید.",
]

_FILLERS = [
    "در این بخش",
    "طبق متن",
    "بر اساس اسناد",
    "برای من",
    "در عمل",
    "همان‌طور که اشاره شد",
]


# ---------------------------------------------------------------------------
# Transformation helpers
# ---------------------------------------------------------------------------

def _swap_synonyms(text: str, max_swaps: int) -> str:
    words = text.split()
    swapped = 0
    out = []
    for w in words:
        key = w
        repl = _SYNONYMS.get(key, _SYNONYMS.get(key.rstrip("،"), []))
        if repl and swapped < max_swaps:
            # Prefer a synonym that actually differs from the original.
            options = [r for r in repl if r != key and r != key.rstrip("،")]
            if options:
                out.append(random.choice(options))
                swapped += 1
                continue
            out.append(w)
        else:
            out.append(w)
    return " ".join(out)


def _shuffle_middle(text: str, swaps: int) -> str:
    """Move a few non-edge words around while keeping first/last stable."""
    words = text.split()
    if len(words) <= 3:
        return text
    idxs = list(range(1, len(words) - 1))
    if len(idxs) < 2:
        return text
    chosen = random.sample(idxs, min(swaps, len(idxs) - 1))
    seg = [words[i] for i in chosen]
    random.shuffle(seg)
    for i, w in zip(chosen, seg, strict=False):
        words[i] = w
    return " ".join(words)


def _drop_words(text: str, count: int) -> str:
    words = tokenize(text)
    if len(words) <= 3:
        return text
    drop = random.sample(words, min(count, len(words) - 2))
    result = []
    for w in text.split():
        if w in drop:
            continue
        result.append(w)
    return " ".join(result)


# ---------------------------------------------------------------------------
# Level generators
# ---------------------------------------------------------------------------

def level_easy(question: str) -> str:
    """Near-verbatim: minor politeness changes only."""
    return question.rstrip("؟?")


def level_medium(question: str) -> str:
    """Paraphrase: swap a few synonyms, shuffle middle words, rephrase ask."""
    v = _swap_synonyms(question, max_swaps=3)
    v = _shuffle_middle(v, swaps=2)
    v = _swap_synonyms(v, max_swaps=2)
    if random.random() < 0.8:
        v = f"{random.choice(_ASK_PREFIXES)}{v}{random.choice(_ASK_SUFFIXES)}"
    return v


def _drop_ratio(text: str, ratio: float) -> str:
    """Remove a fraction of non-edge content words."""
    words = tokenize(text)
    if len(words) <= 4:
        return text
    drop_n = max(1, int(len(words) * ratio))
    drop = set(random.sample(words[1:-1] if len(words) > 2 else words,
                             min(drop_n, len(words) - 2)))
    result = []
    for w in text.split():
        if w in drop:
            continue
        result.append(w)
    return " ".join(result)


def level_hard(question: str) -> str:
    """Heavy reword: drop ~40% content words, swap synonyms, reorder."""
    v = _drop_ratio(question, ratio=0.4)
    v = _swap_synonyms(v, max_swaps=len(tokenize(v)))
    v = _shuffle_middle(v, swaps=3)
    v = _swap_synonyms(v, max_swaps=2)
    prefix = random.choice(_ASK_PREFIXES)
    filler = random.choice(_FILLERS)
    return f"{prefix}{filler} {v}{random.choice(_ASK_SUFFIXES)}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_FORMATS = ["verbatim", "paraphrase", "reworded", "keyword_only", "typo", "conversational"]


def sample_questions(conn: sqlite3.Connection, count: int) -> list[dict]:
    """Sample QA chunks with their question, keywords, and answer."""
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT id, metadata, content FROM chunks "
        "WHERE chunk_type = 'qa_pair' ORDER BY RANDOM() LIMIT ?",
        (count,),
    ).fetchall()
    questions = []
    for cid, meta_json, content in rows:
        meta = json.loads(meta_json) if meta_json else {}
        fields = meta.get("fields", {}) if isinstance(meta, dict) else {}
        q = fields.get("question", "")
        if not q:
            continue
        # Pull keywords from the chunk content header line if present.
        kw = fields.get("keywords", "")
        if not kw:
            m = re.search(r"کلیدواژه‌ها?[:：]\s*([^\n]+)", content or "")
            if m:
                kw = m.group(1).strip()
        questions.append({"chunk_id": cid, "question": q, "keywords": kw})
    return questions


def main(num_bases: int = 20, seed: int = 42, formats: list[str] | None = None) -> None:
    random.seed(seed)
    out = io.StringIO()

    def p(*a):
        print(*a, file=out)

    fmt_names = formats or list(_FORMATS)
    conn = sqlite3.connect(DB_PATH)
    bases = sample_questions(conn, num_bases)

    dataset: list[dict] = []
    stats: dict[str, list[float]] = {f: [] for f in fmt_names}
    difficulty_map = {
        "verbatim": "easy",
        "paraphrase": "medium",
        "reworded": "hard",
        "keyword_only": "hard",
        "typo": "medium",
        "conversational": "hard",
    }

    for base in bases[:num_bases]:
        gt = base["question"]
        gt_tokens = tokenize(gt)
        p("=" * 70)
        p("GT  : " + gt)

        for fmt in fmt_names:
            variant = apply_format(fmt, gt, base.get("keywords", ""))
            sim = jaccard(tokenize(variant), gt_tokens)

            # Guard per-format similarity band with retries.
            lo, hi = _FORMAT_BANDS[fmt]
            attempt = 0
            while not (lo <= sim <= hi) and attempt < 8:
                variant = apply_format(fmt, gt, base.get("keywords", ""))
                sim = jaccard(tokenize(variant), gt_tokens)
                attempt += 1

            stats[fmt].append(sim)
            p(f"  {fmt:<12} (sim={sim:.2f}) : " + variant)

            dataset.append({
                "query": variant,
                "expected_chunk_ids": [base["chunk_id"]],
                "expected_answer": "",
                "relevance_scores": {base["chunk_id"]: 1.0},
                "category": "factual",
                "difficulty": difficulty_map[fmt],
                "format": fmt,
                "gt": gt,
                "keywords": base.get("keywords", ""),
                "gt_similarity": round(sim, 3),
            })

    conn.close()

    p("\n" + "=" * 70)
    p(f"TOTAL variants: {len(dataset)}  (bases={len(bases)}, formats={len(fmt_names)})")
    for fmt in fmt_names:
        vals = stats[fmt]
        avg = sum(vals) / len(vals) if vals else 0.0
        lo = min(vals) if vals else 0.0
        hi = max(vals) if vals else 0.0
        p(f"  {fmt:<12} sim min={lo:.2f} max={hi:.2f} avg={avg:.2f}  n={len(vals)}")
    p("\nGT_similarity mapping: higher = closer to original wording = easier.")
    p("Formats: verbatim ~ paraphrased ~ reworded ~ keyword-only ~ typo ~ conversational")

    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(out.getvalue())

    print("WROTE", OUT_JSON, "and", OUT_REPORT)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate multi-format test questions.")
    parser.add_argument("--num-bases", type=int, default=20, help="GT questions to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--formats",
        nargs="*",
        choices=_FORMATS,
        default=None,
        help="Subset of formats (default: all)",
    )
    args = parser.parse_args()
    main(num_bases=args.num_bases, seed=args.seed, formats=args.formats)
