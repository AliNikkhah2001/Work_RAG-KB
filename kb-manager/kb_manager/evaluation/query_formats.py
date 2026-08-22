"""Query format transformations for retrieval testing.

Each format rewrites a ground-truth question differently so benchmarks can
measure robustness of the retrieval system across phrasing styles:

    verbatim       the question unchanged (trivial baseline)
    paraphrase     synonym swaps + reordering (moderate)
    reworded       keyword drops + heavy reorder (hard)
    keyword_only   just the domain keywords (no question frame)
    typo           realistic Persian spelling errors
    conversational informal spoken style (e.g. "می‌خوام", "میشه")
"""

from __future__ import annotations

import random
import re

_FORMAT_NAMES = (
    "verbatim",
    "paraphrase",
    "reworded",
    "keyword_only",
    "typo",
    "conversational",
)

_SYNONYMS: dict[str, list[str]] = {
    "قرارداد": ["عقد", "قرارداد", "توافق‌نامه"],
    "آغاز": ["شروع", "آغاز"],
    "فرآیند": ["روند", "فرآیند"],
    "ثبت": ["قید", "ثبت", "درج"],
    "نامه": ["مکاتبه", "نامه"],
    "امضا": ["امضا", "تایید"],
    "ارسال": ["فرستادن", "ارسال"],
    "مدارک": ["اسناد", "مدارک", "مستندات"],
    "سرویس": ["خدمت", "سرویس", "وب‌سرویس"],
    "دسترسی": ["دسترسی", "ورود"],
    "بررسی": ["بازبینی", "بررسی", "تأیید"],
    "پرداخت": ["پرداخت", "تسویه"],
    "گزارش": ["خروجی", "گزارش"],
    "مشتری": ["کاربر", "مشتری", "متقاضی"],
    "اثرگذار": ["تاثیرگذار", "اثرگذار", "موثر"],
    "لازم": ["الزامی", "لازم", "ضروری"],
    "تاخیر": ["دیرکرد", "تاخیر", "تأخیر"],
}

_ASK_PREFIXES = [
    "لطفاً بگویید ",
    "ممکن است توضیح دهید که ",
    "می‌خواهم بدانم ",
    "در این باره بگویید که ",
]
_ASK_SUFFIXES = ["؟", " . لطفاً دقیق توضیح دهید.", " . اگر جزئیات دارید در میان بگذارید."]
_FILLERS = ["در این بخش", "طبق متن", "بر اساس اسناد", "در عمل", "همان‌طور که اشاره شد"]

_TYPO_MAP = [
    ("دسترسی", "دسترسى"),
    ("می‌شود", "میشود"),
    ("می‌کنند", "میکنند"),
    ("باید", "باید"),
    ("دریافت", "دریافت"),
    ("قرارداد", "قرارداد"),
    ("اعتباری", "اعتبارى"),
    ("گزارش", "گزراش"),
    ("تسهیلات", "تسهیلات"),
    ("تسویه", "تسویه"),
    ("کاربر", "کاربر"),
    ("سازمان", "سازمان"),
]


def normalize_persian(text: str) -> str:
    """Normalise Arabic variant letters and ZWNJ for comparison."""
    return (
        text.replace("\u064a", "\u06cc")
        .replace("\u0643", "\u06a9")
        .replace("\u0671", "\u0627")
        .replace("\u200c", " ")
        .strip()
    )


# ---------------------------------------------------------------------------
# Individual format generators
# ---------------------------------------------------------------------------

def _swap_synonyms(text: str, max_swaps: int) -> str:
    words = text.split()
    swapped = 0
    out: list[str] = []
    for w in words:
        key = w.rstrip("،")
        repl = _SYNONYMS.get(key)
        if repl and swapped < max_swaps:
            options = [r for r in repl if r != key]
            if options:
                suffix = w[len(key):]
                out.append(random.choice(options) + suffix)
                swapped += 1
                continue
        out.append(w)
    return " ".join(out)


def _shuffle_middle(text: str, swaps: int) -> str:
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


def _drop_ratio(text: str, ratio: float) -> str:
    """Remove a fraction of non-edge content words."""
    words = text.split()
    if len(words) <= 4:
        return text
    drop_n = max(1, int(len(words) * ratio))
    drop = set(
        random.sample(
            words[1:-1] if len(words) > 2 else words,
            min(drop_n, len(words) - 2),
        )
    )
    return " ".join(w for w in words if w not in drop)


def format_verbatim(question: str, keywords: str = "") -> str:
    return question.rstrip("؟")


def format_paraphrase(question: str, keywords: str = "") -> str:
    v = _swap_synonyms(question, max_swaps=3)
    v = _shuffle_middle(v, swaps=2)
    v = _swap_synonyms(v, max_swaps=2)
    if random.random() < 0.8:
        v = f"{random.choice(_ASK_PREFIXES)}{v}{random.choice(_ASK_SUFFIXES)}"
    return v


def format_reworded(question: str, keywords: str = "") -> str:
    v = _drop_ratio(question, ratio=0.4)
    v = _swap_synonyms(v, max_swaps=len(v.split()))
    v = _shuffle_middle(v, swaps=3)
    v = _swap_synonyms(v, max_swaps=2)
    prefix = random.choice(_ASK_PREFIXES)
    filler = random.choice(_FILLERS)
    return f"{prefix}{filler} {v}{random.choice(_ASK_SUFFIXES)}"


def format_keyword_only(question: str, keywords: str = "") -> str:
    """Query built from the chunk's keyword field, or extracted terms."""
    if keywords:
        # Sanitize junk metadata that leaks from Excel: "مدل: حقیقی و حقوقی…" suffix
        # Example raw: "بروزرسانی، بازپرداخت، وام، گزارش اعتباری مدل: حقیقی و حقوقی…"
        keywords = re.sub(r"\s*مدل\s*:\s*[^\n,،؛]+", "", keywords)
        keywords = re.sub(r"[…\u2026]+", "", keywords)
        keywords = keywords.replace("\u200c", " ")
        parts = [
            k.strip("[]\"' …\u2026")
            for k in re.split(r"[،,؛;\n]+", keywords)
            if k.strip("[]\"' …\u2026")
        ]
        # Filter remaining junk: isolated model words or colon remnants
        junk_exact = {"حقیقی", "حقوقی", "و", "مدل"}
        filtered: list[str] = []
        for p in parts:
            p = p.strip()
            if not p or len(p) < 2:
                continue
            if ":" in p or "مدل" in p:
                continue
            if p in junk_exact:
                continue
            if set(p) <= {"…", ".", " "}:
                continue
            filtered.append(p)
        if filtered:
            return " ".join(filtered[:6])
    # Fallback: reuse high-content words from the question.
    words = [w for w in re.findall(r"[\u0600-\u06FF]+", question) if len(w) > 3]
    return " ".join(words[:6])


def format_typo(question: str, keywords: str = "") -> str:
    """Introduce realistic Persian typographical errors."""
    result = question
    for src, dst in _TYPO_MAP:
        if src in result and random.random() < 0.5:
            result = result.replace(src, dst, 1)
    # Occasionally drop a ZWNJ.
    if "\u200c" in result and random.random() < 0.5:
        result = result.replace("\u200c", "", 1)
    return result


def format_conversational(question: str, keywords: str = "") -> str:
    """Spoken / informal rewording."""
    v = question.rstrip("؟")
    v = v.replace("می‌شود", "میشه").replace("می‌شوند", "میشن")
    v = v.replace("می‌توانم", "می‌تونم").replace("می‌توان", "می‌شه")
    v = v.replace("به‌طور", "به صورت").replace("اینکه", "که")
    prefixes = ["ببخشید، ", "سلام، ", "راستی، ", "یه سوال داشتم، "]
    return f"{random.choice(prefixes)}{v}{random.choice(_ASK_SUFFIXES)}"


_FORMAT_FUNCS = {
    "verbatim": format_verbatim,
    "paraphrase": format_paraphrase,
    "reworded": format_reworded,
    "keyword_only": format_keyword_only,
    "typo": format_typo,
    "conversational": format_conversational,
}


def apply_format(
    fmt: str,
    question: str,
    keywords: str = "",
    rng: random.Random | None = None,
) -> str:
    """Apply a named format to a ground-truth question."""
    if rng is not None:
        # Temporarily seed module random for deterministic regeneration.
        state = random.getstate()
        random.setstate(rng.getstate())
        try:
            return _FORMAT_FUNCS.get(fmt, format_verbatim)(question, keywords)
        finally:
            random.setstate(state)
    return _FORMAT_FUNCS.get(fmt, format_verbatim)(question, keywords)


def format_list() -> tuple[str, ...]:
    return _FORMAT_NAMES


def default_difficulty(fmt: str) -> str:
    """Map a format to a coarse difficulty label for grouping."""
    return {
        "verbatim": "easy",
        "paraphrase": "medium",
        "typo": "medium",
        "reworded": "hard",
        "keyword_only": "hard",
        "conversational": "hard",
    }.get(fmt, "medium")
