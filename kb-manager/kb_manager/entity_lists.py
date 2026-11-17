"""Supplementary EntityList helpers — score tables + list catalogs.

Score Excel sheets use a merged-cell style color column (filled only on the
first row of each band). We forward-fill color and emit structured band rows
so the LLM can answer number / letter / color without hallucination.

Each score table is tagged with subspace_ids so retrieval can inject only the
model that matches the active sticky subspace (cheque vs person vs legal).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

SCORE_TABLE_SPECS: list[dict[str, Any]] = [
    {
        "match": ("جدول امتیاز اعتبارسنجی چک", "امتیاز اعتبارسنجی چک", "امتیاز چک"),
        "list_name": "جدول امتیاز اعتبارسنجی چک",
        "model": "cheque",
        "subspace_ids": ["cheque"],
        "score_scale": "1-100",
        "score_min": 1,
        "score_max": 100,
        "description": (
            "جدول امتیاز و رتبه و رنگ مدل اعتبارسنجی چک (اشخاص حقیقی / گزارش چک). "
            "مقیاس امتیاز ۱ تا ۱۰۰. شامل رنگ (آبی سبز زرد نارنجی قرمز)، "
            "رتبه حرفی (A1 تا E3)، حد بالا/پایین امتیاز و ریسک نکول. "
            "پرسش‌های امتیاز چک، رنگ رتبه چک، گرید A1 چک."
        ),
    },
    {
        "match": ("جدول امتیاز اعتباری اشخاص حقیقی", "امتیاز اعتباری اشخاص حقیقی"),
        "list_name": "جدول امتیاز اعتباری اشخاص حقیقی",
        "model": "person",
        "subspace_ids": [
            "facilities_person",
            "credit_ladder",
            "guarantor",
            "customers_person",
        ],
        "score_scale": "250-900",
        "score_min": 250,
        "score_max": 900,
        "description": (
            "جدول امتیاز و رتبه و رنگ مدل اعتبارسنجی تسهیلات اشخاص حقیقی. "
            "مقیاس امتیاز حدود ۲۵۰ تا ۹۰۰. شامل رنگ، رتبه حرفی A1 تا E3، "
            "حد بالا/پایین امتیاز و ریسک نکول. "
            "پرسش‌های امتیاز حقیقی، رنگ رتبه حقیقی، گرید A2 حقیقی، نردبان/کفیل."
        ),
    },
    {
        "match": ("جدول امتیاز اعتباری اشخاص حقوقی", "امتیاز اعتباری اشخاص حقوقی"),
        "list_name": "جدول امتیاز اعتباری اشخاص حقوقی",
        "model": "legal",
        "subspace_ids": [
            "facilities_legal",
            "supplier_legal",
            "customers_legal",
        ],
        "score_scale": "250-900",
        "score_min": 250,
        "score_max": 900,
        "description": (
            "جدول امتیاز و رتبه و رنگ مدل اعتبارسنجی تسهیلات اشخاص حقوقی / شرکت‌ها. "
            "مقیاس امتیاز حدود ۲۵۰ تا ۹۰۰. شامل رنگ، رتبه حرفی A1 تا E3، "
            "حد بالا/پایین امتیاز و ریسک نکول. "
            "پرسش‌های امتیاز حقوقی، رنگ رتبه شرکت، گرید B1 حقوقی."
        ),
    },
]

LIST_CATALOG_SPECS: list[dict[str, Any]] = [
    {
        "match": ("لیست و عناوین وام ها", "لیست و عناوین وام‌ها"),
        "list_name": "لیست و عناوین وام ها",
        "kind": "loan_catalog",
        "subspace_ids": ["facilities_person", "facilities_legal", "general"],
        "description": (
            "فهرست نام و نوع وام‌ها و تسهیلات (ازدواج، خرید اقساطی، BNPL و …). "
            "برای پرسش درباره عنوان وام یا نوع تسهیلات."
        ),
    },
    {
        "match": ("سهامداران",),
        "list_name": "سهامداران",
        "kind": "shareholders",
        "subspace_ids": ["general", "facilities_legal"],
        "description": (
            "فهرست سهامداران شرکت اعتبارسنجی ایران (بانک‌ها، بیمه، لیزینگ) "
            "با نوع سهامدار و حوزه فعالیت."
        ),
    },
    {
        "match": ("تامین‌کنندگان دیتا", "تامین کنندگان دیتا", "تامین‌کننده"),
        "list_name": "تامین‌کنندگان دیتا",
        "kind": "data_providers",
        "subspace_ids": ["supplier_legal", "general"],
        "description": (
            "فهرست تامین‌کنندگان داده اعتبارسنجی (بانک، لندتک، لیزینگ، صندوق) "
            "و کانال‌های نئوبانک/موبایل‌بانک."
        ),
    },
    {
        "match": ("استفاده کنندگان گزارش", "استفاده‌کنندگان گزارش"),
        "list_name": "استفاده کنندگان گزارش",
        "kind": "report_consumers",
        "subspace_ids": ["customers_person", "customers_legal", "general"],
        "description": (
            "فهرست استفاده‌کنندگان/خریداران گزارش اعتبارسنجی (لندتک، ریسلر و …) "
            "با نوع قرارداد."
        ),
    },
]

# Reason-code catalogs — one workbook per (domain, model). ``model`` and
# ``hierarchy`` are emitted into ``_meta`` so a code stays unambiguous even
# when the same code exists in two models (AGE2 main/substitute, NGL1-3
# main/legal). Matching is by exact stem (see ``_match_spec``), so the
# reason-code workbook no longer collides with the score-table spec.
REASON_CODE_SPECS: list[dict[str, Any]] = [
    {
        "match": (
            "توضیحات تشریحی دلایل کاهش امتیاز چک و روش بهبود آن",
            "دلایل کاهش امتیاز چک",
        ),
        "list_name": "کدهای دلیل — چک",
        "kind": "reason_code_catalog",
        "domain": "cheque",
        "model": "cheque",
        "model_label": "مدل چک",
        "subspace_ids": ["cheque"],
        "hierarchy": ["کدهای دلیل", "گزارش اعتبارسنجی چک"],
        "description": (
            "کدهای دلیل کاهش امتیاز چک (گزارش اعتبارسنجی چک) با متن دلیل، "
            "توضیح کوتاه/کامل و روش بهبود. برای پرسش «کد دلیل X در چک»."
        ),
    },
    {
        "match": (
            "توضیحات تشریحی دلایل کاهش امتیاز اعتباری مدل اصلی و روش بهبود آن",
            "دلایل کاهش امتیاز اعتباری مدل اصلی",
        ),
        "list_name": "کدهای دلیل — اشخاص حقیقی (مدل اصلی)",
        "kind": "reason_code_catalog",
        "domain": "credit",
        "model": "person_main",
        "model_label": "مدل اصلی اشخاص حقیقی",
        "subspace_ids": ["facilities_person"],
        "hierarchy": ["کدهای دلیل", "اعتبارسنجی تسهیلات", "اشخاص حقیقی", "مدل اصلی"],
        "description": (
            "کدهای دلیل کاهش امتیاز اعتباری اشخاص حقیقی — مدل اصلی، با متن دلیل، "
            "توضیح کوتاه/کامل و روش بهبود."
        ),
    },
    {
        "match": (
            "توضیحات تشریحی دلایل کاهش امتیاز اعتباری مدل جایگزین و روش بهبود آن",
            "دلایل کاهش امتیاز اعتباری مدل جایگزین",
        ),
        "list_name": "کدهای دلیل — اشخاص حقیقی (مدل جایگزین)",
        "kind": "reason_code_catalog",
        "domain": "credit",
        "model": "person_substitute",
        "model_label": "مدل جایگزین اشخاص حقیقی",
        "subspace_ids": ["facilities_person"],
        "hierarchy": ["کدهای دلیل", "اعتبارسنجی تسهیلات", "اشخاص حقیقی", "مدل جایگزین"],
        "description": (
            "کدهای دلیل کاهش امتیاز اعتباری اشخاص حقیقی — مدل جایگزین، با متن دلیل، "
            "توضیح کوتاه/کامل و روش بهبود."
        ),
    },
    {
        "match": (
            "توضیحات تشریحی دلایل کاهش امتیاز اعتباری شرکتها",
            "توضیحات تشریحی دلایل کاهش امتیاز اعتباری شرکت‌ها",
            "دلایل کاهش امتیاز اعتباری شرکتها",
        ),
        "list_name": "کدهای دلیل — اشخاص حقوقی",
        "kind": "reason_code_catalog",
        "domain": "credit",
        "model": "legal",
        "model_label": "مدل اشخاص حقوقی",
        "subspace_ids": ["facilities_legal"],
        "hierarchy": ["کدهای دلیل", "اعتبارسنجی تسهیلات", "اشخاص حقوقی"],
        "description": (
            "کدهای دلیل کاهش امتیاز اعتباری اشخاص حقوقی/شرکتها، با متن دلیل، "
            "توضیح کوتاه/کامل و روش بهبود."
        ),
    },
]

# Reason-code records that appear in more than one model. Callers must ask
# the user which model before answering instead of guessing.
AMBIGUOUS_REASON_CODES: dict[str, list[str]] = {
    "AGE2": ["person_main", "person_substitute"],
    "NGL1": ["person_main", "legal"],
    "NGL2": ["person_main", "legal"],
    "NGL3": ["person_main", "legal"],
}

# Canonical reason-code field names, in emission order (Persian labels added
# when the sheet uses a different header, e.g. «Keyword»/«Improvement Suggestion»).
_REASON_FIELDS: list[str] = [
    "reason_code",
    "reason_text",
    "brief_explanation",
    "detailed_explanation",
    "improvement_suggestions",
    "keywords",
]

_REASON_FIELD_ALIASES: dict[str, str] = {
    "reasoncode": "reason_code",
    "reasontext": "reason_text",
    "briefexplanation": "brief_explanation",
    "detailedexplanation": "detailed_explanation",
    "improvementsuggestions": "improvement_suggestions",
    "improvementsuggestion": "improvement_suggestions",
    "keyword": "keywords",
    "keywords": "keywords",
}

_GRADE_RE = re.compile(r"^[A-Ea-e][1-3]$")
_COLOR_ALIASES = {
    "آبی": "آبی",
    "سبز": "سبز",
    "زرد": "زرد",
    "نارنجی": "نارنجی",
    "قرمز": "قرمز",
    "blue": "آبی",
    "green": "سبز",
    "yellow": "زرد",
    "orange": "نارنجی",
    "red": "قرمز",
}
_RISK_TO_COLOR = {
    "ممتاز": "آبی",
    "خوب": "سبز",
    "مناسب": "زرد",
    "ضعیف": "نارنجی",
    "بحرانی": "قرمز",
    "ریسک خیلی پایین": "آبی",
    "ریسک پایین": "سبز",
    "ریسک متوسط": "زرد",
    "ریسک بالا": "نارنجی",
    "ریسک خیلی بالا": "قرمز",
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _match_spec(file_path: str, specs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Match a file against catalog specs by exact (normalized) stem.

    Stem equality — NOT substring — is deliberate. A substring match on
    «امتیاز چک» made the reason-code workbook «… دلایل کاهش امتیاز چک …»
    masquerade as the cheque score table (same ``list_name``), which silently
    overwrote the real score table. Exact stem matching keeps each workbook
    bound to exactly one spec.
    """
    stem = _norm(Path(file_path).stem)
    if not stem:
        return None
    for spec in specs:
        for m in spec.get("match") or ():
            if stem == _norm(m):
                return spec
    return None


def match_score_spec(file_path: str) -> dict[str, Any] | None:
    return _match_spec(file_path, SCORE_TABLE_SPECS)


def match_list_spec(file_path: str) -> dict[str, Any] | None:
    return _match_spec(file_path, LIST_CATALOG_SPECS)


def match_reason_spec(file_path: str) -> dict[str, Any] | None:
    return _match_spec(file_path, REASON_CODE_SPECS)


def _cell_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return str(v).strip()


def _to_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _grade_index(cells: list[Any]) -> int | None:
    for i, c in enumerate(cells[:6]):
        if _GRADE_RE.match(_cell_str(c)):
            return i
    return None


def normalize_score_rows(raw_rows: list[Any]) -> list[dict[str, Any]]:
    """Forward-fill color; emit structured band dicts from score Excel rows.

    Supports:
      A) Raw openpyxl: [color?, risk, grade, high, low, pd_high, pd_low] (+ header)
      B) xlsx_parser:  [risk, grade, high, low, pd_high, pd_low] (header/color stripped)
    """
    if not raw_rows:
        return []

    header_idx = -1
    for i, row in enumerate(raw_rows[:5]):
        cells = [_cell_str(c) for c in (row or [])]
        joined = " ".join(cells)
        if ("رتبه" in joined or "امتیاز" in joined) and not any(
            _GRADE_RE.match(c) for c in cells
        ):
            header_idx = i
            break
    data_rows = raw_rows[header_idx + 1 :] if header_idx >= 0 else raw_rows

    bands: list[dict[str, Any]] = []
    last_color = ""
    for row in data_rows:
        if not row:
            continue
        cells = list(row)
        while cells and cells[-1] is None and len(cells) > 7:
            cells.pop()
        if len(cells) < 4:
            continue

        gidx = _grade_index(cells)
        if gidx is None:
            continue
        grade = _cell_str(cells[gidx]).upper()
        score_high = _to_float(cells[gidx + 1]) if len(cells) > gidx + 1 else None
        score_low = _to_float(cells[gidx + 2]) if len(cells) > gidx + 2 else None
        pd_high = _to_float(cells[gidx + 3]) if len(cells) > gidx + 3 else None
        pd_low = _to_float(cells[gidx + 4]) if len(cells) > gidx + 4 else None
        if score_high is None or score_low is None:
            continue

        before = [_cell_str(c) for c in cells[:gidx]]
        color_raw = ""
        risk = ""
        if len(before) >= 2:
            color_raw, risk = before[0], before[1]
        elif len(before) == 1:
            only = before[0]
            if only in _COLOR_ALIASES:
                color_raw = only
            else:
                risk = only

        if color_raw in _COLOR_ALIASES:
            last_color = _COLOR_ALIASES[color_raw]
        elif color_raw and color_raw not in _RISK_TO_COLOR and "ریسک" not in color_raw:
            last_color = color_raw
        if risk in _RISK_TO_COLOR:
            last_color = _RISK_TO_COLOR[risk]

        bands.append(
            {
                "color": last_color or None,
                "risk_label": risk or None,
                "grade": grade,
                "score_high": score_high,
                "score_low": score_low,
                "pd_high": pd_high,
                "pd_low": pd_low,
            }
        )
    return bands


def _build_color_groups(bands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse A1–A3 / B1–B3 / … into color buckets for listing answers."""
    groups: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    for b in bands:
        color = b.get("color")
        letter = (b.get("grade") or "?")[0]
        if cur and cur.get("color") == color and cur.get("letter") == letter:
            cur["grades"].append(b.get("grade"))
            cur["score_high"] = max(cur["score_high"], b["score_high"])
            cur["score_low"] = min(cur["score_low"], b["score_low"])
            cur["bands"].append(b)
        else:
            if cur:
                groups.append(cur)
            cur = {
                "color": color,
                "letter": letter,
                "risk_label": b.get("risk_label"),
                "grades": [b.get("grade")],
                "score_high": b["score_high"],
                "score_low": b["score_low"],
                "bands": [b],
            }
    if cur:
        groups.append(cur)
    return groups


def _full_table_text(bands: list[dict[str, Any]], *, model: str, scale: str) -> str:
    """Human-readable dump of the whole score table for list-all questions."""
    lines = [
        f"جدول کامل امتیاز مدل {model} (مقیاس {scale})",
        "ساختار: رنگ‌ها → گروه حرفی A/B/C/D/E → زیررتبه‌های 1/2/3",
        "رنگ | میزان ریسک | رتبه | حد بالای امتیاز | حد پایین امتیاز | حد بالای ریسک نکول | حد پایین ریسک نکول",
    ]
    for b in bands:
        lines.append(
            f"{b.get('color')} | {b.get('risk_label')} | {b.get('grade')} | "
            f"{b.get('score_high')} | {b.get('score_low')} | "
            f"{b.get('pd_high')} | {b.get('pd_low')}"
        )
    groups = _build_color_groups(bands)
    lines.append("خلاصه گروه‌های رنگی:")
    for g in groups:
        lines.append(
            f"- رنگ {g['color']}: گروه {g['letter']} "
            f"({', '.join(g['grades'])}) — {g['risk_label']} — "
            f"بازه امتیاز {g['score_low']} تا {g['score_high']}"
        )
    return "\n".join(lines)


def build_score_entity(
    file_path: str,
    sheets: list[dict[str, Any]] | None,
    *,
    title: str | None = None,
) -> dict[str, Any] | None:
    """Build EntityList payload for a score table file."""
    spec = match_score_spec(file_path)
    if not spec:
        return None
    raw_rows: list[Any] = []
    sheet_name = "Sheet1"
    if sheets:
        for sh in sheets:
            rows = sh.get("rows") or []
            if rows:
                raw_rows = rows
                sheet_name = sh.get("name") or sheet_name
                break
    bands = normalize_score_rows(raw_rows)
    if not bands:
        return None
    last = bands[0].get("color")
    for b in bands:
        if b.get("color"):
            last = b["color"]
        elif last:
            b["color"] = last

    color_groups = _build_color_groups(bands)
    full_text = _full_table_text(
        bands, model=spec["model"], scale=spec["score_scale"]
    )
    desc = (
        f"{spec['description']} "
        f"جدول کامل شامل ۱۵ رتبه A1 تا E3 در ۵ رنگ "
        f"(آبی={color_groups[0]['letter'] if color_groups else 'A'}, "
        f"سبز، زرد، نارنجی، قرمز). "
        f"برای پرسش فهرست رنگ‌ها، فهرست رتبه‌ها، یا جدول کامل امتیاز از full_table_text استفاده شود."
    )

    content = {
        "_meta": {
            "kind": "score_table",
            "model": spec["model"],
            "subspace_ids": list(spec["subspace_ids"]),
            "score_scale": spec["score_scale"],
            "score_min": spec["score_min"],
            "score_max": spec["score_max"],
            "source_file": Path(file_path).name,
            "structure": (
                "score number ranges → letter grades A–E "
                "(each split into 1/2/3) → color per letter group"
            ),
        },
        "bands": bands,
        "color_groups": [
            {
                "رنگ": g["color"],
                "گروه حرفی": g["letter"],
                "رتبه‌ها": g["grades"],
                "میزان ریسک": g["risk_label"],
                "حد بالای امتیاز گروه": g["score_high"],
                "حد پایین امتیاز گروه": g["score_low"],
            }
            for g in color_groups
        ],
        "full_table_text": full_text,
        sheet_name: [
            {
                "رنگ": b.get("color"),
                "میزان ریسک": b.get("risk_label"),
                "رتبه اعتباری": b.get("grade"),
                "حد بالای امتیاز": b.get("score_high"),
                "حد پایین امتیاز": b.get("score_low"),
                "حد بالای ریسک نکول": b.get("pd_high"),
                "حد پایین ریسک نکول": b.get("pd_low"),
            }
            for b in bands
        ],
    }
    return {
        "list_name": spec["list_name"],
        "description": desc,
        "content_json": content,
        "subspace_ids": list(spec["subspace_ids"]),
        "kind": "score_table",
        "model": spec["model"],
    }


def _read_catalog_matrix(file_path: str) -> tuple[str, list[str], list[list[Any]]]:
    """Read catalog xlsx via openpyxl (keeps 'Column 1' that xlsx_parser drops)."""
    try:
        from openpyxl import load_workbook
    except ImportError:
        return "Sheet1", [], []
    wb = load_workbook(file_path, data_only=True)
    ws = wb.active
    sheet_name = ws.title or "Sheet1"
    # Find last used col among first 12
    max_c = min(ws.max_column or 1, 12)
    header_vals = [_cell_str(ws.cell(1, c).value) for c in range(1, max_c + 1)]
    # Trim trailing empty header cells
    while header_vals and not header_vals[-1]:
        header_vals.pop()
    if not header_vals:
        return sheet_name, [], []
    ncols = len(header_vals)
    # Rename junk Excel headers to useful Persian labels by position/kind later
    rows: list[list[Any]] = []
    for r in range(2, (ws.max_row or 1) + 1):
        vals = [ws.cell(r, c).value for c in range(1, ncols + 1)]
        if not any(v is not None and str(v).strip() for v in vals):
            continue
        rows.append(vals)
    return sheet_name, header_vals, rows


def _catalog_field_names(kind: str, headers: list[str]) -> list[str]:
    """Map raw Excel headers to stable Persian field names."""
    defaults = {
        "shareholders": ["نام سهامدار", "نوع سهامدار", "حوزه فعالیت"],
        "loan_catalog": ["نام وام", "نوع وام"],
        "report_consumers": ["نام شرکت", "نوع کانال", "نوع قرارداد"],
        "data_providers": [
            "نام تامین کننده",
            "نوع",
            "نئوبانک",
            "نئوبانک ۲",
            "موبایل بانک",
            "موبایل بانک ۲",
        ],
    }
    preferred = defaults.get(kind) or []
    out: list[str] = []
    for i, h in enumerate(headers):
        hn = _norm(h)
        if not hn or hn.lower().startswith("column") or hn in ("None",):
            out.append(preferred[i] if i < len(preferred) else f"ستون{i+1}")
        else:
            out.append(hn)
    # Ensure preferred length if headers were short
    if preferred and len(out) < len(preferred) and len(out) == len(headers):
        pass
    return out


def _catalog_records(
    field_names: list[str], rows: list[list[Any]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        rec: dict[str, Any] = {}
        empty = True
        for i, name in enumerate(field_names):
            raw = row[i] if i < len(row) else None
            val = _cell_str(raw)
            if val:
                empty = False
            rec[name] = val or None
        if not empty:
            records.append(rec)
    return records


def _catalog_groups(
    records: list[dict[str, Any]], group_field: str
) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for rec in records:
        key = rec.get(group_field) or "نامشخص"
        counts[str(key)] = counts.get(str(key), 0) + 1
    return [
        {"گروه": k, "تعداد": n}
        for k, n in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    ]


def _catalog_full_text(
    *,
    list_name: str,
    kind: str,
    field_names: list[str],
    records: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    group_field: str | None,
) -> str:
    lines = [
        f"فهرست کامل «{list_name}» ({len(records)} مورد)",
        "ستون‌ها: " + " | ".join(field_names),
    ]
    if groups and group_field:
        lines.append(f"جمع‌بندی بر اساس {group_field}:")
        for g in groups:
            lines.append(f"- {g['گروه']}: {g['تعداد']} مورد")
    lines.append("ردیف‌ها:")
    # Cap dump length for huge loan lists but keep all names (needed for lookup)
    for i, rec in enumerate(records, 1):
        parts = [str(rec.get(f) or "") for f in field_names]
        lines.append(f"{i}. " + " | ".join(parts))
    return "\n".join(lines)


def build_catalog_entity(
    file_path: str,
    sheets: list[dict[str, Any]] | None,
    *,
    title: str | None = None,
) -> dict[str, Any] | None:
    """Build EntityList payload for non-score catalog xlsx under لیست/.

    Prefers a direct openpyxl read so useful columns named ``Column 1``
    (excluded by the general xlsx_parser) are preserved for report consumers.
    """
    spec = match_list_spec(file_path)
    kind = (spec or {}).get("kind") or "catalog"
    sheet_name, headers, raw_rows = _read_catalog_matrix(file_path)

    # Fallback to pipeline-parsed sheets when openpyxl path empty
    if not raw_rows and sheets:
        for sheet in sheets:
            rows = sheet.get("rows") or []
            cleaned = []
            for row in rows:
                if isinstance(row, (list, tuple)) and any(
                    c is not None and str(c).strip() for c in row
                ):
                    cleaned.append(list(row))
            if not cleaned:
                continue
            sheet_name = sheet.get("name") or sheet_name
            # Heuristic: if first row looks like header (contains نام/نوع), split it
            first = [_cell_str(c) for c in cleaned[0]]
            if any(k in " ".join(first) for k in ("نام", "نوع", "حوزه")):
                headers = first
                raw_rows = cleaned[1:]
            else:
                headers = [f"ستون{i+1}" for i in range(len(first))]
                raw_rows = cleaned
            break

    if not raw_rows:
        return None

    field_names = _catalog_field_names(kind, headers or [])
    if not field_names:
        return None
    records = _catalog_records(field_names, raw_rows)
    if not records:
        return None

    group_field = {
        "shareholders": "نوع سهامدار",
        "loan_catalog": "نوع وام",
        "report_consumers": "نوع کانال",
        "data_providers": "نوع",
    }.get(kind)
    groups = _catalog_groups(records, group_field) if group_field else []
    list_name = (spec or {}).get("list_name") or (title or Path(file_path).stem)
    base_desc = (spec or {}).get("description") or f"این لیست حاوی اطلاعات {list_name} است."
    group_summary = ""
    if groups:
        top = "، ".join(f"{g['گروه']} ({g['تعداد']})" for g in groups[:8])
        group_summary = f" تعداد کل {len(records)}. گروه‌ها: {top}."
    description = f"{base_desc}{group_summary} برای فهرست کامل یا جستجوی نام از records و full_list_text استفاده شود."
    subspace_ids = list((spec or {}).get("subspace_ids") or [])
    full_text = _catalog_full_text(
        list_name=list_name,
        kind=kind,
        field_names=field_names,
        records=records,
        groups=groups,
        group_field=group_field,
    )
    content_json: dict[str, Any] = {
        "_meta": {
            "kind": kind,
            "subspace_ids": subspace_ids,
            "source_file": Path(file_path).name,
            "fields": field_names,
            "count": len(records),
        },
        "records": records,
        "groups": groups,
        "full_list_text": full_text,
        # Keep a compact sheet view for older consumers
        sheet_name: records,
    }
    return {
        "list_name": list_name,
        "description": description,
        "content_json": content_json,
        "subspace_ids": subspace_ids,
        "kind": kind,
    }


def _reason_field_name(header: Any) -> str | None:
    """Map a raw reason-code header (EN or FA) to a canonical field name."""
    h = _norm(_cell_str(header)).lower().replace(" ", "").replace("_", "")
    if not h:
        return None
    return _REASON_FIELD_ALIASES.get(h)


def _build_reason_code_records(
    headers: list[Any], raw_rows: list[list[Any]]
) -> list[dict[str, Any]]:
    """Project reason-code rows onto the canonical field set."""
    idx_to_field: dict[int, str] = {}
    for i, h in enumerate(headers):
        f = _reason_field_name(h)
        if f and f not in idx_to_field.values():
            idx_to_field[i] = f
    # Positional fallback when headers are missing/junk (schema order is stable).
    if "reason_code" not in idx_to_field.values() and raw_rows:
        for i in range(min(len(_REASON_FIELDS), max(len(headers), len(raw_rows[0])))):
            idx_to_field.setdefault(i, _REASON_FIELDS[i])
    records: list[dict[str, Any]] = []
    by_code: dict[str, int] = {}
    for row in raw_rows:
        rec: dict[str, Any] = {f: None for f in _REASON_FIELDS}
        for i, f in idx_to_field.items():
            val = _cell_str(row[i]) if i < len(row) else ""
            rec[f] = val or None
        code = rec.get("reason_code")
        if not code:
            continue
        # Source workbooks contain a few exact duplicate rows (AGE1/AGE2);
        # keep the richest copy so a code maps to exactly one record.
        prev = by_code.get(code)
        if prev is not None:
            old_len = sum(len(str(records[prev].get(f) or "")) for f in _REASON_FIELDS)
            new_len = sum(len(str(rec.get(f) or "")) for f in _REASON_FIELDS)
            if new_len > old_len:
                records[prev] = rec
            continue
        by_code[code] = len(records)
        records.append(rec)
    return records


def _reason_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group reason codes by their leading alpha family (HAD, ACH, DAG, …)."""
    counts: dict[str, int] = {}
    for rec in records:
        code = str(rec.get("reason_code") or "")
        m = re.match(r"^([A-Za-z]+)", code)
        if not m:
            continue
        fam = m.group(1).upper()
        counts[fam] = counts.get(fam, 0) + 1
    return [
        {"گروه": k, "تعداد": n}
        for k, n in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    ]


def _reason_full_text(
    list_name: str,
    model_label: str,
    hierarchy: list[str],
    records: list[dict[str, Any]],
) -> str:
    lines = [
        f"فهرست کامل «{list_name}» ({len(records)} کد دلیل) — {model_label}",
        "مسیر: " + " / ".join(hierarchy),
        "ستون‌ها: " + " | ".join(_REASON_FIELDS),
        "ردیف‌ها:",
    ]
    for i, rec in enumerate(records, 1):
        code = rec.get("reason_code") or ""
        txt = rec.get("reason_text") or ""
        brief = rec.get("brief_explanation") or ""
        lines.append(f"{i}. {code} | {txt} | {brief}")
    return "\n".join(lines)


def build_reason_code_entity(
    file_path: str,
    sheets: list[dict[str, Any]] | None,
    *,
    title: str | None = None,
) -> dict[str, Any] | None:
    """Build an EntityList payload for a reason-code workbook.

    Emits ``kind=reason_code_catalog`` with ``_meta.model`` / ``_meta.hierarchy``
    / ``_meta.subspace_ids`` so the four reason-code catalogs (cheque,
    person-main, person-substitute, legal) stay a clear hierarchical mapping
    and never cross-bleed.
    """
    spec = match_reason_spec(file_path)
    if spec is None:
        return None

    headers: list[Any] = []
    raw_rows: list[list[Any]] = []
    sheet_name, headers, raw_rows = _read_reason_matrix(file_path)
    if not raw_rows and sheets:
        for sheet in sheets:
            rows = sheet.get("rows") or []
            cleaned = [
                list(r)
                for r in rows
                if isinstance(r, (list, tuple)) and any(c is not None and str(c).strip() for c in r)
            ]
            if not cleaned:
                continue
            sheet_name = sheet.get("name") or sheet_name
            headers = cleaned[0]
            raw_rows = cleaned[1:]
            break

    records = _build_reason_code_records(headers or [], raw_rows)
    if not records:
        return None

    model = str(spec.get("model") or "")
    model_label = str(spec.get("model_label") or model)
    domain = str(spec.get("domain") or "")
    hierarchy = list(spec.get("hierarchy") or [])
    subspace_ids = list(spec.get("subspace_ids") or [])
    list_name = str(spec.get("list_name") or title or Path(file_path).stem)
    base_desc = str(spec.get("description") or f"کدهای دلیل {model_label}.")
    groups = _reason_groups(records)
    family_summary = ""
    if groups:
        top = "، ".join(f"{g['گروه']} ({g['تعداد']})" for g in groups[:8])
        family_summary = f" تعداد کل {len(records)}. خانواده‌ها: {top}."
    description = (
        f"{base_desc} مسیر: {' / '.join(hierarchy)}.{family_summary} "
        "برای کد مشخص، reason_code را جستجو کنید."
    )
    full_text = _reason_full_text(list_name, model_label, hierarchy, records)
    content_json: dict[str, Any] = {
        "_meta": {
            "kind": "reason_code_catalog",
            "domain": domain,
            "model": model,
            "model_label": model_label,
            "hierarchy": hierarchy,
            "subspace_ids": subspace_ids,
            "source_file": Path(file_path).name,
            "fields": list(_REASON_FIELDS),
            "count": len(records),
        },
        "records": records,
        "groups": groups,
        "full_list_text": full_text,
        sheet_name: records,
    }
    return {
        "list_name": list_name,
        "description": description,
        "content_json": content_json,
        "subspace_ids": subspace_ids,
        "kind": "reason_code_catalog",
        "model": model,
    }


def build_reason_code_index_entity() -> dict[str, Any]:
    """Build the top-level reason-code hierarchy index supplement.

    One catalog listing all reason-code dictionaries, their model path and
    code counts, so «چه کدهای دلیلی داریم؟» / «کدهای دلیل شامل چه مواردی
    است؟» can be answered across models in one shot without cross-model
    drift. Derived entirely from :data:`REASON_CODE_SPECS` — no hardcoded
    code lists.
    """
    rows: list[dict[str, Any]] = []
    for spec in REASON_CODE_SPECS:
        rows.append(
            {
                "model": spec.get("model"),
                "model_label": spec.get("model_label"),
                "hierarchy": list(spec.get("hierarchy") or []),
                "list_name": spec.get("list_name"),
                "subspace_ids": list(spec.get("subspace_ids") or []),
                "domain": spec.get("domain"),
            }
        )
    lines = [
        "فهرست سلسله‌مراتبی کدهای دلیل کاهش امتیاز (نمای کلی همه مدل‌ها)",
        "برای کد مشخص، خودِ کد (مثل BCC1) یا «کد دلیل» را جستجو کنید.",
        "مسیرها:",
    ]
    for r in rows:
        lines.append(f"- {' / '.join(r['hierarchy'])} — «{r['list_name']}»")
    lines.append("")
    lines.append("کدهای مشترک بین چند مدل (نیاز به پرسش مدل از کاربر):")
    for code, owners in sorted(AMBIGUOUS_REASON_CODES.items()):
        lines.append(f"- {code}: {', '.join(owners)}")
    description = (
        "نمای کلی و سلسله‌مراتبی کدهای دلیل کاهش امتیاز برای همه مدل‌ها: "
        "چک، اعتبارسنجی تسهیلات اشخاص حقیقی (مدل اصلی و مدل جایگزین) و "
        "اشخاص حقوقی. شامل مسیر هر فهرست و کدهای مشترک بین مدل‌ها."
    )
    return {
        "list_name": "فهرست سلسله‌مراتبی کدهای دلیل",
        "description": description,
        "content_json": {
            "_meta": {
                "kind": "reason_code_index",
                "domain": "credit",
                "source_file": "derived:REASON_CODE_SPECS",
                "fields": ["model", "model_label", "hierarchy", "list_name", "subspace_ids", "domain"],
                "count": len(rows),
            },
            "records": rows,
            "ambiguous_codes": dict(AMBIGUOUS_REASON_CODES),
            "full_list_text": "\n".join(lines),
        },
        "subspace_ids": sorted({s for r in rows for s in r["subspace_ids"]}),
        "kind": "reason_code_index",
        "model": "",
    }


def _read_reason_matrix(file_path: str) -> tuple[str, list[Any], list[list[Any]]]:
    """Read the first sheet of a reason-code xlsx via openpyxl.

    Mirrors ``_read_catalog_matrix`` but keeps the canonical reason-code
    columns even when Excel stores a trailing «نظر»/comment column.
    """
    try:
        from openpyxl import load_workbook
    except ImportError:
        return "Sheet1", [], []
    wb = load_workbook(file_path, data_only=True, read_only=True)
    ws = wb[wb.sheetnames[0]]
    headers = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    while headers and (headers[-1] is None or not str(headers[-1]).strip()):
        headers.pop()
    if not headers:
        wb.close()
        return ws.title or "Sheet1", [], []
    ncols = len(headers)
    rows: list[list[Any]] = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        vals = list(row[:ncols])
        if not any(v is not None and str(v).strip() for v in vals):
            continue
        rows.append(vals)
    sheet_name = ws.title or "Sheet1"
    wb.close()
    return sheet_name, headers, rows


def entity_subspace_ids(ent: Any) -> list[str]:
    """Read subspace tags from EntityList ORM / dict / content_json."""
    content = None
    if isinstance(ent, dict):
        content = ent.get("content") or ent.get("content_json")
    else:
        content = getattr(ent, "content_json", None)
    if isinstance(content, dict):
        meta = content.get("_meta") or {}
        ids = meta.get("subspace_ids") or []
        if isinstance(ids, list):
            return [str(x) for x in ids]
    return []


def entity_kind(ent: Any) -> str:
    content = None
    if isinstance(ent, dict):
        content = ent.get("content") or ent.get("content_json")
    else:
        content = getattr(ent, "content_json", None)
    if isinstance(content, dict):
        meta = content.get("_meta") or {}
        return str(meta.get("kind") or "")
    return ""


def infer_score_model_from_query(query: str) -> str | None:
    """Heuristic model pick from query text / numeric score scale."""
    q = query or ""
    ql = q.lower()
    if any(k in q for k in ("چک", "صیاد", "برگشتی")) or "cheque" in ql:
        return "cheque"
    if any(k in q for k in ("حقوقی", "شرکت", "تامین‌کننده", "تامین کننده")):
        return "legal"
    if any(k in q for k in ("حقیقی", "نردبان", "کفیل", "اشخاص حقیقی")):
        return "person"
    nums = [float(m.group(0)) for m in re.finditer(r"\d{2,3}(?:\.\d+)?", q)]
    for n in nums:
        if 1 <= n <= 100:
            return "cheque"
        if 250 <= n <= 900:
            return None
    return None


def entity_model(ent: Any) -> str:
    """Read the model tag from an EntityList ORM / dict / content_json."""
    content = None
    if isinstance(ent, dict):
        content = ent.get("content") or ent.get("content_json")
    else:
        content = getattr(ent, "content_json", None)
    if isinstance(content, dict):
        meta = content.get("_meta") or {}
        return str(meta.get("model") or "")
    return ""


def entity_hierarchy(ent: Any) -> list[str]:
    """Read the hierarchy path from an EntityList ORM / dict / content_json."""
    content = None
    if isinstance(ent, dict):
        content = ent.get("content") or ent.get("content_json")
    else:
        content = getattr(ent, "content_json", None)
    if isinstance(content, dict):
        meta = content.get("_meta") or {}
        h = meta.get("hierarchy") or []
        if isinstance(h, list):
            return [str(x) for x in h]
    return []


def is_ambiguous_reason_code(code: str, model: str | None = None) -> list[str] | None:
    """Return the list of models a code is ambiguous across (or None).

    ``AGE2`` exists in person-main and person-substitute; ``NGL1``–``NGL3``
    in person-main and legal. When *model* is given the code is ambiguous
    only if another model also defines it.
    """
    c = (code or "").strip().upper()
    owners = AMBIGUOUS_REASON_CODES.get(c)
    if not owners:
        return None
    if model:
        return owners if len([m for m in owners if m != model]) >= 1 else None
    return owners


def filter_entities_for_context(
    entities: list[Any],
    *,
    query: str = "",
    subspace: str | None = None,
    prefer_score: bool = False,
    reason_code_model: str | None = None,
) -> list[Any]:
    """Prefer subspace-matching tables; drop conflicting score/reason models."""
    if not entities:
        return []
    sub = (subspace or "").strip()
    model_hint = infer_score_model_from_query(query)
    if sub == "cheque":
        model_hint = "cheque"
    elif sub in ("facilities_person", "credit_ladder", "guarantor", "customers_person"):
        model_hint = "person"
    elif sub in ("facilities_legal", "supplier_legal", "customers_legal"):
        model_hint = "legal"

    score_ents = []
    reason_ents = []
    other = []
    for ent in entities:
        kind = entity_kind(ent)
        name = ""
        if isinstance(ent, dict):
            name = str(ent.get("list_name") or "")
        else:
            name = str(getattr(ent, "list_name", "") or "")
        is_score = kind == "score_table" or name.startswith("جدول امتیاز")
        if is_score:
            score_ents.append(ent)
        elif kind == "reason_code_catalog":
            reason_ents.append(ent)
        else:
            other.append(ent)

    if score_ents and (prefer_score or model_hint or sub):
        kept_scores = []
        for ent in score_ents:
            ids = entity_subspace_ids(ent)
            content = (
                ent.get("content") or ent.get("content_json")
                if isinstance(ent, dict)
                else getattr(ent, "content_json", None)
            )
            model = None
            if isinstance(content, dict):
                model = (content.get("_meta") or {}).get("model")
            if model_hint and model and model != model_hint:
                continue
            if sub and ids and sub not in ids and model_hint is None:
                continue
            kept_scores.append(ent)
        if not kept_scores and sub:
            kept_scores = [e for e in score_ents if sub in entity_subspace_ids(e)]
        score_ents = kept_scores

    # Reason-code catalogs never cross subspaces: a cheque ask must not be
    # answered from the individuals/legal catalog and vice versa.
    if reason_ents and sub:
        kept_reason = [e for e in reason_ents if sub in entity_subspace_ids(e)]
        if kept_reason:
            reason_ents = kept_reason
    # A resolved model pick (AGE2/NGL1–3 disambiguation) pins exactly one
    # catalog and drops the hierarchy index, whose "ask which model" note would
    # otherwise make the model re-ask a question the user already answered.
    if reason_code_model:
        reason_ents = [
            e for e in reason_ents
            if entity_model(e) == reason_code_model
        ]
        other = [e for e in other if entity_kind(e) != "reason_code_index"]
    return score_ents + reason_ents + other


_REASON_CODE_QUERY_RE = re.compile(r"\b[A-Za-z]{2,6}\d{1,2}(?:-\d{1,2})?\b")
_REASON_QUERY_WORDS = (
    "کد دلیل",
    "کدهای دلیل",
    "کدهای علل",
    "کد علت",
    "reason code",
    "دلیل کاهش امتیاز",
    "دلایل کاهش امتیاز",
    "شرح دلیل",
    "شرح کد",
    "متن دلیل",
)


def is_reason_code_query(query: str) -> bool:
    """True when a query targets the reason-code dictionaries.

    Matches either a code-shaped token (``HAD1``, ``DAG1-1``) or the
    reason-code vocabulary («کد دلیل», «دلیل کاهش امتیاز», …).
    """
    q = query or ""
    if any(w in q for w in _REASON_QUERY_WORDS):
        return True
    # Avoid matching ordinary uppercase words: require a digit in the token.
    return bool(_REASON_CODE_QUERY_RE.search(q))


def extract_reason_code(query: str) -> str | None:
    """Return the reason-code token in *query* (uppercased) if present."""
    m = _REASON_CODE_QUERY_RE.search(query or "")
    return m.group(0).upper() if m else None


def is_score_query(query: str) -> bool:
    q = query or ""
    if any(
        k in q
        for k in (
            "امتیاز",
            "رتبه",
            "گرید",
            "رنگ",
            "آبی",
            "سبز",
            "زرد",
            "نارنجی",
            "قرمز",
            "نکول",
            "جدول",
            "فهرست",
            "لیست رنگ",
            "لیست رتبه",
            "بازه",
        )
    ):
        return True
    if re.search(r"\b[A-Za-z]\d{1,2}\b", q):
        return True
    return False
