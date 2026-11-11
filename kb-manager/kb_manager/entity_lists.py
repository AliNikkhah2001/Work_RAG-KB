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
        "match": ("لیست و عناوین وام",),
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


def match_score_spec(file_path: str) -> dict[str, Any] | None:
    name = Path(file_path).stem
    blob = f"{file_path} {name}"
    for spec in SCORE_TABLE_SPECS:
        if any(m in blob for m in spec["match"]):
            return spec
    return None


def match_list_spec(file_path: str) -> dict[str, Any] | None:
    name = Path(file_path).stem
    blob = f"{file_path} {name}"
    for spec in LIST_CATALOG_SPECS:
        if any(m in blob for m in spec["match"]):
            return spec
    return None


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


def filter_entities_for_context(
    entities: list[Any],
    *,
    query: str = "",
    subspace: str | None = None,
    prefer_score: bool = False,
) -> list[Any]:
    """Prefer subspace-matching score tables; drop conflicting score models."""
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
        return kept_scores + other
    return entities


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
