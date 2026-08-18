"""Performance / benchmark plots rendered to PNG files.

Uses matplotlib with the Agg backend so the module can run headless.
Persian labels are handled through a font fallback when a Persian font is
available; otherwise English fallback labels are used.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

_FA = ["DejaVu Sans"]
_persian_fonts = [
    "Tahoma",
    "Segoe UI",
    "Noto Sans Arabic",
    "B Nazanin",
    "IRANSans",
]
for _f in _persian_fonts:
    if any(_f.lower() in (f.name.lower() or "") for f in matplotlib.font_manager.fontManager.ttflist):
        _FA.append(_f)
        break

# Labels: (fa, en). Choose based on whether a Persian font loaded.
_USE_FA = len(_FA) > 1
_FA_LABELS = {
    "hit_rate": "نرخ بازیابی (Hit@K)",
    "mrr": "میانگین رتبه متقابل (MRR)",
    "top1": "نرخ رتبه اول (Top-1)",
    "latency": "زمان پاسخ (میلی‌ثانیه)",
    "format": "فرمت پرسش",
    "queries": "تعداد پرسش",
    "duplicates": "تکرار پرسش‌ها",
    "distinct": "پرسش یکتا",
    "copies": "نسخه‌های تکراری",
}


def _label(key: str, en: str) -> str:
    return (_FA_LABELS.get(key, en) if _USE_FA else en)


def _style_axis(ax, xlabel_en: str, ylabel_en: str, title_en: str, title_key: str = "") -> None:
    ax.set_ylabel(_label(title_key, ylabel_en))
    ax.set_xlabel(xlabel_en)
    ax.set_title(title_en)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(20)
        lbl.set_ha("right")
    ax.grid(axis="y", alpha=0.3)


def _load_json(path_or_dict: str | dict) -> dict:
    if isinstance(path_or_dict, dict):
        return path_or_dict
    import os

    with open(os.fspath(path_or_dict), encoding="utf-8") as f:
        return json.load(f)


def plot_hit_rate_by_format(result_json: str | dict, out_png: str) -> str:
    """Bar chart of hit rate per query format."""
    data = _load_json(result_json)
    by_format = data.get("by_format", {})
    formats = sorted(by_format)
    hits = [by_format[f].get("hit_rate", 0) * 100 for f in formats]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(formats, hits, color="#4a7dbf", alpha=0.9)
    ax.set_ylim(0, 105)
    for i, h in enumerate(hits):
        ax.text(i, h + 1, f"{h:.0f}%", ha="center", fontsize=9)
    _style_axis(ax, _label("format", "Query format"), "Hit rate (%)",
                "Hit@K by Query Format", "hit_rate")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def plot_mrr_by_format(result_json: str | dict, out_png: str) -> str:
    """Bar chart of MRR per query format."""
    data = _load_json(result_json)
    by_format = data.get("by_format", {})
    formats = sorted(by_format)
    mrrs = [by_format[f].get("mrr", 0) for f in formats]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(formats, mrrs, color="#7f9c5b", alpha=0.9)
    ax.set_ylim(0, 1.05)
    for i, m in enumerate(mrrs):
        ax.text(i, m + 0.02, f"{m:.2f}", ha="center", fontsize=9)
    _style_axis(ax, _label("format", "Query format"), "MRR",
                "MRR by Query Format", "mrr")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def plot_latency_distribution(result_json: str | dict, out_png: str) -> str:
    """Histogram of per-query latencies."""
    data = _load_json(result_json)
    lat = [q.get("elapsed_ms", 0) for q in data.get("queries", []) if q.get("elapsed_ms", 0) > 0]

    fig, ax = plt.subplots(figsize=(8, 4))
    if lat:
        ax.hist(lat, bins=30, color="#b5713f", alpha=0.85, edgecolor="white")
    ax.set_xlabel(_label("latency", "Latency (ms)"))
    ax.set_ylabel("Queries")
    ax.set_title("Query Latency Distribution")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def plot_duplicate_stats(stats_json: str | dict, out_png: str) -> str:
    """Bar of distinct vs duplicated questions in the KB."""
    stats = _load_json(stats_json)
    distinct = stats.get("distinct_questions", 0)
    copies = stats.get("duplicate_instances", 0)

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [_label("distinct", "Distinct"), _label("copies", "Duplicate copies")]
    values = [distinct, copies]
    ax.bar(labels, values, color=["#4a7dbf", "#c0504d"], alpha=0.9)
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.01, str(v), ha="center", fontsize=10)
    ax.set_title("QA Duplication in KB")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def render_benchmark_plots(result_json: str, out_dir: str) -> list[str]:
    """Render all standard benchmark plots into *out_dir*."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    generated = [
        plot_hit_rate_by_format(result_json, str(out / "hit_rate_by_format.png")),
        plot_mrr_by_format(result_json, str(out / "mrr_by_format.png")),
        plot_latency_distribution(result_json, str(out / "latency_distribution.png")),
    ]
    return generated
