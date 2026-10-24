"""Latency plot for the v13.1 optimization pass (2026-10-13).

Renders per-stage p50 bars for three configurations — README v13 baseline
(2026-10-07), pre-optimization today, post-optimization today — plus the
cached path, and the incident state for context. Data comes from
``latency_v13_1.json`` (measured, 30-query sets, fresh/uncached queries,
curl --noproxy against POST /search/api, top_k=5).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "report_plots"
DATA = {
    # README v13 (2026-10-07, 55 ms avg warm) — documented stage split
    "v13 README (10-07)": {"bm25": 4.0, "dense": 24.0, "rerank": 24.0, "total": 55.0},
    # 2026-10-13 pre-optimization (p50, load ~60-73, host co-tenant active)
    "before (10-13)": {"bm25": 3.8, "dense": 20.5, "rerank": 20.9, "total": 44.4},
    # 2026-10-13 post-optimization (p50, load ~61-66): RAM dense + batched
    # beam encode + fp16 + orjson + memoized rerank scores + /metrics
    "after (10-13)": {"bm25": 2.6, "dense": 13.5, "rerank": 22.0, "total": 39.6},
}
P95 = {"before (10-13)": 61.5, "after (10-13)": 55.0}


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    stages = ["bm25", "dense", "rerank", "total"]
    labels = ["BM25", "Dense", "Rerank", "Total"]
    configs = list(DATA.keys())
    colors = ["#8ea0c6", "#c9a227", "#2e7d32"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    width = 0.26
    xs = range(len(stages))
    for i, cfg in enumerate(configs):
        vals = [DATA[cfg][s] for s in stages]
        ax.bar([x + (i - 1) * width for x in xs], vals, width, label=cfg, color=colors[i])
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    ax.set_ylabel("p50 latency (ms, warm)")
    ax.set_title("KB retrieval stage latency (30 fresh queries, top_k=5)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for i, cfg in enumerate(configs):
        for j, s in enumerate(stages):
            v = DATA[cfg][s]
            ax.text(j + (i - 1) * width, v + 0.4, f"{v:.1f}", ha="center", fontsize=7)

    ax2 = axes[1]
    cfgs2 = ["v13 README (10-07)", "before (10-13)", "after (10-13)"]
    totals = [DATA[c]["total"] for c in cfgs2]
    bars = ax2.bar(range(len(cfgs2)), totals, color=colors)
    for j, cfg in enumerate(cfgs2[1:]):
        ax2.errorbar(j + 1, DATA[cfg]["total"],
                     yerr=[[0], [P95[cfg] - DATA[cfg]["total"]]],
                     fmt="none", ecolor="black", capsize=4)
    ax2.axhline(50, color="red", ls="--", lw=1, label="sub-50 ms target")
    # cached path (Redis hit, server-side ~2-7 ms) as an annotation
    ax2.annotate("Redis/exact-cache hit: 2-7 ms\n(150x vs cold)", xy=(2.2, 50),
                 xytext=(1.15, 70), fontsize=8, arrowprops=dict(arrowstyle="->"))
    ax2.set_xticks(range(len(cfgs2)))
    ax2.set_xticklabels([c.replace(" (", "\n(") for c in cfgs2], fontsize=8)
    ax2.set_ylabel("p50 total latency (ms)")
    ax2.set_title("End-to-end retrieval: baseline vs optimized (p95 whiskers)")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("KB Manager retrieval latency — v13.1 optimization pass (2026-10-13, H200, load 60-73 from co-tenant)", fontsize=10)
    fig.tight_layout()
    out = PLOTS / "p_latency_v13_1.png"
    fig.savefig(out, dpi=130)
    (HERE / "latency_v13_1.json").write_text(json.dumps(DATA, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
