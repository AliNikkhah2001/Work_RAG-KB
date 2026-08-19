"""Run the retrieval benchmark over the live KB and render plots.

This is the scriptable equivalent of the web /benchmarks/run endpoint:
1. Loads a test dataset (multi-format queries).
2. Runs every query through the search pipeline (BM25 + TF-IDF RRF).
3. Computes per-format Hit@K / MRR / latency plus ranx IR metrics.
4. Writes data/benchmark_results.json, data/ir_metrics.json,
   data/qa_duplication.json and renders PNG plots into data/plots/.

Supports FaMTEB datasets via --famteb flag.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, List


async def main(
    dataset_name: str = "test_questions.json",
    top_k: int = 5,
    use_famteb: bool = False,
    famteb_datasets: Optional[List[str]] = None,
    max_samples: int = 100,
) -> None:
    from kb_manager.config import PROJECT_ROOT
    from kb_manager.evaluation.benchmark import (
        AsyncBenchmarkRunner,
        summarize_ir_metrics,
    )
    from kb_manager.evaluation.plots import plot_duplicate_stats, render_benchmark_plots
    from kb_manager.versioning.snapshot import export_compact
    from kb_manager.web.routes.search import search_knowledge_base

    data_dir = PROJECT_ROOT / "data"

    # Load dataset
    if use_famteb:
        from kb_manager.famteb import load_famteb_benchmark
        print(f"Loading FaMTEB datasets: {famteb_datasets or 'all'}")
        dataset = load_famteb_benchmark(
            dataset_keys=famteb_datasets,
            max_samples_per_dataset=max_samples,
        )
        dataset_name = "famteb_benchmark"
    else:
        dataset_path = data_dir / dataset_name
        if not dataset_path.exists():
            print(f"dataset not found: {dataset_path}")
            sys.exit(1)
        with open(dataset_path, encoding="utf-8") as f:
            dataset = json.load(f)

    async def _search(query: str, k: int):
        steps = await search_knowledge_base(query, k)
        return [(r.chunk_id, r.hybrid_score) for r in steps.final_results]

    from kb_manager.evaluation.benchmark import AsyncBenchmarkRunner

    runner = AsyncBenchmarkRunner(_search, top_k=top_k, version="live-kb")

    async def _with_progress():
        result = await runner._run_async(
            dataset,
            progress=lambda done, total: print(
                f"\r  [{done}/{total}]", end="", flush=True
            ),
        )
        print()
        return result

    print(f"Benchmarking {len(dataset)} queries (top_k={top_k}) against live KB...")
    result = await _with_progress()

    try:
        from kb_manager.evaluation.benchmark import summarize_ir_metrics
        ir = summarize_ir_metrics(result)
    except Exception as exc:
        print("IR metrics unavailable:", exc)
        ir = {}

    result_json = result.to_dict()
    (data_dir / "benchmark_results.json").write_text(
        json.dumps(result_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (data_dir / "ir_metrics.json").write_text(
        json.dumps({"top_k": top_k, "metrics": ir}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    from kb_manager.versioning.snapshot import export_compact
    export = export_compact(str(PROJECT_ROOT / "data" / "kb_test.db"))
    dups = export["counts"]
    (data_dir / "qa_duplication.json").write_text(
        json.dumps(dups, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plots_dir = data_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    from kb_manager.evaluation.plots import render_benchmark_plots, plot_duplicate_stats
    render_benchmark_plots(
        str(data_dir / "benchmark_results.json"), str(plots_dir)
    )
    plot_duplicate_stats(data_dir / "qa_duplication.json", plots_dir / "qa_duplication.png")

    print("\n=== Overall ===")
    print(json.dumps(result.overall, indent=2, ensure_ascii=False))
    print("\n=== By Format ===")
    print(json.dumps(result.by_format, indent=2, ensure_ascii=False))
    print("\n=== IR Metrics ===")
    print(json.dumps(ir, indent=2, ensure_ascii=False))
    print("\nPlots:", [p.name for p in plots_dir.glob("*.png")])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retrieval benchmark")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="test_questions.json",
        help="Dataset file name (default: test_questions.json)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to retrieve (default: 5)",
    )
    parser.add_argument(
        "--famteb",
        action="store_true",
        help="Run benchmark on FaMTEB datasets",
    )
    parser.add_argument(
        "--famteb-datasets",
        nargs="+",
        default=None,
        help="Specific FaMTEB datasets to use (default: all retrieval datasets)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Max samples per FaMTEB dataset (default: 100)",
    )
    args = parser.parse_args()

    asyncio.run(main(
        dataset_name=args.dataset,
        top_k=args.top_k,
        use_famteb=args.famteb,
        famteb_datasets=args.famteb_datasets,
        max_samples=args.max_samples,
    ))
