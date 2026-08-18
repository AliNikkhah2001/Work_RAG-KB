"""Runnable retrieval benchmarks.

Loads a test dataset (queries + expected chunk ids + format/difficulty
labels), runs every query through the search pipeline, and computes
per-format and overall IR metrics. Results are persisted as JSON so they
can be diffed across KB versions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeAlias

from kb_manager.evaluation.metrics import RanxRetrievalEvaluator, RetrievalResult

logger = logging.getLogger(__name__)

SearchFn: TypeAlias = Callable[..., object]


@dataclass
class BenchmarkQuery:
    """A single benchmark query and its verdict."""

    query: str
    expected_ids: list[str]
    format: str = "verbatim"
    difficulty: str = "easy"
    similarity_to_gt: float | None = None
    hit: bool = False
    rank: int = -1
    top1_hit: bool = False
    elapsed_ms: float = 0.0
    retrieved_ids: list[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Aggregated benchmark output for one run."""

    version: str
    created_at: str
    top_k: int
    total_queries: int
    by_format: dict[str, dict[str, float]] = field(default_factory=dict)
    overall: dict[str, float] = field(default_factory=dict)
    queries: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class BenchmarkRunner:
    """Execute a test dataset against a search function."""

    def __init__(
        self,
        search_fn: SearchFn,
        top_k: int = 5,
        version: str = "current",
    ) -> None:
        self._search = search_fn
        self._top_k = top_k
        self._version = version

    @staticmethod
    def load_dataset(path: str) -> list[dict]:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def run(
        self,
        dataset: list[dict],
        progress: Callable[[int, int], None] | None = None,
    ) -> BenchmarkResult:
        """Run the benchmark synchronously (assumes a sync search fn)."""
        queries: list[BenchmarkQuery] = []
        total = len(dataset)

        for i, item in enumerate(dataset):
            query = item.get("query", "")
            expected = item.get("expected_chunk_ids", [])
            fmt = item.get("format", "verbatim")
            difficulty = item.get("difficulty", "medium")

            start = time.monotonic()
            raw = self._search(query, self._top_k)
            elapsed_ms = (time.monotonic() - start) * 1000

            retrieved = [r[0] for r in raw] if raw else []
            rank = -1
            for idx, cid in enumerate(retrieved):
                if cid in expected:
                    rank = idx + 1
                    break
            hit = rank != -1
            top1_hit = rank == 1

            queries.append(
                BenchmarkQuery(
                    query=query,
                    expected_ids=expected,
                    format=fmt,
                    difficulty=difficulty,
                    similarity_to_gt=item.get("gt_similarity"),
                    hit=hit,
                    rank=rank,
                    top1_hit=top1_hit,
                    elapsed_ms=round(elapsed_ms, 1),
                    retrieved_ids=retrieved,
                )
            )
            if progress:
                progress(i + 1, total)

        result = BenchmarkResult(
            version=self._version,
            created_at=datetime.now(UTC).isoformat(),
            top_k=self._top_k,
            total_queries=total,
        )

        formats = sorted({q.format for q in queries})
        for fmt in formats:
            fmt_queries = [q for q in queries if q.format == fmt]
            result.by_format[fmt] = self._aggregate(fmt_queries)

        result.overall = self._aggregate(queries)
        result.queries = [q.__dict__ for q in queries]
        return result

    def _aggregate(self, queries: list[BenchmarkQuery]) -> dict[str, float]:
        n = len(queries)
        if n == 0:
            return {}
        hits = sum(1 for q in queries if q.hit)
        top1 = sum(1 for q in queries if q.top1_hit)
        latencies = [q.elapsed_ms for q in queries if q.elapsed_ms > 0]
        ranks = [q.rank for q in queries if q.rank != -1]
        mrr = sum(1.0 / q.rank for q in queries if q.rank != -1) / n if n else 0.0

        return {
            "queries": n,
            "hit_rate": round(hits / n, 4),
            "top1_hit_rate": round(top1 / n, 4),
            "mrr": round(mrr, 4),
            "avg_rank": round(sum(ranks) / len(ranks), 2) if ranks else 0.0,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        }

    @staticmethod
    def save(result: BenchmarkResult, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)


class AsyncBenchmarkRunner(BenchmarkRunner):
    """Benchmark runner for an async search function."""

    def run(
        self,
        dataset: list[dict],
        progress: Callable[[int, int], None] | None = None,
    ) -> BenchmarkResult:
        return asyncio.run(self._run_async(dataset, progress))

    async def _run_async(
        self,
        dataset: list[dict],
        progress: Callable[[int, int], None] | None = None,
    ) -> BenchmarkResult:
        queries: list[BenchmarkQuery] = []
        total = len(dataset)

        for i, item in enumerate(dataset):
            query = item.get("query", "")
            expected = item.get("expected_chunk_ids", [])
            fmt = item.get("format", "verbatim")
            difficulty = item.get("difficulty", "medium")

            start = time.monotonic()
            raw = await self._search(query, self._top_k)
            elapsed_ms = (time.monotonic() - start) * 1000

            retrieved = [r[0] for r in raw] if raw else []
            rank = -1
            for idx, cid in enumerate(retrieved):
                if cid in expected:
                    rank = idx + 1
                    break
            hit = rank != -1
            top1_hit = rank == 1

            queries.append(
                BenchmarkQuery(
                    query=query,
                    expected_ids=expected,
                    format=fmt,
                    difficulty=difficulty,
                    similarity_to_gt=item.get("gt_similarity"),
                    hit=hit,
                    rank=rank,
                    top1_hit=top1_hit,
                    elapsed_ms=round(elapsed_ms, 1),
                    retrieved_ids=retrieved,
                )
            )
            if progress:
                progress(i + 1, total)

        result = BenchmarkResult(
            version=self._version,
            created_at=datetime.now(UTC).isoformat(),
            top_k=self._top_k,
            total_queries=total,
        )

        formats = sorted({q.format for q in queries})
        for fmt in formats:
            result.by_format[fmt] = self._aggregate([q for q in queries if q.format == fmt])
        result.overall = self._aggregate(queries)
        result.queries = [q.__dict__ for q in queries]
        return result


def summarize_ir_metrics(
    result: BenchmarkResult,
    relevance_builder: Callable[[dict], dict[str, float]] | None = None,
) -> dict[str, float]:
    """Run ranx / pure-Python IR metrics over the benchmark queries.

    ``relevance_builder`` maps a query dict to a {chunk_id: relevance}
    dict (defaults to treating expected ids as relevant).
    """
    if relevance_builder is None:
        relevance_builder = lambda q: dict.fromkeys(q.get("expected_ids", []), 1.0)  # noqa: E731

    ir_results: list[RetrievalResult] = []
    for q in result.queries:
        ir_results.append(
            RetrievalResult(
                query=q["query"],
                retrieved_ids=q["retrieved_ids"],
                retrieved_scores=[1.0] * len(q["retrieved_ids"]),
                expected_ids=q["expected_ids"],
                relevance_scores=relevance_builder(q),
            )
        )
    return RanxRetrievalEvaluator.compute_all(ir_results, k=result.top_k)
