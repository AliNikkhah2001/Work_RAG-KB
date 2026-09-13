"""Retrieval evaluation metrics and framework."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field


@dataclass
class RetrievalResult:
    """Result of a single query retrieval."""

    query: str
    retrieved_ids: list[str]
    retrieved_scores: list[float]
    expected_ids: list[str]
    relevance_scores: dict[str, float] = field(default_factory=dict)


class RetrievalMetrics:
    """Standard IR evaluation metrics.

    All methods take a list of RetrievalResult and return metric values.
    """

    @staticmethod
    def precision_at_k(results: list[RetrievalResult], k: int = 10) -> float:
        """Fraction of top-K results that are relevant."""
        if not results:
            return 0.0
        total = 0.0
        for r in results:
            top_k = r.retrieved_ids[:k]
            relevant = sum(
                1 for rid in top_k
                if r.relevance_scores.get(rid, 0.0) > 0
            )
            total += relevant / k
        return total / len(results)

    @staticmethod
    def recall_at_k(results: list[RetrievalResult], k: int = 10) -> float:
        """Fraction of relevant docs found in top-K."""
        if not results:
            return 0.0
        total = 0.0
        for r in results:
            top_k = r.retrieved_ids[:k]
            relevant_expected = sum(
                1 for eid in r.expected_ids
                if r.relevance_scores.get(eid, 0.0) > 0
            )
            if relevant_expected == 0:
                continue
            found = sum(
                1 for rid in top_k
                if rid in r.expected_ids
                and r.relevance_scores.get(rid, 0.0) > 0
            )
            total += found / relevant_expected
        return total / len(results)

    @staticmethod
    def hit_rate_at_k(results: list[RetrievalResult], k: int = 10) -> float:
        """Fraction of queries with at least one relevant result in top-K."""
        if not results:
            return 0.0
        hits = 0
        for r in results:
            top_k = r.retrieved_ids[:k]
            if any(
                r.relevance_scores.get(rid, 0.0) > 0 for rid in top_k
            ):
                hits += 1
        return hits / len(results)

    @staticmethod
    def mrr(results: list[RetrievalResult]) -> float:
        """Mean Reciprocal Rank — 1/rank of first relevant result."""
        if not results:
            return 0.0
        total = 0.0
        for r in results:
            for rank, rid in enumerate(r.retrieved_ids, 1):
                if r.relevance_scores.get(rid, 0.0) > 0:
                    total += 1.0 / rank
                    break
        return total / len(results)

    @staticmethod
    def ndcg_at_k(results: list[RetrievalResult], k: int = 10) -> float:
        """Normalized Discounted Cumulative Gain at K."""
        if not results:
            return 0.0
        total = 0.0
        for r in results:
            dcg = 0.0
            for i, rid in enumerate(r.retrieved_ids[:k]):
                rel = r.relevance_scores.get(rid, 0.0)
                dcg += rel / math.log2(i + 2)  # i+2 because log2(1)=0

            # Ideal DCG
            ideal_rels = sorted(
                [r.relevance_scores.get(eid, 0.0) for eid in r.expected_ids],
                reverse=True,
            )[:k]
            idcg = sum(
                rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels)
            )

            total += dcg / idcg if idcg > 0 else 0.0
        return total / len(results)

    @staticmethod
    def map_at_k(results: list[RetrievalResult], k: int = 10) -> float:
        """Mean Average Precision at K."""
        if not results:
            return 0.0
        total = 0.0
        for r in results:
            relevant_count = 0
            precision_sum = 0.0
            for i, rid in enumerate(r.retrieved_ids[:k]):
                if r.relevance_scores.get(rid, 0.0) > 0:
                    relevant_count += 1
                    precision_sum += relevant_count / (i + 1)
            relevant_expected = sum(
                1 for eid in r.expected_ids
                if r.relevance_scores.get(eid, 0.0) > 0
            )
            if relevant_expected > 0:
                total += precision_sum / relevant_expected
        return total / len(results)

    @classmethod
    def compute_all(
        cls, results: list[RetrievalResult], k: int = 10
    ) -> dict[str, float]:
        """Compute all metrics at once."""
        return {
            f"precision@{k}": cls.precision_at_k(results, k),
            f"recall@{k}": cls.recall_at_k(results, k),
            f"hit_rate@{k}": cls.hit_rate_at_k(results, k),
            "mrr": cls.mrr(results),
            f"ndcg@{k}": cls.ndcg_at_k(results, k),
            f"map@{k}": cls.map_at_k(results, k),
        }


class RanxRetrievalEvaluator:
    """ranx-backed IR retrieval metrics.

    Wraps ``ranx.evaluate`` with the same ``RetrievalResult`` interface as
    ``RetrievalMetrics``. Falls back to the pure-Python implementation when
    ranx is not installed.
    """

    _ranx_available: bool | None = None

    @classmethod
    def available(cls) -> bool:
        """Return True if ranx can be imported (memoized)."""
        if cls._ranx_available is None:
            try:
                import ranx  # noqa: F401
            except ImportError:
                cls._ranx_available = False
            else:
                cls._ranx_available = True
        return cls._ranx_available

    @classmethod
    def compute_all(
        cls, results: list[RetrievalResult], k: int = 10
    ) -> dict[str, float]:
        """Compute IR metrics, using ranx when available."""
        if not cls.available():
            return RetrievalMetrics.compute_all(results, k)
        return _ranx_compute_all(results, k)


def _ranx_compute_all(
    results: list[RetrievalResult], k: int
) -> dict[str, float]:
    import ranx

    metric_labels = [
        f"map@{k}",
        f"mrr@{k}",
        f"ndcg@{k}",
        f"recall@{k}",
        f"precision@{k}",
    ]

    # Skip queries with no relevance info — ranx requires a qrels entry.
    usable = [
        r for r in results
        if any(v > 0 for v in r.relevance_scores.values())
    ]
    if not usable:
        return dict.fromkeys(metric_labels, 0.0)

    qrels_dict: dict[str, dict[str, int]] = {}
    run_dict: dict[str, dict[str, float]] = {}

    for idx, r in enumerate(usable):
        qid = f"q{idx}"
        qrels_dict[qid] = {
            doc_id: int(score > 0)
            for doc_id, score in r.relevance_scores.items()
            if score > 0
        }
        # ranx needs a shared run signature (union of all doc ids seen).
        run_dict[qid] = dict(zip(r.retrieved_ids, r.retrieved_scores, strict=False))

    qrels = ranx.Qrels(qrels_dict)
    run = ranx.Run(run_dict)

    scores = ranx.evaluate(qrels, run, metric_labels)
    # ranx returns dict values as numpy scalars.
    return {label: float(scores[label]) for label in metric_labels}


class HeuristicOverlapEvaluator:
    """Heuristic overlap metrics — NOT LLM-judged.

    Previous name RAGEvaluator implied LLM faithfulness; renamed to avoid
    conflation with real RAGAS (ragas_metrics.RagasEvaluator). Pure term overlap.
    """

    @staticmethod
    def context_relevance(
        query: str, contexts: list[str]
    ) -> float:
        """Simple heuristic: fraction of contexts that contain query terms."""
        if not contexts:
            return 0.0
        query_terms = set(query.split())
        relevant = 0
        for ctx in contexts:
            ctx_terms = set(ctx.split())
            overlap = len(query_terms & ctx_terms)
            if overlap > 0:
                relevant += 1
        return relevant / len(contexts)

    @staticmethod
    def answer_coverage(
        answer: str, contexts: list[str]
    ) -> float:
        """Simple heuristic: fraction of answer terms found in contexts."""
        if not answer or not contexts:
            return 0.0
        answer_terms = set(answer.split())
        combined_context = " ".join(contexts)
        context_terms = set(combined_context.split())
        covered = len(answer_terms & context_terms)
        return covered / len(answer_terms) if answer_terms else 0.0

    @staticmethod
    def faithfulness_score(
        answer: str, contexts: list[str]
    ) -> float:
        """Simple heuristic: answer term overlap with context."""
        return HeuristicOverlapEvaluator.answer_coverage(answer, contexts)


# Backward compat alias — do not use in new code
RAGEvaluator = HeuristicOverlapEvaluator


class EvaluationRunner:
    """Run full evaluation pipeline."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def run_retrieval_evaluation(
        self,
        eval_dataset_path: str,
        search_fn,
        k: int = 10,
    ) -> dict[str, float]:
        """Run retrieval metrics on an evaluation dataset.

        Args:
            eval_dataset_path: Path to JSON evaluation dataset.
            search_fn: Callable(query, k) → list of (chunk_id, score) tuples.
            k: Number of results to evaluate.

        Returns:
            Dictionary of metric name → value.
        """
        with open(eval_dataset_path, encoding="utf-8") as f:
            dataset = json.load(f)

        results: list[RetrievalResult] = []
        for item in dataset:
            query = item["query"]
            expected_ids = item["expected_chunk_ids"]
            relevance_scores = item.get("relevance_scores", {})

            # Run search
            search_results = search_fn(query, k)
            retrieved_ids = [r[0] for r in search_results]
            retrieved_scores = [r[1] for r in search_results]

            results.append(
                RetrievalResult(
                    query=query,
                    retrieved_ids=retrieved_ids,
                    retrieved_scores=retrieved_scores,
                    expected_ids=expected_ids,
                    relevance_scores=relevance_scores,
                )
            )

        return RetrievalMetrics.compute_all(results, k)

    def run_rag_evaluation(
        self,
        eval_dataset_path: str,
        search_fn,
        answer_fn=None,
        k: int = 5,
    ) -> dict[str, float]:
        """Run RAG-specific metrics.

        Args:
            eval_dataset_path: Path to JSON evaluation dataset.
            search_fn: Callable(query, k) → list of (chunk_id, score, content) tuples.
            answer_fn: Callable(query, contexts) → answer string. If None, uses context only.
            k: Number of context chunks.

        Returns:
            Dictionary of metric name → value.
        """
        with open(eval_dataset_path, encoding="utf-8") as f:
            dataset = json.load(f)

        context_relevances = []
        answer_coverages = []
        faithfulness_scores = []

        for item in dataset:
            query = item["query"]
            expected_answer = item.get("expected_answer", "")

            # Run search
            search_results = search_fn(query, k)
            contexts = [r[2] for r in search_results if len(r) > 2]

            # Context relevance — heuristic, not LLM (see HeuristicOverlapEvaluator)
            cr = HeuristicOverlapEvaluator.context_relevance(query, contexts)
            context_relevances.append(cr)

            # Use expected answer if no answer function provided
            answer = expected_answer
            if answer_fn:
                answer = answer_fn(query, contexts)

            # Answer coverage / faithfulness — heuristic overlap
            ac = HeuristicOverlapEvaluator.answer_coverage(answer, contexts)
            answer_coverages.append(ac)
            fs = HeuristicOverlapEvaluator.faithfulness_score(answer, contexts)
            faithfulness_scores.append(fs)

        return {
            "context_relevance": (
                sum(context_relevances) / len(context_relevances)
                if context_relevances else 0.0
            ),
            "answer_coverage": (
                sum(answer_coverages) / len(answer_coverages)
                if answer_coverages else 0.0
            ),
            "faithfulness": (
                sum(faithfulness_scores) / len(faithfulness_scores)
                if faithfulness_scores else 0.0
            ),
        }
