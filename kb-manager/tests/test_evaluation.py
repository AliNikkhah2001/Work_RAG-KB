from __future__ import annotations

import pytest

from kb_manager.evaluation.metrics import (
    RanxRetrievalEvaluator,
    RetrievalMetrics,
    RetrievalResult,
)


def _make_results() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            query="سوال یک",
            retrieved_ids=["d1", "d2", "d3"],
            retrieved_scores=[0.9, 0.8, 0.1],
            expected_ids=["d1", "d2"],
            relevance_scores={"d1": 1.0, "d2": 1.0, "d3": 0.0},
        ),
        RetrievalResult(
            query="سوال دو",
            retrieved_ids=["d3", "d1", "d2"],
            retrieved_scores=[0.7, 0.4, 0.2],
            expected_ids=["d1"],
            relevance_scores={"d1": 1.0, "d2": 0.0, "d3": 0.0},
        ),
    ]


class TestRanxRetrievalEvaluator:
    def test_compute_all_matches_pure_python(self):
        results = _make_results()
        if not RanxRetrievalEvaluator.available():
            pytest.skip("ranx not installed")

        ranx_metrics = RanxRetrievalEvaluator.compute_all(results, k=10)
        pure_metrics = RetrievalMetrics.compute_all(results, k=10)

        for key in ("map@10", "recall@10", "ndcg@10"):
            assert ranx_metrics[key] == pytest.approx(pure_metrics[key], abs=1e-6)

    def test_compute_all_empty(self):
        if not RanxRetrievalEvaluator.available():
            pytest.skip("ranx not installed")
        metrics = RanxRetrievalEvaluator.compute_all([], k=10)
        assert set(metrics) == {"map@10", "mrr@10", "ndcg@10", "recall@10", "precision@10"}

    def test_compute_all_no_relevance(self):
        if not RanxRetrievalEvaluator.available():
            pytest.skip("ranx not installed")
        results = [
            RetrievalResult(
                query="q", retrieved_ids=["a", "b"], retrieved_scores=[0.5, 0.4],
                expected_ids=[], relevance_scores={},
            )
        ]
        metrics = RanxRetrievalEvaluator.compute_all(results, k=10)
        assert all(v == 0.0 for v in metrics.values())

    def test_falls_back_when_ranx_absent(self, monkeypatch: pytest.MonkeyPatch):
        import builtins

        import kb_manager.evaluation.metrics as metrics_mod

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "ranx" or name.startswith("ranx."):
                raise ImportError("no ranx")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        metrics_mod.RanxRetrievalEvaluator._ranx_available = None

        results = _make_results()
        metrics = RanxRetrievalEvaluator.compute_all(results, k=10)
        assert set(metrics) == {
            "precision@10", "recall@10", "hit_rate@10", "mrr", "ndcg@10", "map@10",
        }

        metrics_mod.RanxRetrievalEvaluator._ranx_available = None


class TestRagasEvaluator:
    def test_available(self):
        from kb_manager.evaluation.ragas_metrics import RagasEvaluator

        # Either True or False; the point is it must not raise.
        assert isinstance(RagasEvaluator.available(), bool)

    def test_evaluate_empty_input_no_imports(self):
        from kb_manager.evaluation.ragas_metrics import RagasEvaluator

        result = RagasEvaluator().evaluate([], [], [])
        assert result == {}

    def test_unknown_metrics_resolve_to_empty(self):
        from kb_manager.config import RagasConfig
        from kb_manager.evaluation.ragas_metrics import RagasEvaluator

        evaluator = RagasEvaluator(
            RagasConfig(metrics=("not_a_real_metric",))
        )
        assert evaluator._resolve_metrics() == []
