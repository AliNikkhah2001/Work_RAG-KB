"""RAGAS integration for reference-free LLM-judged RAG evaluation.

Wraps the RAGAS library so that KB Manager can run faithfullness,
answer-relevancy, and context-recall metrics over retrieved chunks.
All imports of ragas / langchain are deferred so that importing this
module never fails when the optional dependencies are absent.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from kb_manager.config import RagasConfig

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_METRIC_MAP: dict[str, str] = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "context_recall": "context_recall",
    "context_precision": "context_precision",
}


class RagasEvaluator:
    """Evaluate RAG answers with RAGAS LLM metrics.

    Attributes:
        config: RagasConfig with model / key / metric choices.
    """

    def __init__(self, config: RagasConfig | None = None) -> None:
        self.config = config or RagasConfig()

    @staticmethod
    def available() -> bool:
        try:
            import langchain_openai  # noqa: F401
            import ragas  # noqa: F401
        except ImportError:
            return False
        return True

    def _build_llm(self):
        from langchain_openai.chat_models import ChatOpenAI

        kwargs: dict = {"model": self.config.llm_model}
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        return ChatOpenAI(**kwargs)

    def _build_embeddings(self):
        from langchain_openai.embeddings import OpenAIEmbeddings

        kwargs: dict = {"model": self.config.embedding_model}
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        return OpenAIEmbeddings(**kwargs)

    def _resolve_metrics(self):
        import ragas.metrics as rm

        selected: list = []
        for name in self.config.metrics:
            if name not in _METRIC_MAP:
                logger.warning("Unknown RAGAS metric %r, skipping", name)
                continue
            metric = getattr(rm, _METRIC_MAP[name])
            selected.append(metric)
        return selected

    def evaluate(
        self,
        questions: list[str],
        answers: list[str],
        retrieved_contexts: list[list[str]],
        ground_truth: list[str] | None = None,
    ) -> dict[str, float]:
        """Run RAGAS metrics over prepared RAG results.

        All four lists must be the same length (one entry per query).

        Args:
            questions: The user questions.
            answers: The generated (or reference) answers.
            retrieved_contexts: Per-query list of retrieved chunk texts.
            ground_truth: Optional reference answers for context_recall.

        Returns:
            Dict mapping ragas metric name → averaged score (0..1).
            Returns empty dict if ragas is unavailable.
        """
        if not self.available():
            logger.warning("ragas / langchain_openai not installed; RAGAS skipped")
            return {}
        if not (questions and answers and retrieved_contexts):
            logger.warning("Empty RAGAS evaluation input")
            return {}

        from datasets import Dataset
        from ragas import evaluate

        llm = self._build_llm()
        embeddings = self._build_embeddings()
        metrics = self._resolve_metrics()
        if not metrics:
            logger.warning("No RAGAS metrics resolved")
            return {}

        n = len(questions)
        data: dict = {
            "question": questions,
            "answer": answers,
            "contexts": retrieved_contexts,
        }
        if ground_truth and len(ground_truth) == n:
            data["ground_truth"] = ground_truth

        dataset = Dataset.from_dict(data)
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            show_progress=False,
        )

        df = result.to_pandas()
        scores: dict[str, float] = {}
        for metric in metrics:
            column = metric.name
            if column in df.columns:
                scores[column] = float(df[column].astype(float).mean())
        return scores

    @classmethod
    def evaluate_with_search(
        cls,
        config: RagasConfig,
        questions: list[str],
        search_fn: Callable[[str, int], list[tuple]],
        answer_fn: Callable[[str, list[str]], str],
        top_k: int = 5,
    ) -> dict[str, float]:
        """Convenience wrapper that runs search + answer generation per query."""
        evaluator = cls(config)

        retrieved_contexts: list[list[str]] = []
        answers: list[str] = []
        for question in questions:
            hits = search_fn(question, top_k)
            contexts = [h[2] for h in hits if len(h) > 2]
            retrieved_contexts.append(contexts)
            answers.append(answer_fn(question, contexts))

        return evaluator.evaluate(
            questions=questions,
            answers=answers,
            retrieved_contexts=retrieved_contexts,
        )
