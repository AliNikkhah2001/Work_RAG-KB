"""Retrieval-training scaffolding (lightweight, no heavy compute).

Stage order (each stage is a module in this package):

1. :mod:`kb_manager.retrieval_training.mining` -- mine hard-negative
   candidates (Tier1..Tier4) from the frozen retrieval pipeline.
2. :mod:`kb_manager.retrieval_training.validation` -- drop false negatives
   and incomplete QA rows; mark ``validation_status``.
3. :mod:`kb_manager.retrieval_training.dataset` -- build versioned
   train/val/test splits (:class:`DatasetSplit`) from mined negatives.
4. :mod:`kb_manager.retrieval_training.reranker_training` -- fine-tune the
   cross-encoder reranker (contract only; heavy compute lives with later agents).
5. :mod:`kb_manager.retrieval_training.dense_training` -- fine-tune the
   bi-encoder / dense index (contract only).
6. :mod:`kb_manager.retrieval_training.evaluation` -- score runs with the
   existing IR metrics (P/R/Hit/MRR/nDCG/MAP) against the frozen baseline.
7. :mod:`kb_manager.retrieval_training.fusion` -- RRF / score-fusion
   ablations over BM25 + dense + reranker legs (contract only).

Shared typed records live in :mod:`kb_manager.retrieval_training.schemas`.
The CLI entry point is :mod:`kb_manager.retrieval_training.cli`::

    python -m kb_manager.retrieval_training.cli --help

This package must stay import-light: no ``torch`` / ``transformers`` /
``sentence-transformers`` imports at module scope.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
