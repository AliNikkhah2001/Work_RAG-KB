"""Preprocessing sub-package for the KB manager.

Exports
-------
- :class:`PersianPreprocessor` – Persian-specific text normalisation.
- :class:`PreprocessingPipeline` – Full pipeline (clean → normalise → keywords).
"""

from kb_manager.preprocessor.persian import PersianPreprocessor
from kb_manager.preprocessor.pipeline import PreprocessingPipeline

__all__ = ["PersianPreprocessor", "PreprocessingPipeline"]
