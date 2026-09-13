"""Preprocessing sub-package for the KB manager.

Exports
-------
- :class:`PersianPreprocessor` – Persian-specific text normalisation.
- :class:`PreprocessingPipeline` – Full pipeline (clean → normalise → keywords).
"""

from kb_manager.preprocessor.persian import PersianPreprocessor
from kb_manager.preprocessor.pipeline import PreprocessingPipeline
from kb_manager.preprocessor.regex_persian import (
    build_persian_regex,
    has_mojibake,
    is_persian,
    normalize_digits,
    strip_diacritics,
)
from kb_manager.preprocessor.validators import extract_entities, validate_national_id, validate_sheba

__all__ = [
    "PersianPreprocessor",
    "PreprocessingPipeline",
    "build_persian_regex",
    "has_mojibake",
    "is_persian",
    "normalize_digits",
    "strip_diacritics",
    "extract_entities",
    "validate_national_id",
    "validate_sheba",
]
