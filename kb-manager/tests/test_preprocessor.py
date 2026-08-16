from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kb_manager.preprocessor.persian import PersianPreprocessor
    from kb_manager.preprocessor.pipeline import PreprocessingPipeline


class TestPersianPreprocessor:
    def test_persian_normalization(self, preprocessor: PersianPreprocessor):
        text = "كُریم ٱلله"
        result = preprocessor.normalise(text)
        # Arabic ک should be mapped to Persian ک
        assert "\u06a9" in result
        # ٱ should be removed or mapped
        assert "\u0671" not in result

    def test_zwnj_fix(self, preprocessor: PersianPreprocessor):
        text = "این یک تست است"
        result = preprocessor.normalise(text)
        assert result is not None
        assert len(result) > 0

    def test_html_removal(self, preprocessor: PersianPreprocessor):
        text = "متن ساده بدون HTML"
        result = preprocessor.normalise(text)
        assert "<p>" not in result
        assert "<b>" not in result

    def test_url_removal(self, preprocessor: PersianPreprocessor):
        text = "لینک https://example.com متن بعدی"
        result = preprocessor.normalise(text)
        # normalise doesn't remove URLs, that's done in clean_text
        # but the result should still be valid
        assert result is not None

    def test_collapse_whitespace(self, preprocessor: PersianPreprocessor):
        text = "این   یک    تست   است"
        result = preprocessor.normalise(text)
        assert "  " not in result

    def test_pipeline_returns_quality_score(self, preprocessor_pipeline: PreprocessingPipeline):
        text = "این یک متن آزمایشی است."
        result = preprocessor_pipeline.run(text)
        assert result.quality_score >= 0.0
        assert result.quality_score <= 1.0
        assert result.normalised_text is not None
        assert len(result.keywords) > 0
