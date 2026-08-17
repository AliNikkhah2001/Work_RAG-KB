from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kb_manager.chunker import FixedChunker, SemanticChunker


class TestSemanticChunker:
    def test_semantic_chunker_on_article(self, chunker: SemanticChunker):
        text = "## مقدمه\nاین بخش مقدمه است.\n\n## بدنه\nمحتوای اصلی اینجا نوشته می‌شود. " * 20
        chunks = chunker.chunk(text, metadata={"doc_id": "test-1"})

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.content
            assert chunk.heading_path is not None

    def test_semantic_chunker_single_entity(self, chunker: SemanticChunker):
        text = "فقط یک خط متن."
        chunks = chunker.chunk(text, metadata={"doc_id": "test-2"})

        assert len(chunks) == 1
        assert chunks[0].ordinal == 0

    def test_chunks_have_heading_path(self, chunker: SemanticChunker):
        text = "## عنوان\nمحتوا"
        chunks = chunker.chunk(text, metadata={"doc_id": "test-4"})

        assert all(c.heading_path is not None for c in chunks)


class TestContentFormat:
    """Content uses clean newlines with Persian field names (no pipes)."""

    @staticmethod
    def _chunk_row(
        chunker: SemanticChunker,
        headers: list[str],
        row: list[str],
        schema: str = "crm_qa",
    ) -> str:
        sheets = [
            {
                "name": "QA1",
                "schema": schema,
                "headers": headers,
                "rows": [row],
            }
        ]
        chunks = chunker.chunk(
            "",
            metadata={"doc_type": "qa_pair", "sheets": sheets},
        )
        children = [c for c in chunks if not c.metadata.get("is_parent", False)]
        assert children
        return children[0].content

    def test_qa_content_uses_newlines_not_pipes(self, chunker: SemanticChunker):
        content = self._chunk_row(
            chunker,
            ["question", "answer", "keyword"],
            ["چه زمانی", "پاسخ نمونه", "کلید"],
        )
        assert "\n" in content
        assert "|" not in content

    def test_qa_content_has_persian_field_labels(self, chunker: SemanticChunker):
        content = self._chunk_row(
            chunker,
            ["question", "briefanswer", "keyword"],
            ["چه چیزی", "خلاصه", "ک"],
        )
        assert "سوال" in content
        assert "پاسخ کوتاه" in content
        assert "کلیدواژه" in content

    def test_reason_code_content_has_persian_labels(self, chunker: SemanticChunker):
        content = self._chunk_row(
            chunker,
            ["reason_code", "model", "detailed_explanation"],
            ["RC1", "M1", "توضیح کامل"],
            schema="reason_code",
        )
        assert "کد دلیل" in content
        assert "مدل" in content
        assert "توضیح کامل" in content

    def test_extra_fields_are_appended(self, chunker: SemanticChunker):
        content = self._chunk_row(
            chunker,
            ["question", "answer", "custom_field"],
            ["س", "پ", "مقدار سفارشی"],
        )
        assert "custom_field" in content
        assert "مقدار سفارشی" in content


class TestFixedChunker:
    def test_fixed_chunker_respects_max_tokens(self, fixed_chunker: FixedChunker):
        text = "کلمه " * 200
        chunks = fixed_chunker.chunk(text, metadata={"doc_id": "test-3"})

        for chunk in chunks:
            assert chunk.token_count <= 512 + 100  # max_tokens + overlap margin
