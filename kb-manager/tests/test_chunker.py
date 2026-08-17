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


class TestIncompleteQaFiltering:
    """Option A: incomplete QA rows (missing question or answer) are skipped."""

    @staticmethod
    def _sheets(headers: list[str], rows: list[list[str]]) -> list[dict]:
        return [
            {
                "name": "QA1",
                "schema": "crm_qa",
                "headers": headers,
                "rows": rows,
            }
        ]

    def test_incomplete_qa_rows_are_skipped(self, chunker: SemanticChunker):
        sheets = self._sheets(
            ["question", "answer", "keyword"],
            [
                ["سوال کامل", "پاسخ کامل", "کلیدواژه"],
                ["سوال بدون پاسخ", "", ""],
                ["", "پاسخ بدون سوال", ""],
            ],
        )
        chunks = chunker.chunk(
            "",
            metadata={"doc_type": "qa_pair", "sheets": sheets},
        )
        contents = [c.content for c in chunks if not c.metadata.get("is_parent", False)]

        # Complete row retained; both incomplete rows skipped.
        assert len(contents) == 1
        assert "سوال کامل" in contents[0]
        assert "پاسخ بدون سوال" not in chunks[0].content if chunks else True

    def test_all_incomplete_rows_yield_no_child_chunks(self, chunker: SemanticChunker):
        sheets = self._sheets(
            ["question", "answer"],
            [["سربرگ بدون سوال", ""], ["", "بدون پاسخ"]],
        )
        chunks = chunker.chunk(
            "",
            metadata={"doc_type": "qa_pair", "sheets": sheets},
        )
        children = [c for c in chunks if not c.metadata.get("is_parent", False)]
        assert children == []

    def test_complete_qa_row_is_kept(self, chunker: SemanticChunker):
        sheets = self._sheets(
            ["question", "answer", "keyword"],
            [["سوال", "پاسخ", "کلید"]],
        )
        chunks = chunker.chunk(
            "",
            metadata={"doc_type": "qa_pair", "sheets": sheets},
        )
        children = [c for c in chunks if not c.metadata.get("is_parent", False)]
        assert len(children) == 1
        assert "سوال" in children[0].content


class TestFixedChunker:
    def test_fixed_chunker_respects_max_tokens(self, fixed_chunker: FixedChunker):
        text = "کلمه " * 200
        chunks = fixed_chunker.chunk(text, metadata={"doc_id": "test-3"})

        for chunk in chunks:
            assert chunk.token_count <= 512 + 100  # max_tokens + overlap margin
