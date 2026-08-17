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


class TestContentFormat:
    """Content uses clean newlines with Persian field names (no pipes)."""

    @staticmethod
    def _chunk_row(
        chunker: SemanticChunker,
        headers: list[str],
        row: list[str],
        schema: str = "crm_qa",
        doc_type: str = "qa_pair",
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
            metadata={"doc_type": doc_type, "sheets": sheets},
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
            doc_type="reason_detail",
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


class TestParentScope:
    """Parent chunks aggregate children per-sheet or per-document."""

    @staticmethod
    def _sheets() -> list[dict]:
        return [
            {
                "name": "s1",
                "schema": "crm_qa",
                "headers": ["question", "answer"],
                "rows": [["q1", "a1"], ["q2", "a2"]],
            },
            {
                "name": "s2",
                "schema": "crm_qa",
                "headers": ["question", "answer"],
                "rows": [["q3", "a3"]],
            },
        ]

    @staticmethod
    def _children(chunks):
        return [c for c in chunks if not c.metadata.get("is_parent", False)]

    @staticmethod
    def _parents(chunks):
        return [c for c in chunks if c.metadata.get("is_parent", False)]

    def test_sheet_scope_one_parent_per_sheet(self, chunker: SemanticChunker):
        chunker.parent_scope = "sheet"
        chunks = chunker.chunk("", metadata={"doc_type": "qa_pair", "sheets": self._sheets()})
        parents = self._parents(chunks)
        children = self._children(chunks)

        assert len(parents) == 2
        assert {p.metadata["parent_key"] for p in parents} == {"s1", "s2"}
        assert len(children) == 3
        # Each child links to its sheet's parent key.
        assert {c.metadata["parent_key"] for c in children} == {"s1", "s2"}

    def test_document_scope_one_parent_for_document(self, chunker: SemanticChunker):
        chunker.parent_scope = "document"
        chunks = chunker.chunk("", metadata={"doc_type": "qa_pair", "sheets": self._sheets()})
        parents = self._parents(chunks)
        children = self._children(chunks)

        assert len(parents) == 1
        assert parents[0].metadata["parent_key"] == "document"
        assert len(children) == 3
        # Every child is linked to the single document parent.
        assert all(c.metadata["parent_key"] == "document" for c in children)

    def test_parent_scope_from_metadata_overrides(self, chunker: SemanticChunker):
        chunker.parent_scope = "sheet"
        chunks = chunker.chunk(
            "",
            metadata={"doc_type": "qa_pair", "sheets": self._sheets(), "parent_scope": "document"},
        )
        parents = self._parents(chunks)
        assert len(parents) == 1
        assert parents[0].metadata["parent_key"] == "document"

    def test_parent_chunks_aggregate_child_content(self, chunker: SemanticChunker):
        chunker.parent_scope = "sheet"
        chunker.overlap_tokens = 0
        chunks = chunker.chunk("", metadata={"doc_type": "qa_pair", "sheets": self._sheets()})
        parents = self._parents(chunks)
        s1_parent = next(p for p in parents if p.metadata["parent_key"] == "s1")
        assert "q1" in s1_parent.content
        assert "q2" in s1_parent.content
        assert "q3" not in s1_parent.content


class TestFixedChunker:
    def test_fixed_chunker_respects_max_tokens(self, fixed_chunker: FixedChunker):
        text = "کلمه " * 200
        chunks = fixed_chunker.chunk(text, metadata={"doc_id": "test-3"})

        for chunk in chunks:
            assert chunk.token_count <= 512 + 100  # max_tokens + overlap margin
