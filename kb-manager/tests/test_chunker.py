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


class TestFixedChunker:
    def test_fixed_chunker_respects_max_tokens(self, fixed_chunker: FixedChunker):
        text = "کلمه " * 200
        chunks = fixed_chunker.chunk(text, metadata={"doc_id": "test-3"})

        for chunk in chunks:
            assert chunk.token_count <= 512 + 100  # max_tokens + overlap margin
