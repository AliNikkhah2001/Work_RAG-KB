from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.conftest import MockEmbedder


class TestEmbedder:
    def test_embed_texts_returns_correct_dimensions(self, embedder: MockEmbedder):
        texts = ["سلام دنیا", "测试文本", "test text"]
        vectors = embedder.embed_texts(texts)

        assert len(vectors) == 3
        for vec in vectors:
            assert len(vec) == 384

    def test_embed_query_returns_vector(self, embedder: MockEmbedder):
        vec = embedder.embed_query("این یک پرسش است")

        assert isinstance(vec, list)
        assert len(vec) == 384

    def test_content_hash_caching(self, embedder: MockEmbedder):
        text = "متن تکراری"
        v1 = embedder.embed_texts([text])[0]
        v2 = embedder.embed_texts([text])[0]

        assert v1 == v2
