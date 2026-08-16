"""Abstract base class for text embedders."""

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Interface that all embedder implementations must satisfy."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into dense vectors.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        Args:
            query: The query text to embed.

        Returns:
            Embedding vector for the query.
        """
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name or identifier of the underlying model."""
        ...
