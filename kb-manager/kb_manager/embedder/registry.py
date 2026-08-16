"""Factory registry for embedder backends."""

from __future__ import annotations

import logging
from typing import Any

from kb_manager.embedder.base import BaseEmbedder

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[BaseEmbedder]] = {}


def _register_defaults() -> None:
    """Populate the registry with the built-in backends."""
    if not _REGISTRY:
        from kb_manager.embedder.sentence_transformer import (
            SentenceTransformerEmbedder,
        )

        _REGISTRY["sentence-transformer"] = SentenceTransformerEmbedder
        _REGISTRY["st"] = SentenceTransformerEmbedder  # short alias


def register_embedder(name: str, cls: type[BaseEmbedder]) -> None:
    """Register a custom embedder class under *name*.

    Args:
        name: Case-insensitive key used in :func:`get_embedder`.
        cls: Embedder class (must satisfy :class:`BaseEmbedder`).

    Raises:
        TypeError: If *cls* is not a subclass of :class:`BaseEmbedder`.
    """
    if not (isinstance(cls, type) and issubclass(cls, BaseEmbedder)):
        raise TypeError(f"{cls!r} is not a subclass of BaseEmbedder")
    _REGISTRY[name.lower()] = cls


def get_embedder(name: str = "sentence-transformer", **kwargs: Any) -> BaseEmbedder:
    """Return an embedder instance identified by *name*.

    Args:
        name: Registered embedder key (e.g. ``"sentence-transformer"`` or
            the shorthand ``"st"``).  Case-insensitive.
        **kwargs: Forwarded to the embedder constructor.

    Returns:
        A configured :class:`BaseEmbedder` instance.

    Raises:
        ValueError: If *name* is not registered.
    """
    _register_defaults()
    key = name.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown embedder {name!r}.  Available: {available}")
    cls = _REGISTRY[key]
    logger.debug("Creating embedder %r from %s", name, cls.__name__)
    return cls(**kwargs)
