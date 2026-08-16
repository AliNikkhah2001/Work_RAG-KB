from __future__ import annotations

from typing import Any

from .base import BaseChunker
from .fixed import FixedChunker
from .semantic import SemanticChunker

# ---------------------------------------------------------------------------
# Strategy → chunker mapping
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[BaseChunker]] = {
    "semantic": SemanticChunker,
    "fixed": FixedChunker,
}


def get_chunker(
    strategy: str,
    **kwargs: Any,
) -> BaseChunker:
    """Return a chunker instance for the given *strategy* name.

    Args:
        strategy: One of the registered strategy names
                  (``"semantic"``, ``"fixed"``).
        **kwargs: Keyword arguments forwarded to the chunker constructor.

    Returns:
        A concrete :class:`BaseChunker` implementation.

    Raises:
        ValueError: If *strategy* is not registered.
    """
    cls = _REGISTRY.get(strategy.lower().strip())
    if cls is None:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown chunker strategy {strategy!r}. Available strategies: {available}"
        )
    return cls(**kwargs)


def register_chunker(name: str, cls: type[BaseChunker]) -> None:
    """Register a custom chunker class under *name*.

    Args:
        name: Strategy name to register.
        cls: Chunker class to associate with *name*.

    Raises:
        TypeError: If *cls* is not a subclass of :class:`BaseChunker`.
    """
    if not (isinstance(cls, type) and issubclass(cls, BaseChunker)):
        raise TypeError(f"Expected a subclass of BaseChunker, got {cls!r}")
    _REGISTRY[name.lower().strip()] = cls
