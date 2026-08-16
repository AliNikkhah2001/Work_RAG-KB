"""Parser registry with auto-detection and selection."""

import os
from pathlib import Path

from kb_manager.parsers.base import BaseParser
from kb_manager.parsers.docx_parser import DocxParser
from kb_manager.parsers.pdf_parser import PdfParser
from kb_manager.parsers.xlsx_parser import XlsxParser


# Supported file extensions to parser mapping
_EXTENSION_PARSERS: dict[str, type[BaseParser]] = {
    ".xlsx": XlsxParser,
    ".pdf": PdfParser,
    ".docx": DocxParser,
}

# Cache parser instances for reuse
_parser_instances: dict[str, BaseParser] = {}


def get_parser(file_path: str) -> BaseParser:
    """Get the appropriate parser for a given file path.

    Auto-detects file type based on extension and returns
    the matching parser instance.

    Args:
        file_path: Path to the file to parse.

    Returns:
        Parser instance capable of handling the file type.

    Raises:
        ValueError: If the file type is not supported.
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = Path(file_path).suffix.lower()

    if ext not in _EXTENSION_PARSERS:
        supported = ", ".join(sorted(_EXTENSION_PARSERS.keys()))
        raise ValueError(f"Unsupported file type '{ext}'. Supported types: {supported}")

    parser_class = _EXTENSION_PARSERS[ext]
    cache_key = parser_class.__name__

    if cache_key not in _parser_instances:
        _parser_instances[cache_key] = parser_class()

    return _parser_instances[cache_key]


def register_parser(extension: str, parser_class: type[BaseParser]) -> None:
    """Register a custom parser for a file extension.

    Args:
        extension: File extension (with dot, e.g., '.md').
        parser_class: Parser class implementing BaseParser.

    Raises:
        TypeError: If parser_class does not implement BaseParser.
        ValueError: If extension format is invalid.
    """
    if not extension.startswith(".") or len(extension) < 2:
        raise ValueError(f"Invalid extension format: {extension!r}. Must start with '.'.")

    if not issubclass(parser_class, BaseParser):
        raise TypeError(
            f"parser_class must be a subclass of BaseParser, got {parser_class.__name__}"
        )

    _EXTENSION_PARSERS[extension.lower()] = parser_class
    # Clear cached instance if exists
    _parser_instances.pop(parser_class.__name__, None)


def get_supported_extensions() -> list[str]:
    """Get list of all supported file extensions.

    Returns:
        Sorted list of supported extensions.
    """
    return sorted(_EXTENSION_PARSERS.keys())
