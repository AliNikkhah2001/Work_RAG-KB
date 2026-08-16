"""Base classes and data models for KB document parsers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ParsedDocument:
    """Represents a parsed document with extracted content and metadata.

    Attributes:
        source_path: Absolute path to the source file.
        title: Document title derived from filename or content.
        content: Full extracted text content.
        file_type: File extension identifier (xlsx, pdf, docx).
        metadata: Additional document metadata (author, dates, etc.).
        sheets: For XLSX files, list of sheets with name, headers, and rows.
        sections: For DOCX/PDF files, list of sections with heading and text.
    """

    source_path: str
    title: str
    content: str
    file_type: str
    metadata: dict = field(default_factory=dict)
    sheets: list[dict] | None = None
    sections: list[dict] | None = None


class BaseParser(ABC):
    """Abstract base class for all document parsers.

    Subclasses must implement parse() to extract content and
    can_parse() to identify supported file types.
    """

    @abstractmethod
    def parse(self, file_path: str) -> ParsedDocument:
        """Parse the document at the given path and return extracted content.

        Args:
            file_path: Path to the document file.

        Returns:
            ParsedDocument with extracted content and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be parsed.
        """
        ...

    @abstractmethod
    def can_parse(self, file_path: str) -> bool:
        """Check whether this parser can handle the given file.

        Args:
            file_path: Path to the document file.

        Returns:
            True if the file can be parsed by this parser.
        """
        ...
