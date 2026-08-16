"""PDF parser using PyMuPDF (fitz) for text extraction."""

import os
import re
from pathlib import Path

import fitz  # PyMuPDF

from kb_manager.parsers.base import BaseParser, ParsedDocument


# Arabic numeral suffix patterns for article detection
_ARTICLE_PATTERN = re.compile(
    r"(?:ماد[هة]|مادة|المادة|فقر[هة]|الفقر[هة]|بند|البند)\s*"
    r"(?:ال\s*)?\(?[٠-٩0-9]+\)?",
    re.UNICODE,
)


def _clean_text(text: str) -> str:
    """Clean extracted PDF text by normalizing whitespace.

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text with normalized whitespace.
    """
    # Replace multiple whitespace/newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _detect_sections(text: str) -> list[dict]:
    """Split text into sections based on article/paragraph markers.

    Detects Arabic article markers (ماده/فقره) to identify section boundaries.

    Args:
        text: Full page text.

    Returns:
        List of section dictionaries with heading and text.
    """
    sections: list[dict] = []
    matches = list(_ARTICLE_PATTERN.finditer(text))

    if not matches:
        return [{"heading": "", "text": text}]

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        heading = match.group().strip()

        sections.append({"heading": heading, "text": section_text})

    return sections


class PdfParser(BaseParser):
    """Parser for PDF documents using PyMuPDF.

    Extracts text from each page and detects article boundaries
    for structured documents (Arabic legal/technical content).
    """

    def can_parse(self, file_path: str) -> bool:
        """Check if the file is a PDF that can be parsed.

        Args:
            file_path: Path to check.

        Returns:
            True if the file has a .pdf extension.
        """
        return Path(file_path).suffix.lower() == ".pdf"

    def parse(self, file_path: str) -> ParsedDocument:
        """Parse a PDF file and extract text content.

        Args:
            file_path: Path to the PDF file.

        Returns:
            ParsedDocument with page-by-page content and detected sections.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the PDF cannot be opened or is empty.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        abs_path = str(Path(file_path).resolve())
        title = Path(file_path).stem

        try:
            doc = fitz.open(abs_path)
        except Exception as e:
            raise ValueError(f"Failed to open PDF file: {e}") from e

        pages_content: list[str] = []
        metadata: dict = {}

        try:
            # Extract document metadata
            doc_info = doc.metadata
            if doc_info:
                metadata = {
                    "title": doc_info.get("title", ""),
                    "author": doc_info.get("author", ""),
                    "subject": doc_info.get("subject", ""),
                    "page_count": doc.page_count,
                }

            # Extract text from each page
            for page_num in range(doc.page_count):
                try:
                    page = doc.load_page(page_num)
                    # Use text extraction with basic options
                    page_text = page.get_text("text")
                    cleaned = _clean_text(page_text)
                    if cleaned:
                        pages_content.append(cleaned)
                except Exception:
                    # Skip pages that cannot be read
                    continue
        finally:
            doc.close()

        if not pages_content:
            raise ValueError(f"No text content extracted from PDF: {file_path}")

        combined_text = "\n\n".join(pages_content)
        sections = _detect_sections(combined_text)

        return ParsedDocument(
            source_path=abs_path,
            title=title,
            content=combined_text,
            file_type="pdf",
            metadata=metadata,
            sheets=None,
            sections=sections,
        )
