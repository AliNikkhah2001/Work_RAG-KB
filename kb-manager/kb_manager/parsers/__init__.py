"""KB document parsers package.

Provides parsing functionality for Excel, PDF, and DOCX files
with support for multiple document schemas.
"""

from kb_manager.parsers.base import BaseParser, ParsedDocument
from kb_manager.parsers.registry import get_parser

__all__ = ["BaseParser", "ParsedDocument", "get_parser"]
