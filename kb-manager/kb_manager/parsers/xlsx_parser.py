"""Excel XLSX parser with support for multiple KB schemas."""

import os
import re
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

from kb_manager.parsers.base import BaseParser, ParsedDocument


# Known schema column patterns
SCHEMA_A_COLUMNS = {
    "reason_code",
    "model_name",
    "model_id",
    "brief_explanation",
    "detailed_explanation",
    "reason_text",
    "improvement_suggestions",
    "bin_score",
    "bin_impact",
    "feature_score",
    "feature_impact",
    "bin_details",
    "keywords",
    "feature_name",
    "data_source",
}

SCHEMA_B_COLUMNS = {"question", "model", "briefanswer", "answer", "keyword"}

SCHEMA_C_COLUMNS = {
    "documentname",
    "title",
    "sectiontitle",
    "content",
    "type",
    "version",
    "author(s)",
    "heading",
    "keywords",
    "summary",
}


def _normalize_col(name: str) -> str:
    """Normalize column name for comparison."""
    return re.sub(r"[\s_\-]+", "", name.strip().lower())


def _detect_schema(headers: list[str]) -> str | None:
    """Detect which KB schema a sheet uses based on column names.

    Uses a threshold-based match: if >= 60% of a schema's expected columns
    are present, treat it as that schema.  This handles files that are
    missing one optional column (e.g. ``model`` in CRM Q&A).

    Args:
        headers: List of column header strings.

    Returns:
        Schema identifier string or None for generic format.
    """
    normalized = {_normalize_col(h) for h in headers if h}

    schemas = [
        ("reason_codes", SCHEMA_A_COLUMNS),
        ("crm_qa", SCHEMA_B_COLUMNS),
        ("articles", SCHEMA_C_COLUMNS),
    ]
    for name, required in schemas:
        overlap = len(normalized & required)
        if overlap >= len(required) * 0.6:
            return name
    return None


def _is_temp_file(file_path: str) -> bool:
    """Check if file is a temporary Excel lock file."""
    name = Path(file_path).name
    return name.startswith("~$")


class XlsxParser(BaseParser):
    """Parser for Excel XLSX files supporting multiple KB schemas.

    Handles three known schemas:
    - reason_codes: Reason code analysis with model scoring
    - crm_qa: CRM Q&A pairs with keywords
    - articles: Article documentation with sections and metadata

    Falls back to generic key-value parsing for unknown formats.
    """

    def can_parse(self, file_path: str) -> bool:
        """Check if the file is an Excel file that can be parsed.

        Args:
            file_path: Path to check.

        Returns:
            True if the file has an xlsx extension and is not a temp file.
        """
        if _is_temp_file(file_path):
            return False
        return Path(file_path).suffix.lower() == ".xlsx"

    def parse(self, file_path: str) -> ParsedDocument:
        """Parse an Excel file and extract content from all sheets.

        Args:
            file_path: Path to the XLSX file.

        Returns:
            ParsedDocument with sheets data and combined text content.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be opened or parsed.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if _is_temp_file(file_path):
            raise ValueError(f"Cannot parse temporary Excel file: {file_path}")

        abs_path = str(Path(file_path).resolve())
        title = Path(file_path).stem

        try:
            wb = load_workbook(abs_path, read_only=True, data_only=True)
        except Exception as e:
            raise ValueError(f"Failed to open Excel file: {e}") from e

        sheets_data: list[dict] = []
        all_content: list[str] = []

        try:
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                sheet_result = self._parse_sheet(ws, sheet_name)
                if sheet_result is not None:
                    sheets_data.append(sheet_result)
                    all_content.append(self._sheet_to_text(sheet_result))
        finally:
            wb.close()

        if not sheets_data:
            raise ValueError(f"No valid sheets found in {file_path}")

        combined_content = "\n\n".join(all_content)

        return ParsedDocument(
            source_path=abs_path,
            title=title,
            content=combined_content,
            file_type="xlsx",
            metadata={"sheet_count": len(sheets_data)},
            sheets=sheets_data,
            sections=None,
        )

    def _parse_sheet(self, ws: object, sheet_name: str) -> dict | None:
        """Parse a single worksheet and detect its schema.

        Args:
            ws: Openpyxl worksheet object.
            sheet_name: Name of the worksheet.

        Returns:
            Dictionary with sheet data or None if sheet is empty/invalid.
        """
        rows_iter = ws.iter_rows(values_only=True)
        try:
            header_row = next(rows_iter)
        except StopIteration:
            return None

        if header_row is None:
            return None

        headers = [str(h).strip() if h is not None else "" for h in header_row]

        # Filter out empty header columns at the end
        while headers and not headers[-1]:
            headers.pop()

        if len(headers) < 2:
            return None

        # Collect remaining rows
        data_rows: list[list[str]] = []
        for row in rows_iter:
            if row is None:
                continue
            values = []
            for i, cell in enumerate(row):
                if i >= len(headers):
                    break
                if cell is None:
                    values.append("")
                elif isinstance(cell, float):
                    # Preserve numeric precision
                    values.append(f"{cell:g}" if cell == int(cell) else str(cell))
                else:
                    values.append(str(cell).strip())
            # Skip fully empty rows
            if any(v for v in values):
                data_rows.append(values)

        if not data_rows:
            return None

        schema = _detect_schema(headers)

        return {
            "name": sheet_name,
            "headers": headers,
            "rows": data_rows,
            "schema": schema,
        }

    def _sheet_to_text(self, sheet_data: dict) -> str:
        """Convert sheet data to readable text format.

        Multi-line cell values are collapsed to a single line so that
        the pipe-delimited row format stays intact for the chunker.

        Args:
            sheet_data: Sheet dictionary with name, headers, rows, and schema.

        Returns:
            Formatted text representation of the sheet.
        """
        lines: list[str] = [f"=== Sheet: {sheet_data['name']} ==="]

        if sheet_data["schema"]:
            lines.append(f"[Schema: {sheet_data['schema']}]")
            lines.append("")

        headers = sheet_data["headers"]
        rows = sheet_data["rows"]

        for row in rows:
            parts: list[str] = []
            for header, value in zip(headers, row):
                if value and value.strip():
                    # Collapse multi-line cells into a single line
                    collapsed = value.strip().replace("\n", " ").replace("\r", "")
                    parts.append(f"{header}: {collapsed}")
            if parts:
                lines.append(" | ".join(parts))

        return "\n".join(lines)
