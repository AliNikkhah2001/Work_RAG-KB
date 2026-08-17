from __future__ import annotations

import re
from typing import Any

from .base import BaseChunker, Chunk, _estimate_tokens

# ---------------------------------------------------------------------------
# Structural marker patterns (Persian / bilingual documents)
# ---------------------------------------------------------------------------
_HEADING_RE = re.compile(
    r"^(#{1,6})\s+(.+)$",
    re.MULTILINE,
)
_ARTICLE_RE = re.compile(
    r"^[\u0645\u0627\u062F\u0647]\s*[\d\u06F0-\u06F9\u0660-\u0669]+",
    re.MULTILINE,
)
_SECTION_RE = re.compile(
    r"^([\u0628\u0634\u0631\u0637\u06CC][\u0647\u0627]?\s*[\d\u06F0-\u06F9\u0660-\u0669]+"
    r"|\u0641\u0635\u0644\s*[\d\u06F0-\u06F9\u0660-\u0669]+)",
    re.MULTILINE,
)
_QA_RE = re.compile(
    r"^[-*]?\s*(?:\u0633\u0624\u0627\u0644|\u0633\u0648\u0627\u0644|\u067E\u0631\u0633\u0634|Question)\s*[:\uFF1A]",
    re.MULTILINE | re.IGNORECASE,
)
_REASON_CODE_RE = re.compile(
    r"^(?:\u06A9\u062F\s*\u062F\u0644\u06CC\u0644|reason\s*code)\s*[:\u061B\uFF1A]",
    re.MULTILINE | re.IGNORECASE,
)
_SHEET_HEADER_RE = re.compile(
    r"^={3,}\s*(?:Sheet:)?\s*.+",
    re.MULTILINE,
)
_SCHEMA_MARKER_RE = re.compile(
    r"^\[Schema:.*\]",
    re.MULTILINE,
)
_PIPE_ROW_RE = re.compile(
    r"^([^|\n]+\|[^|\n]+(?:\|[^|\n]+)*)$",
    re.MULTILINE,
)
_SENTENCE_END_RE = re.compile(
    r"(?<=[.!\?])\s+",
)
_DOUBLE_NEWLINE = re.compile(r"\n\s*\n")
_SINGLE_NEWLINE = re.compile(r"\n")

_OVERLAP_TOKENS = 50


class SemanticChunker(BaseChunker):
    """Structure-aware chunker for Persian / bilingual legal documents.

    Multi-strategy chunking:
    * **reason_detail** - one reason code -> one chunk.
    * **qa_pair** - one Q&A -> one chunk.
    * **article** - split on article boundaries.
    * **section** - split on section/heading boundaries.
    * **body** - fallback: split on sheet headers, pipe rows, double newlines,
      sentence boundaries, then single newlines for oversized pieces.

    A heading path is prepended to every chunk to improve retrieval quality.
    Adjacent chunks share a configurable overlap.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        min_tokens: int = 64,
        overlap_tokens: int = _OVERLAP_TOKENS,
        parent_scope: str = "sheet",
        parent_max_tokens: int = 1536,
    ) -> None:
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        self.parent_scope = parent_scope
        self.parent_max_tokens = parent_max_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Chunk *text* using structural markers.

        If *metadata* contains ``"sheets"`` (raw XLSX row data) and a
        ``"doc_type"`` of ``qa_pair`` or ``reason_detail``, each row is
        turned into its own chunk — no splitting across QA pairs.
        """
        meta = metadata or {}
        doc_type = meta.get("doc_type", "body")
        sheets = meta.get("sheets")

        # If we have raw sheet data for Q&A or reason codes,
        # chunk by rows directly (never split a QA pair)
        if sheets and doc_type in ("qa_pair", "reason_detail"):
            return self._chunk_excel_rows(sheets, doc_type, meta)

        if doc_type == "reason_detail":
            return self._chunk_reason_codes(text, meta)
        if doc_type == "qa_pair":
            return self._chunk_qa_pairs(text, meta)

        return self._chunk_structural(text, meta)

    # ------------------------------------------------------------------
    # Strategy: reason codes
    # ------------------------------------------------------------------

    def _chunk_reason_codes(self, text: str, metadata: dict[str, Any]) -> list[Chunk]:
        parts = _split_on_pattern(text, _REASON_CODE_RE)
        return self._build_chunks(parts, "reason_detail", metadata)

    # ------------------------------------------------------------------
    # Strategy: Q&A pairs
    # ------------------------------------------------------------------

    def _chunk_qa_pairs(self, text: str, metadata: dict[str, Any]) -> list[Chunk]:
        parts = _split_on_pattern(text, _QA_RE)
        return self._build_chunks(parts, "qa_pair", metadata)

    # ------------------------------------------------------------------
    # Strategy: Excel rows (QA pairs / reason codes — never split)
    # ------------------------------------------------------------------

    # Persian field name mapping for content display
    _FIELD_NAMES_FA: dict[str, str] = {
        "question": "\u0633\u0648\u0627\u0644",
        "briefanswer": "\u067e\u0627\u0633\u062e \u06a9\u0648\u062a\u0627\u0647",
        "brief_answer": "\u067e\u0627\u0633\u062e \u06a9\u0648\u062a\u0627\u0647",
        "answer": "\u067e\u0627\u0633\u062e \u06a9\u0627\u0645\u0644",
        "keyword": "\u06a9\u0644\u06cc\u062f\u0648\u0627\u0698\u0647\u200c\u0647\u0627",
        "keywords": "\u06a9\u0644\u06cc\u062f\u0648\u0627\u0698\u0647\u200c\u0647\u0627",
        "model": "\u0645\u062f\u0644",
        "reason_code": "\u06a9\u062f \u062f\u0644\u06cc\u0644",
        "brief_explanation": "\u062a\u0648\u0636\u06cc\u062d \u06a9\u0648\u062a\u0627\u0647",
        "detailed_explanation": "\u062a\u0648\u0636\u06cc\u062d \u06a9\u0627\u0645\u0644",
    }

    def _chunk_excel_rows(
        self,
        sheets: list[dict],
        doc_type: str,
        metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Create one chunk per data row from raw XLSX sheet data.

        Each row becomes a single chunk.  For ``qa_pair`` rows, the content
        is formatted with Persian field names and newlines.  Incomplete rows
        (missing question or answer) are skipped with a warning.

        Parent chunks aggregate child chunks.  The ``parent_scope`` metadata
        key (falling back to the constructor default) selects:
        * ``"sheet"`` - one parent per worksheet.
        * ``"document"`` - one parent per document (all sheets combined).

        Children are linked to their parent via the ``parent_key`` metadata
        field (the sheet name for ``"sheet"`` scope, ``"document"`` otherwise).

        Returns both child and parent chunks.
        """
        chunks: list[Chunk] = []
        parent_chunks: list[Chunk] = []
        ordinal = 0
        skipped_incomplete = 0
        parent_scope = metadata.get("parent_scope") or self.parent_scope
        if parent_scope not in ("sheet", "document"):
            parent_scope = "sheet"

        # Track per-sheet child groups so parents can be built for either scope.
        sheet_groups: list[tuple[str, str, list[Chunk]]] = []

        for sheet in sheets:
            headers = sheet.get("headers", [])
            rows = sheet.get("rows", [])
            sheet_name = sheet.get("name", "")
            schema = sheet.get("schema", "")
            sheet_children: list[Chunk] = []

            for row in rows:
                # Build field map
                fields: dict[str, str] = {}
                for header, value in zip(headers, row):
                    if value and str(value).strip():
                        collapsed = str(value).strip().replace("\n", " ")
                        fields[header.lower()] = collapsed

                # --- Phase 1: Filter incomplete QA rows ---
                if doc_type == "qa_pair":
                    has_question = any(
                        k in fields for k in ("question",)
                    )
                    has_answer = any(
                        k in fields
                        for k in ("answer", "briefanswer", "brief_answer")
                    )
                    # A QA row is incomplete if it is missing the question or
                    # the answer.  Such rows are skipped (never serve a user),
                    # and a warning is logged with the count.
                    if not has_question or not has_answer:
                        skipped_incomplete += 1
                        continue

                if not fields:
                    continue

                # --- Phase 2: Structured format with Persian field names ---
                if doc_type in ("qa_pair", "reason_detail"):
                    content = self._format_qa_content(fields, schema)
                else:
                    content = " | ".join(
                        f"{h}: {v}" for h, v in zip(headers, row) if v and str(v).strip()
                    )

                if not content.strip():
                    continue

                chunk = Chunk(
                    content=content,
                    ordinal=ordinal,
                    chunk_type=doc_type,
                    heading_path=f"Sheet: {sheet_name}" if sheet_name else "",
                    keywords=fields.get("keyword", "").split("\u060c")
                    if "keyword" in fields
                    else [],
                    token_count=_estimate_tokens(content),
                    metadata={
                        **metadata,
                        "sheet_name": sheet_name,
                        "schema": schema,
                        "fields": fields,
                    },
                )
                chunks.append(chunk)
                sheet_children.append(chunk)
                ordinal += 1

            sheet_groups.append((sheet_name, schema, sheet_children))

        if skipped_incomplete:
            import logging
            logging.getLogger(__name__).warning(
                "Skipped %d incomplete QA rows (missing question or answer)",
                skipped_incomplete,
            )

        # --- Phase 3: Create parent chunks per configured scope ---
        if parent_scope == "document":
            all_children: list[Chunk] = []
            for _, _, sheet_children in sheet_groups:
                all_children.extend(sheet_children)
            if all_children:
                parent_content = "\n\n".join(c.content for c in all_children)
                parent_chunk = Chunk(
                    content=parent_content,
                    ordinal=0,
                    chunk_type=f"{doc_type}_parent",
                    heading_path="",
                    keywords=[],
                    token_count=_estimate_tokens(parent_content),
                    metadata={
                        **metadata,
                        "is_parent": True,
                        "parent_scope": "document",
                        "parent_key": "document",
                        "child_count": len(all_children),
                    },
                )
                parent_chunks.append(parent_chunk)
                for child in all_children:
                    child.metadata["parent_key"] = "document"
        else:
            for sheet_name, schema, sheet_children in sheet_groups:
                if not sheet_children:
                    continue
                parent_content = "\n\n".join(c.content for c in sheet_children)
                parent_chunk = Chunk(
                    content=parent_content,
                    ordinal=0,
                    chunk_type=f"{doc_type}_parent",
                    heading_path=f"Sheet: {sheet_name}" if sheet_name else "",
                    keywords=[],
                    token_count=_estimate_tokens(parent_content),
                    metadata={
                        **metadata,
                        "sheet_name": sheet_name,
                        "schema": schema,
                        "is_parent": True,
                        "parent_scope": "sheet",
                        "parent_key": sheet_name,
                        "child_count": len(sheet_children),
                    },
                )
                parent_chunks.append(parent_chunk)

                # Link children to this parent
                for child in sheet_children:
                    child.metadata["parent_key"] = sheet_name

        if skipped_incomplete:
            import logging
            logging.getLogger(__name__).warning(
                "Skipped %d incomplete QA rows (missing answer field)",
                skipped_incomplete,
            )

        # Return both child and parent chunks
        all_chunks = chunks + parent_chunks
        return self._apply_overlap(all_chunks)

    def _apply_overlap(self, chunks: list[Chunk]) -> list[Chunk]:
        """Apply overlap between consecutive chunks.
        
        Only applies overlap to non-parent chunks.
        """
        if self.overlap_tokens <= 0 or len(chunks) < 2:
            return chunks

        enriched: list[Chunk] = [chunks[0]]
        for i in range(1, len(chunks)):
            # Skip overlap for parent chunks
            if chunks[i].metadata.get("is_parent"):
                enriched.append(chunks[i])
                continue
                
            prev_words = chunks[i - 1].content.split()
            overlap_word_count = max(1, self.overlap_tokens // 2)
            overlap_text = " ".join(prev_words[-overlap_word_count:])
            new_content = f"...\u200c{overlap_text}\u200c...\n{chunks[i].content}"
            enriched.append(
                Chunk(
                    content=new_content,
                    ordinal=chunks[i].ordinal,
                    chunk_type=chunks[i].chunk_type,
                    heading_path=chunks[i].heading_path,
                    keywords=list(chunks[i].keywords),
                    token_count=_estimate_tokens(new_content),
                    metadata=dict(chunks[i].metadata),
                )
            )
        return enriched

    def _format_qa_content(self, fields: dict[str, str], schema: str) -> str:
        """Format a QA/reason-code row with Persian field names and newlines."""
        lines: list[str] = []

        # Determine field order based on schema
        if schema == "reason_code":
            field_order = [
                ("reason_code", "\u06a9\u062f \u062f\u0644\u06cc\u0644"),
                ("model", "\u0645\u062f\u0644"),
                ("brief_explanation", "\u062a\u0648\u0636\u06cc\u062d \u06a9\u0648\u062a\u0627\u0647"),
                ("detailed_explanation", "\u062a\u0648\u0636\u06cc\u062d \u06a9\u0627\u0645\u0644"),
            ]
        else:
            field_order = [
                ("question", "\u0633\u0648\u0627\u0644"),
                ("briefanswer", "\u067e\u0627\u0633\u062e \u06a9\u0648\u062a\u0627\u0647"),
                ("brief_answer", "\u067e\u0627\u0633\u062e \u06a9\u0648\u062a\u0627\u0647"),
                ("answer", "\u067e\u0627\u0633\u062e \u06a9\u0627\u0645\u0644"),
                ("keyword", "\u06a9\u0644\u06cc\u062f\u0648\u0627\u0698\u0647\u200c\u0647\u0627"),
                ("keywords", "\u06a9\u0644\u06cc\u062f\u0648\u0627\u0698\u0647\u200c\u0647\u0627"),
            ]

        for key, label in field_order:
            if key in fields:
                lines.append(f"{label}: {fields[key]}")

        # Add any remaining fields not in the predefined order
        seen = {k for k, _ in field_order}
        for key, value in fields.items():
            if key not in seen:
                label = self._FIELD_NAMES_FA.get(key, key)
                lines.append(f"{label}: {value}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Strategy: structural (articles, sections, headings, pipe rows)
    # ------------------------------------------------------------------

    def _chunk_structural(self, text: str, metadata: dict[str, Any]) -> list[Chunk]:
        # Level 1: Try article boundaries
        if _ARTICLE_RE.search(text):
            sections = _split_on_pattern(text, _ARTICLE_RE)
        # Level 2: Try section/portion boundaries
        elif _SECTION_RE.search(text):
            sections = _split_on_pattern(text, _SECTION_RE)
        # Level 3: Try markdown headings
        elif _HEADING_RE.search(text):
            sections = _split_on_pattern(text, _HEADING_RE)
        # Level 4: Try sheet headers (=== Sheet: ...) and schema markers
        elif _SHEET_HEADER_RE.search(text):
            sections = _split_on_pattern(text, _SHEET_HEADER_RE)
        else:
            sections = [text]

        # Level 5: Further split oversized sections on pipe rows, double newlines
        refined: list[str] = []
        for section in sections:
            if _estimate_tokens(section) > self.max_tokens:
                refined.extend(self._split_fine_grained(section))
            else:
                refined.append(section)

        # Level 6: Final pass - split anything still too large on sentences/newlines
        final: list[str] = []
        for piece in refined:
            if _estimate_tokens(piece) > self.max_tokens:
                final.extend(self._split_emergency(piece))
            else:
                final.append(piece)

        # Merge tiny chunks with the next one
        merged = _merge_short(final, self.min_tokens)

        return self._build_chunks(merged, "body", metadata)

    def _split_fine_grained(self, text: str) -> list[str]:
        """Split oversized text using pipe rows, double newlines, then schema markers."""
        # Try pipe-delimited rows first
        if _PIPE_ROW_RE.search(text) and text.count("|") >= 3:
            parts = _split_on_pattern(text, _PIPE_ROW_RE)
            if len(parts) > 1:
                return parts

        # Try schema markers
        if _SCHEMA_MARKER_RE.search(text):
            parts = _split_on_pattern(text, _SCHEMA_MARKER_RE)
            if len(parts) > 1:
                return parts

        # Double newlines
        parts = _split_on_pattern(text, _DOUBLE_NEWLINE)
        if len(parts) > 1:
            return parts

        # Single newlines
        parts = _split_on_pattern(text, _SINGLE_NEWLINE)
        if len(parts) > 1:
            return parts

        return [text]

    def _split_emergency(self, text: str) -> list[str]:
        """Last resort: split on sentence boundaries, then hard split."""
        sentences = _SENTENCE_END_RE.split(text)
        if len(sentences) > 1:
            # Group sentences into chunks of max_tokens
            chunks = []
            current = ""
            for sent in sentences:
                if _estimate_tokens(current + " " + sent) > self.max_tokens:
                    if current:
                        chunks.append(current.strip())
                    current = sent
                else:
                    current = f"{current} {sent}".strip() if current else sent
            if current:
                chunks.append(current.strip())
            return chunks if chunks else [text]

        # Hard split by character count
        max_chars = self.max_tokens * 4  # rough estimate
        parts = []
        for i in range(0, len(text), max_chars):
            parts.append(text[i : i + max_chars])
        return parts

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_chunks(
        self,
        texts: list[str],
        chunk_type: str,
        metadata: dict[str, Any],
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for idx, content in enumerate(texts):
            content = content.strip()
            if not content:
                continue
            heading_path = _extract_heading_path(content)
            chunk = Chunk(
                content=content,
                ordinal=idx,
                chunk_type=chunk_type,
                heading_path=heading_path,
                metadata=dict(metadata),
            )
            chunks.append(chunk)
        return self._apply_overlap(chunks)


# ======================================================================
# Module-level helpers
# ======================================================================


def _split_on_pattern(text: str, pattern: re.Pattern[str]) -> list[str]:  # type: ignore[type-arg]
    parts: list[str] = []
    last_end = 0
    for match in pattern.finditer(text):
        start = match.start()
        if start > last_end:
            parts.append(text[last_end:start])
        last_end = start
    if last_end < len(text):
        parts.append(text[last_end:])
    return [p for p in parts if p.strip()]


def _merge_short(parts: list[str], min_tokens: int) -> list[str]:
    if not parts:
        return []
    merged: list[str] = []
    buffer = parts[0]
    for part in parts[1:]:
        if _estimate_tokens(buffer) < min_tokens:
            buffer = f"{buffer}\n\n{part}"
        else:
            merged.append(buffer)
            buffer = part
    merged.append(buffer)
    return merged


def _extract_heading_path(text: str) -> str:
    match = _HEADING_RE.search(text)
    if match:
        return match.group(2).strip()
    return ""
