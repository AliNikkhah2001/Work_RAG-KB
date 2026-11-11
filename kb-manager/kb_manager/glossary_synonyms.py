"""Build query-expansion synonym entries from the واژگان معادل glossary sheet.

Supplementary glossary workbooks are never vectorized. Each non-empty row is a
synonym cluster: every term expands to the other terms in the same row.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from kb_manager.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

GLOSSARY_MAP_PATH = PROJECT_ROOT / "data" / "synonym_map_glossary.json"
PACK_GLOSSARY_PATH = PROJECT_ROOT / "packs" / "synonyms" / "v1.0.0.json"


def clusters_from_sheets(sheets: list[dict[str, Any]]) -> list[list[str]]:
    """Return synonym clusters (one list of terms per non-empty data row)."""
    clusters: list[list[str]] = []
    for sheet in sheets or []:
        for row in sheet.get("rows") or []:
            terms: list[str] = []
            seen: set[str] = set()
            for cell in row:
                if cell is None:
                    continue
                term = str(cell).strip()
                if not term or term in seen:
                    continue
                seen.add(term)
                terms.append(term)
            if len(terms) >= 2:
                clusters.append(terms)
    return clusters


def clusters_to_map(clusters: list[list[str]]) -> dict[str, list[str]]:
    """Map each term to the other terms in its cluster (order preserved, unique)."""
    out: dict[str, list[str]] = {}
    for terms in clusters:
        for term in terms:
            others = [t for t in terms if t != term]
            prev = out.get(term, [])
            merged = list(dict.fromkeys([*prev, *others]))
            out[term] = merged
    return out


def _write_map(target: Path, merged: dict[str, list[str]]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def apply_glossary_sheets_to_map(sheets: list[dict[str, Any]], path: Path | None = None) -> int:
    """Write/merge glossary synonym map to disk. Returns number of keys written.

    Writes the runtime cache under ``data/`` and refreshes the versioned pack at
    ``packs/synonyms/v1.0.0.json`` so query expansion works without a prior ingest.
    """
    new_map = clusters_to_map(clusters_from_sheets(sheets))
    if not new_map:
        logger.warning("glossary sheets produced empty synonym map")
        return 0

    targets = [path] if path is not None else [GLOSSARY_MAP_PATH, PACK_GLOSSARY_PATH]
    last_count = 0
    for target in targets:
        existing: dict[str, list[str]] = {}
        if target.exists():
            try:
                existing = json.loads(target.read_text(encoding="utf-8"))
                if not isinstance(existing, dict):
                    existing = {}
            except Exception as exc:
                logger.warning("could not read existing glossary map %s: %s", target, exc)
                existing = {}
        merged: dict[str, list[str]] = dict(existing)
        for key, vals in new_map.items():
            prev = merged.get(key, [])
            merged[key] = list(dict.fromkeys([*(prev or []), *vals]))
        _write_map(target, merged)
        last_count = len(merged)
    return last_count


def load_glossary_map(path: Path | None = None) -> dict[str, list[str]]:
    """Load glossary synonym overlay; prefer versioned pack, then runtime cache."""
    candidates = [path] if path is not None else [PACK_GLOSSARY_PATH, GLOSSARY_MAP_PATH]
    for target in candidates:
        if target is None or not target.exists():
            continue
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception as exc:
            logger.warning("failed to load glossary synonym map %s: %s", target, exc)
    return {}
