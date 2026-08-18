"""KB version snapshots.

A version snapshot is an immutable, physically-stored archive of a KB state:
a compact export of documents + chunks (without the 2.6GB duplicate ``sheets``
metadata blobs), the test dataset, benchmark results, IR metric report, plots,
and a manifest recording git commit, counts, and duplication stats.

Snapshots live under ``<repo>/versions/<label>/`` so they are versioned and
can be pushed to GitHub.
"""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from kb_manager.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

VERSIONS_ROOT = PROJECT_ROOT / "versions"
DB_PATH = PROJECT_ROOT / "data" / "kb_test.db"


# ---------------------------------------------------------------------------
# Compact DB export
# ---------------------------------------------------------------------------

def _git_head() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def export_compact(db_path: str) -> dict:
    """Export documents + a compact representation of chunks.

    QA chunks are fully included (question, answer, keywords); body chunks
    are reduced to metadata + a content preview. The huge ``sheets`` metadata
    blobs are excluded to keep the archive small.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    docs = []
    for row in cur.execute(
        "SELECT id, title, source_path, domain, category, status, chunk_count "
        "FROM documents ORDER BY id"
    ):
        docs.append(dict(row))

    qa_chunks = []
    body_chunks = []
    for row in cur.execute(
        "SELECT id, document_id, chunk_type, ordinal, content, keywords, "
        "token_count, metadata FROM chunks"
    ):
        d = dict(row)
        meta = {}
        try:
            raw = json.loads(d.pop("metadata")) if d.get("metadata") else {}
        except Exception:
            raw = {}
        if isinstance(raw, dict):
            fields = raw.get("fields")
            if isinstance(fields, dict):
                meta = {
                    "question": fields.get("question"),
                    "answer": fields.get("answer"),
                    "keywords": fields.get("keywords"),
                }
        d["meta"] = meta
        if d["chunk_type"] == "qa_pair":
            qa_chunks.append(d)
        else:
            content = d.pop("content", "")
            d["preview"] = content[:300]
            body_chunks.append(d)

    conn.close()

    stats = {
        "documents": len(docs),
        "chunks_total": len(qa_chunks) + len(body_chunks),
        "qa_chunks": len(qa_chunks),
        "body_chunks": len(body_chunks),
    }

    # QA duplication analysis.
    from collections import Counter
    norm = Counter()
    for c in qa_chunks:
        q = (c.get("meta") or {}).get("question") or ""
        key = q.strip()
        norm[key] += 1
    distinct = len(norm)
    dup_instances = sum(v - 1 for v in norm.values() if v > 1)
    stats["distinct_questions"] = distinct
    stats["duplicate_instances"] = dup_instances
    stats["questions_with_gt1_copies"] = sum(1 for v in norm.values() if v > 1)
    stats["questions_with_gt5_copies"] = sum(1 for v in norm.values() if v > 5)

    return {
        "schema": "kb-snapshot/v1",
        "exported_at": datetime.now(UTC).isoformat(),
        "counts": stats,
        "documents": docs,
        "qa_chunks": qa_chunks,
        "body_chunks": body_chunks,
    }


# ---------------------------------------------------------------------------
# Snapshot management
# ---------------------------------------------------------------------------

def create_snapshot(
    label: str,
    notes: str = "",
    db_path: str | None = None,
    include_benchmark: bool = True,
) -> Path:
    """Create a version snapshot directory populated with archives."""
    db_path = db_path or str(DB_PATH)
    label = label.strip().replace(" ", "_")
    target = VERSIONS_ROOT / label
    if target.exists():
        raise FileExistsError(f"Snapshot already exists: {target}")
    target.mkdir(parents=True, exist_ok=False)

    export = export_compact(db_path)
    with open(target / "kb_export.json", "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

    counts = export["counts"]
    manifest = {
        "version": label,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_head(),
        "source_db": str(Path(db_path).name),
        "notes": notes,
        "counts": counts,
    }
    with open(target / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Copy evaluation artifacts if present.
    data_dir = PROJECT_ROOT / "data"
    for name in ("test_questions.json", "test_questions_report.txt",
                 "eval_clean.json", "eval_questions.json"):
        src = data_dir / name
        if src.exists():
            shutil.copy2(src, target / src.name)

    # Copy latest benchmark results / plots if already computed.
    bench_dir = target / "benchmarks"
    bench_dir.mkdir(exist_ok=True)
    for src_name in ("benchmark_results.json", "ir_metrics.json"):
        src = data_dir / src_name
        if src.exists():
            shutil.copy2(src, bench_dir / src_name)

    plots_dir = data_dir / "plots"
    if plots_dir.exists():
        dst_plots = target / "plots"
        dst_plots.mkdir(exist_ok=True)
        for png in plots_dir.glob("*.png"):
            shutil.copy2(png, dst_plots / png.name)

    (target / "README.md").write_text(
        f"# KB Version `{label}`\n\n"
        f"- Created: {manifest['created_at']}\n"
        f"- Git commit: `{manifest['git_commit']}`\n"
        f"- Notes: {notes or '(none)'}\n\n"
        f"## Counts\n\n"
        f"- Documents: {counts['documents']}\n"
        f"- Chunks: {counts['chunks_total']} (QA: {counts['qa_chunks']}, "
        f"body: {counts['body_chunks']})\n"
        f"- Distinct questions: {counts['distinct_questions']}\n"
        f"- Duplicate QA copies: {counts['duplicate_instances']}\n",
        encoding="utf-8",
    )
    logger.info("Created snapshot %s at %s", label, target)
    return target


def list_snapshots() -> list[dict]:
    """List snapshots newest-first."""
    if not VERSIONS_ROOT.exists():
        return []
    snapshots = []
    for d in sorted(VERSIONS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        manifest_path = d / "manifest.json"
        info = {"label": d.name, "created_at": "", "git_commit": "",
                "counts": {}}
        if manifest_path.exists():
            with contextlib.suppress(Exception):
                info.update(json.loads(manifest_path.read_text(encoding="utf-8")))
        snapshots.append(info)
    snapshots.sort(key=lambda s: s["created_at"], reverse=True)
    return snapshots


def latest_snapshot() -> Path | None:
    snaps = list_snapshots()
    if not snaps:
        return None
    return VERSIONS_ROOT / snaps[0]["label"]


def snapshot_exists(label: str) -> bool:
    return (VERSIONS_ROOT / label).exists()
