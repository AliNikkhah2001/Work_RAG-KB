"""Export and restore KB processed artifacts for DVC.

Snapshots live in the kb-source repo (Work_RAG-KB-SourceFiles), next to the
raw corpus. Git keeps the ``.dvc`` pointers; DVC keeps the bytes.

Layout under ``processed/<corpus-date>/``:

- ``chunks/documents.jsonl`` and ``chunks/chunks.jsonl`` (no vectors)
- ``vectors/<model>-<dim>/embeddings.jsonl``
- ``entity-lists/entity_lists.jsonl``
- ``packs/`` copy of the pinned YAML packs
- ``manifest.json`` IndexManifest plus file counts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

from kb_manager.registry.loader import PACKS_ROOT, get_registry
from kb_manager.registry.manifest import build_manifest

DEFAULT_DB_URL = "postgresql://postgres:postgres@127.0.0.1:15432/kb_manager"
SNAPSHOT_NAME = "dvc_snapshot.json"


def parse_embedding(value: Any) -> list[float] | None:
    """Normalize a pgvector / JSON / text embedding to a float list."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [float(part) for part in inner.split(",")]
    return None


def _sync_url(url: str | None) -> str:
    raw = (url or os.environ.get("KB_DB_URL") or DEFAULT_DB_URL).strip()
    return (
        raw.replace("postgresql+asyncpg://", "postgresql://")
        .replace("postgres+asyncpg://", "postgresql://")
    )


def _json_default(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"not serializable: {type(value).__name__}")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=_json_default))
            fh.write("\n")
            count += 1
    return count


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vector_dir_name(model: str, dim: int) -> str:
    safe = (model or "unknown").replace("/", "__")
    return f"{safe}-{dim}"


def _connect(url: str):
    import psycopg2
    import psycopg2.extras

    conn = psycopg2.connect(url)
    conn.autocommit = False
    return conn, psycopg2.extras


def corpus_date(kb_source: Path) -> str:
    pin = kb_source / ".dvc_pins" / "current.txt"
    if pin.is_file():
        text = pin.read_text(encoding="utf-8").strip()
        if text:
            return text
    return "unpinned"


def _copy_active_packs(dest: Path) -> dict[str, str]:
    reg = get_registry()
    pins = reg.active_pins()
    mapping = {
        "schemas": ("schemas", None, pins.get("schemas")),
        "taxonomy": ("taxonomy", None, pins.get("taxonomy")),
        "canned": ("canned", None, pins.get("canned")),
        "retrieval": ("retrieval", None, pins.get("retrieval")),
        "prompts_orchestrator": ("prompts", "orchestrator", pins.get("prompts_orchestrator")),
        "prompts_guardrails": ("prompts", "guardrails", pins.get("prompts_guardrails")),
    }
    copied: dict[str, str] = {}
    for pin_key, (kind, scope, version) in mapping.items():
        if not version:
            continue
        src = reg._pack_path(kind, version, scope)
        if not src.is_file():
            continue
        rel = src.relative_to(reg.root)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        copied[pin_key] = str(rel)
    return copied


def export_snapshot(
    *,
    kb_source: Path,
    db_url: str | None = None,
    corpus: str | None = None,
) -> dict[str, Any]:
    """Dump chunks, vectors, entity lists, and pinned packs into kb-source."""
    kb_source = kb_source.resolve()
    day = corpus or corpus_date(kb_source)
    out = kb_source / "processed" / day
    if out.exists():
        shutil.rmtree(out)
    url = _sync_url(db_url)
    conn, extras = _connect(url)
    try:
        with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, source_path, title, domain, category, status, version,
                       source_hash, content_hash, chunk_count, metadata,
                       created_at, updated_at
                FROM documents
                ORDER BY id
                """
            )
            documents = [dict(row) for row in cur.fetchall()]
            cur.execute(
                """
                SELECT id, document_id, parent_id, ordinal, chunk_type, content,
                       heading_path, keywords, token_count, embedding_model,
                       embedding::text AS embedding, quality_score, is_verified,
                       metadata, created_at
                FROM chunks
                ORDER BY id
                """
            )
            chunk_rows = [dict(row) for row in cur.fetchall()]
            cur.execute(
                """
                SELECT id, list_name, description, description_embedding,
                       content_json, created_at
                FROM entity_lists
                ORDER BY id
                """
            )
            entities = [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()

    chunks_out: list[dict[str, Any]] = []
    vectors: dict[str, list[dict[str, Any]]] = {}
    for row in chunk_rows:
        embedding = parse_embedding(row.pop("embedding"))
        model = row.get("embedding_model") or ""
        dim = len(embedding) if embedding else 0
        chunks_out.append(row)
        if embedding is None:
            continue
        key = _vector_dir_name(model, dim)
        vectors.setdefault(key, []).append(
            {"id": row["id"], "embedding_model": model, "dim": dim, "embedding": embedding}
        )

    doc_n = _write_jsonl(out / "chunks" / "documents.jsonl", documents)
    chunk_n = _write_jsonl(out / "chunks" / "chunks.jsonl", chunks_out)
    vector_counts: dict[str, int] = {}
    for key, rows in vectors.items():
        vector_counts[key] = _write_jsonl(out / "vectors" / key / "embeddings.jsonl", rows)
    entity_n = _write_jsonl(out / "entity-lists" / "entity_lists.jsonl", entities)
    pack_files = _copy_active_packs(out / "packs")

    model_name = ""
    dim = 0
    if vectors:
        first_key = next(iter(vectors))
        sample = vectors[first_key][0]
        model_name = sample["embedding_model"]
        dim = int(sample["dim"])

    manifest = build_manifest(
        doc_count=doc_n,
        chunk_count=chunk_n,
        source_dir=str(kb_source / day),
        embedding_model=model_name,
        embedding_dim=dim,
        activate=True,
        artifacts={
            "corpus_date": day,
            "documents": doc_n,
            "chunks": chunk_n,
            "vectors": vector_counts,
            "entity_lists": entity_n,
            "packs": pack_files,
        },
    )
    manifest_path = out / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "corpus_date": day,
        "out": str(out),
        "documents": doc_n,
        "chunks": chunk_n,
        "vectors": vector_counts,
        "entity_lists": entity_n,
        "packs": pack_files,
        "manifest_id": manifest.manifest_id,
        "files": {
            "chunks": _file_md5(out / "chunks" / "chunks.jsonl"),
            "documents": _file_md5(out / "chunks" / "documents.jsonl"),
            "entity_lists": _file_md5(out / "entity-lists" / "entity_lists.jsonl"),
            "manifest": _file_md5(manifest_path),
        },
    }
    (out / "snapshot.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pin = {
        "corpus_date": day,
        "manifest_id": manifest.manifest_id,
        "documents": doc_n,
        "chunks": chunk_n,
        "entity_lists": entity_n,
        "vectors": vector_counts,
        "embedding_model": model_name,
        "embedding_dim": dim,
        "packs": get_registry().active_pins(),
    }
    pin_path = kb_source / ".dvc_pins" / "processed.txt"
    pin_path.parent.mkdir(parents=True, exist_ok=True)
    pin_path.write_text(day + "\n", encoding="utf-8")
    (kb_source / ".dvc_pins" / "processed.json").write_text(
        json.dumps(pin, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # Bind-mounted into the KB container so the running server reports this pin.
    pack_pin = PACKS_ROOT / SNAPSHOT_NAME
    pack_pin.write_text(json.dumps(pin, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary["server_pin"] = str(pack_pin)
    return summary


def _db_counts(url: str) -> dict[str, int]:
    conn, _extras = _connect(url)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM documents")
            docs = int(cur.fetchone()[0])
            cur.execute("SELECT count(*) FROM chunks")
            chunks = int(cur.fetchone()[0])
            cur.execute("SELECT count(*) FROM entity_lists")
            entities = int(cur.fetchone()[0])
            cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")
            embedded = int(cur.fetchone()[0])
        return {
            "documents": docs,
            "chunks": chunks,
            "entity_lists": entities,
            "embedded_chunks": embedded,
        }
    finally:
        conn.close()


def verify_snapshot(*, kb_source: Path, db_url: str | None = None, corpus: str | None = None) -> dict[str, Any]:
    """Compare a DVC snapshot on disk with the live database."""
    kb_source = kb_source.resolve()
    day = corpus or corpus_date(kb_source)
    out = kb_source / "processed" / day
    snap_path = out / "snapshot.json"
    if not snap_path.is_file():
        raise FileNotFoundError(f"missing snapshot: {snap_path}")
    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    live = _db_counts(_sync_url(db_url))
    vector_n = sum(int(n) for n in (snap.get("vectors") or {}).values())
    ok = (
        live["documents"] == int(snap["documents"])
        and live["chunks"] == int(snap["chunks"])
        and live["entity_lists"] == int(snap["entity_lists"])
        and live["embedded_chunks"] == vector_n
    )
    pack_ok = True
    for rel in (snap.get("packs") or {}).values():
        if not (out / "packs" / rel).is_file():
            pack_ok = False
    chunks = _read_jsonl(out / "chunks" / "chunks.jsonl")
    no_vectors_inline = all("embedding" not in row for row in chunks[:20])
    return {
        "ok": bool(ok and pack_ok and no_vectors_inline and chunks),
        "corpus_date": day,
        "snapshot": {
            "documents": snap["documents"],
            "chunks": snap["chunks"],
            "entity_lists": snap["entity_lists"],
            "vectors": vector_n,
        },
        "database": live,
        "packs_present": pack_ok,
        "chunks_without_vectors": no_vectors_inline,
    }


def restore_snapshot(
    *,
    kb_source: Path,
    db_url: str | None = None,
    corpus: str | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    """Load a snapshot back into Postgres. Refuses to write unless apply=True."""
    kb_source = kb_source.resolve()
    day = corpus or (kb_source / ".dvc_pins" / "processed.txt").read_text(encoding="utf-8").strip()
    out = kb_source / "processed" / day
    documents = _read_jsonl(out / "chunks" / "documents.jsonl")
    chunks = _read_jsonl(out / "chunks" / "chunks.jsonl")
    entities = _read_jsonl(out / "entity-lists" / "entity_lists.jsonl")
    embeddings: dict[str, list[float]] = {}
    models: dict[str, str] = {}
    vector_root = out / "vectors"
    if vector_root.is_dir():
        for path in vector_root.glob("*/embeddings.jsonl"):
            for row in _read_jsonl(path):
                embeddings[row["id"]] = row["embedding"]
                models[row["id"]] = row.get("embedding_model") or ""
    report = {
        "corpus_date": day,
        "documents": len(documents),
        "chunks": len(chunks),
        "entity_lists": len(entities),
        "vectors": len(embeddings),
        "applied": False,
    }
    if not apply:
        report["dry_run"] = True
        return report

    url = _sync_url(db_url)
    conn, extras = _connect(url)
    try:
        with conn.cursor() as cur:
            for doc in documents:
                cur.execute(
                    """
                    INSERT INTO documents (
                        id, source_path, title, domain, category, status, version,
                        source_hash, content_hash, chunk_count, metadata, created_at, updated_at
                    ) VALUES (
                        %(id)s, %(source_path)s, %(title)s, %(domain)s, %(category)s, %(status)s,
                        %(version)s, %(source_hash)s, %(content_hash)s, %(chunk_count)s,
                        %(metadata)s, %(created_at)s, %(updated_at)s
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        title = EXCLUDED.title,
                        chunk_count = EXCLUDED.chunk_count,
                        metadata = EXCLUDED.metadata,
                        updated_at = EXCLUDED.updated_at
                    """,
                    {
                        **doc,
                        "metadata": extras.Json(doc.get("metadata") or {}),
                    },
                )
            for chunk in chunks:
                emb = embeddings.get(chunk["id"])
                cur.execute(
                    """
                    INSERT INTO chunks (
                        id, document_id, parent_id, ordinal, chunk_type, content,
                        heading_path, keywords, token_count, embedding_model, embedding,
                        quality_score, is_verified, metadata, created_at
                    ) VALUES (
                        %(id)s, %(document_id)s, %(parent_id)s, %(ordinal)s, %(chunk_type)s,
                        %(content)s, %(heading_path)s, %(keywords)s, %(token_count)s,
                        %(embedding_model)s, %(embedding)s, %(quality_score)s, %(is_verified)s,
                        %(metadata)s, %(created_at)s
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        embedding_model = EXCLUDED.embedding_model
                    """,
                    {
                        **chunk,
                        "keywords": extras.Json(chunk.get("keywords") or []),
                        "metadata": extras.Json(chunk.get("metadata") or {}),
                        "embedding_model": models.get(chunk["id"]) or chunk.get("embedding_model"),
                        "embedding": None if emb is None else "[" + ",".join(str(x) for x in emb) + "]",
                        "is_verified": bool(chunk.get("is_verified")),
                    },
                )
            for ent in entities:
                cur.execute(
                    """
                    INSERT INTO entity_lists (
                        id, list_name, description, description_embedding, content_json, created_at
                    ) VALUES (
                        %(id)s, %(list_name)s, %(description)s, %(description_embedding)s,
                        %(content_json)s, %(created_at)s
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        list_name = EXCLUDED.list_name,
                        description = EXCLUDED.description,
                        description_embedding = EXCLUDED.description_embedding,
                        content_json = EXCLUDED.content_json
                    """,
                    {
                        **ent,
                        "description_embedding": extras.Json(ent.get("description_embedding")),
                        "content_json": extras.Json(ent.get("content_json") or {}),
                    },
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    report["applied"] = True
    return report


def load_server_pin() -> dict[str, Any] | None:
    path = PACKS_ROOT / SNAPSHOT_NAME
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DVC snapshot of processed KB data")
    parser.add_argument("command", choices=("export", "verify", "restore"))
    parser.add_argument("--kb-source", type=Path, required=True)
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "export":
        result = export_snapshot(kb_source=args.kb_source, db_url=args.db_url, corpus=args.corpus)
    elif args.command == "verify":
        result = verify_snapshot(kb_source=args.kb_source, db_url=args.db_url, corpus=args.corpus)
    else:
        result = restore_snapshot(
            kb_source=args.kb_source,
            db_url=args.db_url,
            corpus=args.corpus,
            apply=args.apply,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == "verify" and not result.get("ok"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
