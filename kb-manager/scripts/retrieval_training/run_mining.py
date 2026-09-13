#!/usr/bin/env python3
"""Stage 1 runner: mine Tier1..Tier4 hard negatives for every evaluated row.

Reads ``artifacts/retrieval_training/retrieval_failures.jsonl`` (query_id,
query, gold id), runs ``search_knowledge_base(q, top_k=100)`` in-process per
query (600 s timeout each), and appends :class:`MinedNegative` JSONL rows to
``artifacts/retrieval_training/mined_negatives.jsonl``.

Resume-capable: query_ids already present in the output file are skipped, so
re-running continues where the previous run stopped (torn lines are ignored
and re-mined). Use ``--max-new N`` to bound one invocation.

Pool limit: the public API truncates every exposed leg to ``top_k`` rows, so
the minable RRF pool is the ``merged_candidates`` slice of 100; deeper RRF
ranks are unreachable and Tier3 "outside pool" candidates come from DB
same-document lookups instead.

Launch with ``PYTHONHASHSEED=0`` pinned at the interpreter level (the beam-4
query-expansion shuffle uses ``hash(query)``; a salted hash jitters leg
ranks across processes). The script aborts otherwise.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sqlite3
import subprocess
import sys
import time

# Env BEFORE kb_manager imports (CRITICAL for PYTHONHASHSEED note above and
# for the DB/source-dir wiring).
os.environ.setdefault(
    "KB_DB_URL", "sqlite+aiosqlite:///D:/Code/KB/kb-manager/data/kb_9_7_2026.db"
)
os.environ.setdefault("KB_SOURCE_DIR", "D:/Code/KB/kb-source/KB_9.7.2026")
os.environ.setdefault("PYTHONHASHSEED", "0")

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("FATAL: relaunch this script with PYTHONHASHSEED=0 in the environment.")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kb_manager.retrieval_training.mining import (  # noqa: E402
    MiningConfig,
    mine_negatives,
    normalize_question_text,
)
from kb_manager.retrieval_training.schemas import TierRatio  # noqa: E402
from kb_manager.web.routes.search import search_knowledge_base_sync  # noqa: E402

FAILURES = "artifacts/retrieval_training/retrieval_failures.jsonl"
OUTPUT = "artifacts/retrieval_training/mined_negatives.jsonl"
PER_QUERY_TIMEOUT = 600


def _result_items(steps, attr: str) -> list:
    seq = getattr(steps, attr, None) or []
    if isinstance(seq, dict):
        seq = seq.get("results", [])
    return list(seq)


def _cid(item) -> str:
    if isinstance(item, dict):
        return str(item.get("chunk_id", ""))
    return str(getattr(item, "chunk_id", ""))


def _did(item) -> str:
    if isinstance(item, dict):
        return str(item.get("doc_id", ""))
    return str(getattr(item, "doc_id", ""))


def _rank_map(items: list) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, it in enumerate(items):
        c = _cid(it)
        if c and c not in out:
            out[c] = i
    return out


def _load_db(db_path: str):
    con = sqlite3.connect(db_path)
    chunks: dict[str, dict] = {}
    for cid, doc, ctype, meta, content in con.execute(
        "SELECT id, document_id, chunk_type, metadata, content FROM chunks"
    ):
        question = ""
        is_parent = False
        try:
            m = json.loads(meta) if meta else {}
            if isinstance(m, dict):
                is_parent = bool(m.get("is_parent", False))
                fields = m.get("fields", {})
                if isinstance(fields, dict):
                    for k, v in fields.items():
                        if k.lower() == "question" and isinstance(v, str) and v.strip():
                            question = v
                            break
        except (ValueError, AttributeError):
            pass
        chunks[cid] = {
            "document_id": doc,
            "chunk_type": ctype or "",
            "is_parent": is_parent,
            "question": question,
            "content": content or "",
        }
    docs: dict[str, dict] = {}
    for did, title, spath in con.execute("SELECT id, title, source_path FROM documents"):
        docs[did] = {"title": title or "", "source_path": spath or ""}
    con.close()
    return chunks, docs


def _code_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description="Mine Tier1..Tier4 hard negatives.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", type=str, default="",
                    help="JSON file with tier1..tier4 ratios (TierRatio).")
    ap.add_argument("--failures", type=str, default=FAILURES)
    ap.add_argument("--output", type=str, default=OUTPUT)
    ap.add_argument("--max-new", type=int, default=0,
                    help="Cap new queries this invocation (0 = all remaining).")
    ap.add_argument("--shard", type=int, default=0,
                    help="Shard index (with --num-shards) for parallel workers.")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="Total shards; this worker mines rows[i::num_shards].")
    ap.add_argument("--per-query-timeout", type=int, default=PER_QUERY_TIMEOUT)
    ap.add_argument("--provenance", type=str, default="")
    args = ap.parse_args()

    t0 = time.monotonic()
    ratios = TierRatio()
    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            ratios = TierRatio.from_dict(json.load(fh))
    rows = [json.loads(l) for l in open(args.failures, encoding="utf-8") if l.strip()]
    try:
        with open(args.output, encoding="utf-8") as fh:
            done = set()
            for line in fh:
                try:
                    done.add(json.loads(line).get("query_id", ""))
                except ValueError:
                    continue  # torn line: re-mine that query
    except FileNotFoundError:
        done = set()
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    rev = _code_rev()
    prov = args.provenance or f"{rev}/kb_9_7_2026.db/tier1342"
    db_path = "data/kb_9_7_2026.db"
    chunks, docs = _load_db(db_path)
    non_parent_ids = sorted(cid for cid, c in chunks.items() if not c["chunk_type"].endswith("_parent"))
    by_doc: dict[str, list[str]] = {}
    for cid in non_parent_ids:
        by_doc.setdefault(chunks[cid]["document_id"], []).append(cid)

    def doc_title(doc_id: str) -> str:
        d = docs.get(doc_id, {})
        sp = d.get("source_path", "")
        base = sp.replace("\\", "/").split("/")[-1] if sp else ""
        return d.get("title") or base or doc_id

    todo = [r for i, r in enumerate(rows)
            if i % args.num_shards == args.shard and r.get("query_id") not in done]
    if args.max_new > 0:
        todo = todo[: args.max_new]
    print(f"rows={len(rows)} done={len(done)} todo={len(todo)} seed={args.seed} "
          f"shard={args.shard}/{args.num_shards}", flush=True)

    tier_counts = {"Tier1": 0, "Tier2": 0, "Tier3": 0, "Tier4": 0}
    errors: list[dict] = []
    executor: concurrent.futures.ThreadPoolExecutor | None = None

    def search_guarded(query: str):
        nonlocal executor
        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        fut = executor.submit(search_knowledge_base_sync, query, 100)
        try:
            return fut.result(timeout=args.per_query_timeout)
        except concurrent.futures.TimeoutError:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=False)
            executor = None
            raise TimeoutError(f"search exceeded {args.per_query_timeout}s")

    out_fh = open(args.output, "a", encoding="utf-8")
    try:
        for n, row in enumerate(todo):
            qid = row["query_id"]
            qtext = row["query"]
            gold = row["gold_chunk_id"]
            try:
                steps = search_guarded(qtext)
            except Exception as exc:  # timeout / pipeline error: retry next run
                errors.append({"query_id": qid, "error": f"{type(exc).__name__}: {exc}"})
                print(f"[{n + 1}/{len(todo)}] {qid} ERROR {exc}", flush=True)
                continue
            bm25 = _rank_map(_result_items(steps, "bm25_results"))
            dense_items = _result_items(steps, "dense_results") or _result_items(steps, "semantic_results")
            dense = _rank_map(dense_items)
            merged_items = _result_items(steps, "merged_candidates")
            merged = _rank_map(merged_items)
            final = _rank_map(_result_items(steps, "final_results"))
            gold_doc = (chunks.get(gold) or {}).get("document_id", "")

            pool: list[dict] = []
            for cid in merged:  # full merged slice (<=100): the minable pool
                c = chunks.get(cid, {})
                pool.append({
                    "chunk_id": cid,
                    "bm25_rank": bm25.get(cid, -1),
                    "dense_rank": dense.get(cid, -1),
                    "rrf_rank": merged.get(cid, -1),
                    "reranker_rank": final.get(cid, -1),
                    "source_document": c.get("document_id", ""),
                    "question_text": c.get("question", ""),
                })
            # Tier3 outside-pool fill: same-doc non-parent chunks beyond the
            # merged slice (RRF ranks deeper than 100 are unreachable via the
            # public API, so these come from the DB, rrf_rank=-1).
            in_pool = set(merged) | {gold}
            for cid in by_doc.get(gold_doc, [])[:50]:
                if cid not in in_pool:
                    c = chunks[cid]
                    pool.append({
                        "chunk_id": cid,
                        "bm25_rank": bm25.get(cid, -1),
                        "dense_rank": dense.get(cid, -1),
                        "rrf_rank": -1,
                        "reranker_rank": final.get(cid, -1),
                        "source_document": c["document_id"],
                        "question_text": c["question"],
                    })
            # Tier4 pool: other-doc non-parent chunks (id/doc only; the
            # miner seeded-samples 2 per query; texts resolve in validation).
            others = [cid for cid in non_parent_ids
                      if chunks[cid]["document_id"] != gold_doc and cid not in in_pool]
            for cid in others:
                pool.append({"chunk_id": cid,
                             "source_document": chunks[cid]["document_id"]})

            query = {
                "query_id": qid,
                "query": qtext,
                "positive_chunk_ids": [gold],
                "gold_merged_rank": merged.get(gold, -1),
                "gold_final_rank": final.get(gold, -1),
                "gold_document_id": gold_doc,
                "candidates": pool,
            }
            cfg = MiningConfig(negatives_per_query=12, tier_ratios=ratios)
            recs = mine_negatives([query], cfg, seed=args.seed, provenance=prov)
            for r in recs:
                d = r.to_dict()
                d["source_document_title"] = doc_title(r.source_document)
                out_fh.write(json.dumps(d, ensure_ascii=False) + "\n")
                tier_counts[r.negative_type] += 1
            out_fh.flush()
            if hasattr(os, "fsync"):
                os.fsync(out_fh.fileno())
            done.add(qid)
            print(f"[{n + 1}/{len(todo)}] {qid} merged={merged.get(gold, -1)} "
                  f"final={final.get(gold, -1)} mined={len(recs)}", flush=True)
    finally:
        out_fh.close()
        if executor is not None:
            executor.shutdown(wait=False)

    el = time.monotonic() - t0
    print(f"tier_counts={tier_counts} errors={len(errors)} elapsed_min={el / 60:.1f}", flush=True)
    for e in errors[:20]:
        print(f"ERROR {e}", flush=True)
    _ = normalize_question_text  # re-exported contract use (dedup inside miners)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
