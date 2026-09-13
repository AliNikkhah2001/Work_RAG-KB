#!/usr/bin/env python3
"""Stage 2 runner: 8-rule validation over mined negatives (in place).

Reads ``artifacts/retrieval_training/mined_negatives.jsonl``, enriches each
record's provenance JSON envelope from the DB (SELECT only: negative/gold
question texts, document ids, parent signals), runs
:func:`validate_negatives` with the normalized-question -> goldset map built
from ALL ``retrieval_failures.jsonl`` rows merged in as ``extra_gold_map``
(rule R2 across files), and rewrites the file with ``validation_status``
``passed``/``failed`` + reason chains.

ALL records are kept (uncertain rows stay in this provenance file with their
reason); the dataset stage drops ``failed`` rows from training pools.

Launch with ``PYTHONHASHSEED=0`` (asserted; determinism note in module).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sqlite3
import sys

os.environ.setdefault(
    "KB_DB_URL", "sqlite+aiosqlite:///D:/Code/KB/kb-manager/data/kb_9_7_2026.db"
)
os.environ.setdefault("KB_SOURCE_DIR", "D:/Code/KB/kb-source/KB_9.7.2026")
os.environ.setdefault("PYTHONHASHSEED", "0")

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("FATAL: relaunch this script with PYTHONHASHSEED=0 in the environment.")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kb_manager.chunker.semantic import SemanticChunker  # noqa: E402
from kb_manager.retrieval_training.schemas import MinedNegative  # noqa: E402
from kb_manager.retrieval_training.validation import (  # noqa: E402
    ValidationConfig,
    validate_negatives,
)

MINED = "artifacts/retrieval_training/mined_negatives.jsonl"
FAILURES = "artifacts/retrieval_training/retrieval_failures.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate mined negatives (8-rule filter).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", type=str, default="",
                    help="Unused knob file (ValidationConfig has no ratios); accepted for CLI compat.")
    ap.add_argument("--mined", type=str, default=MINED)
    ap.add_argument("--failures", type=str, default=FAILURES)
    ap.add_argument("--provenance", type=str, default="validate1")
    args = ap.parse_args()

    recs = []
    with open(args.mined, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(MinedNegative.from_dict(json.loads(line)))
    print(f"loaded {len(recs)} mined records", flush=True)

    # Extra gold map over ALL failure rows (same normalized question, any file).
    extra: dict[str, list[str]] = {}
    for line in open(args.failures, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        norm = SemanticChunker._normalize_question(r.get("query", ""))
        if norm and r.get("gold_chunk_id"):
            extra.setdefault(norm, []).append(r["gold_chunk_id"])

    con = sqlite3.connect("data/kb_9_7_2026.db")
    chunk_info: dict[str, dict] = {}
    for cid, doc, ctype, meta in con.execute(
        "SELECT id, document_id, chunk_type, metadata FROM chunks"
    ):
        question, is_parent = "", False
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
        chunk_info[cid] = {"doc": doc, "ctype": ctype or "", "is_parent": is_parent,
                           "question": question}
    con.close()

    def info(cid: str) -> dict:
        return chunk_info.get(cid, {"doc": "", "ctype": "", "is_parent": False, "question": ""})

    enriched: list[MinedNegative] = []
    for r in recs:
        gold = (r.positive_chunk_ids or [""])[0]
        gi, ni = info(gold), info(r.negative_chunk_id)
        try:
            env = json.loads(r.provenance) if r.provenance.strip().startswith("{") else {}
            if not isinstance(env, dict):
                env = {}
        except ValueError:
            env = {}
        env = dict(env)
        env.setdefault("mine", r.provenance if not env else env.get("mine", r.provenance))
        env.update({
            "neg_question": ni["question"],
            "gold_question": gi["question"] or r.query,
            "neg_doc": ni["doc"],
            "gold_doc": gi["doc"],
            "chunk_type": ni["ctype"],
            "is_parent": ni["is_parent"],
        })
        enriched.append(MinedNegative(
            query_id=r.query_id, query=r.query,
            positive_chunk_ids=list(r.positive_chunk_ids),
            negative_chunk_id=r.negative_chunk_id, negative_type=r.negative_type,
            bm25_rank=r.bm25_rank, dense_rank=r.dense_rank, rrf_rank=r.rrf_rank,
            reranker_rank=r.reranker_rank, source_document=r.source_document,
            validation_status="pending",
            provenance=json.dumps(env, ensure_ascii=False),
        ))

    out = validate_negatives(enriched, ValidationConfig(), seed=args.seed,
                             provenance=args.provenance, extra_gold_map=extra)
    tmp = args.mined + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(r.to_json() + "\n")
    os.replace(tmp, args.mined)

    n_pass = sum(1 for r in out if r.validation_status == "passed")
    reasons = collections.Counter()
    for r in out:
        if r.validation_status == "failed":
            for part in r.provenance.split("|validate:")[-1].split(":"):
                if part.startswith("R"):
                    reasons[part.split(",")[0]] += 1
                    break
    print(f"accepted={n_pass}/{len(out)} rate={n_pass / max(len(out), 1):.3f}", flush=True)
    print(f"top_uncertain_reasons={reasons.most_common(10)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
