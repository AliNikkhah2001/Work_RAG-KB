#!/usr/bin/env python3
"""Stage 3 runner: build CE + dense datasets with doc-group-confined splits.

Reads validated ``mined_negatives.jsonl`` (keeps ``passed`` only; ``failed``
rows stay in the provenance file), resolves ``positive_text`` /
``negative_text`` via DB content by chunk id (SELECT only; missing chunks
become tallied error entries, never silent skips), and writes
``artifacts/retrieval_training/datasets/{train,validation,test}.jsonl`` plus
the ``provenance.json`` sidecar (manifest + stats + config + code rev +
DB sha).

Split rule (exact): group queries by source file (``query_id`` prefix);
large files (Individual/Cheque/Public) 80/10/10 over gold-document clusters
(seeded shuffle, greedy fill); small files (Dispute/Etebarito) wholly train;
fail-class (A/B/C) representatives forced into validation+test by moving
whole doc groups. See :mod:`kb_manager.retrieval_training.dataset`.

Launch with ``PYTHONHASHSEED=0`` (asserted).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys

os.environ.setdefault(
    "KB_DB_URL", "sqlite+aiosqlite:///D:/Code/KB/kb-manager/data/kb_9_7_2026.db"
)
os.environ.setdefault("KB_SOURCE_DIR", "D:/Code/KB/kb-source/KB_9.7.2026")
os.environ.setdefault("PYTHONHASHSEED", "0")

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("FATAL: relaunch this script with PYTHONHASHSEED=0 in the environment.")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kb_manager.retrieval_training.dataset import DatasetConfig  # noqa: E402
from kb_manager.retrieval_training.dataset import build_dataset  # noqa: E402
from kb_manager.retrieval_training.schemas import TierRatio  # noqa: E402

MINED = "artifacts/retrieval_training/mined_negatives.jsonl"
FAILURES = "artifacts/retrieval_training/retrieval_failures.jsonl"
OUTDIR = "artifacts/retrieval_training/datasets"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _code_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build retrieval-training datasets.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", type=str, default="",
                    help="JSON file with tier1..tier4 ratios (TierRatio).")
    ap.add_argument("--mined", type=str, default=MINED)
    ap.add_argument("--failures", type=str, default=FAILURES)
    ap.add_argument("--output", type=str, default=OUTDIR)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--provenance", type=str, default="")
    args = ap.parse_args()

    ratios = TierRatio()
    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            ratios = TierRatio.from_dict(json.load(fh))

    mined = [json.loads(l) for l in open(args.mined, encoding="utf-8") if l.strip()]
    passed = [m for m in mined if m.get("validation_status") == "passed"]
    print(f"mined={len(mined)} passed={len(passed)} dropped={len(mined) - len(passed)}", flush=True)

    con = sqlite3.connect("data/kb_9_7_2026.db")
    content_of = {cid: (c or "") for cid, c in con.execute("SELECT id, content FROM chunks")}
    doc_of = {cid: d for cid, d in con.execute("SELECT id, document_id FROM chunks")}
    con.close()

    flat: list[dict] = []
    for m in passed:
        qid = m.get("query_id", "")
        golds = m.get("positive_chunk_ids", [])
        gold = golds[0] if golds else ""
        neg = m.get("negative_chunk_id", "")
        pos_text = content_of.get(gold, "")
        neg_text = content_of.get(neg, "")
        if not gold or not neg:
            flat.append({"query_id": qid, "error": "blank chunk id"})
            continue
        if gold not in content_of:
            flat.append({"query_id": qid, "error": f"missing positive chunk {gold}"})
            continue
        if neg not in content_of:
            flat.append({"query_id": qid, "error": f"missing negative chunk {neg}"})
            continue
        prefix = qid.split("#")[0] if "#" in qid else "unknown"
        flat.append({
            "query_id": qid,
            "query": m.get("query", ""),
            "positive_text": pos_text,
            "negative_text": neg_text,
            "negative_type": m.get("negative_type", "Tier2"),
            "source_file": f"{prefix}.xlsx",
            "gold_document_id": doc_of.get(gold, ""),
        })

    fail_recs = []
    for line in open(args.failures, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            fail_recs.append({"query_id": r.get("query_id", ""),
                              "failure_type": r.get("failure_type", "")})

    cfg = DatasetConfig(train_ratio=args.train_ratio, val_ratio=args.val_ratio,
                        test_ratio=args.test_ratio, negatives_per_positive=4,
                        tier_ratios=ratios)
    rev = _code_rev()
    prov = args.provenance or f"{rev}/kb_9_7_2026.db/tier1342"
    out = build_dataset(flat, fail_recs, cfg, seed=args.seed, provenance=prov)

    os.makedirs(args.output, exist_ok=True)
    for name in ("train", "validation", "test"):
        path = os.path.join(args.output, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for row in out["splits"][name]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = dict(out["manifest"])
    manifest["code_rev"] = rev
    manifest["db_sha"] = _sha256("data/kb_9_7_2026.db")
    sidecar = {
        "manifest": manifest,
        "stats": out["stats"],
        "artifacts": {n: f"{n}.jsonl" for n in ("train", "validation", "test")},
    }
    with open(os.path.join(args.output, "provenance.json"), "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, ensure_ascii=False, indent=2)
    s = out["stats"]
    print(f"queries={s['queries']} negs={s['negatives']} "
          f"avg={s['avg_negatives_per_query']:.2f} splits={s['splits']} "
          f"tiers={s['tier_counts']} errors={len(s['errors'])} warnings={s['warnings']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
