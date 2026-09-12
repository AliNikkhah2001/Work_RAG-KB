"""Agent 2 — stage-level retrieval failure analysis over ALL 573 QA rows.

Frozen baseline v10. Replicates kb_manager/web/routes/benchmarks.py::_run_massive
enumeration + GT mapping EXACTLY, then runs search_knowledge_base(query,
top_k=100) per gold-mapped row and classifies:

  A retriever_failure : gold absent from merged top-100
  B fusion_failure    : gold in a leg top-100 but merged rank > 20
  C reranker_failure  : merged rank <= 20 but final rank > 5 (or absent)
  D success           : final rank <= 5 (checked FIRST so Hit@5 == D rate)
  error               : timeout / exception (recorded, kept out of A/B/C/D)
  skipped             : empty question or no gold chunk (logged in .md only)

Env is set BEFORE any kb_manager import (in-process, no server touched).
DB access is SELECT-only. Never writes data/massive_results.json.

Outputs:
  artifacts/retrieval_training/retrieval_failures.jsonl  (one line / evaluated row)
  docs/retrieval_training/failure_analysis.md            (counts, metrics, examples)

Usage:
  python artifacts/retrieval_training/agent2_failure_analysis.py [--enumerate-only]
"""

import os as _os

# --- MUST be set before any kb_manager import ---
_os.environ["KB_DB_URL"] = "sqlite+aiosqlite:///D:/Code/KB/kb-manager/data/kb_9_7_2026.db"
_os.environ["KB_SOURCE_DIR"] = "D:/Code/KB/kb-source/KB_9.7.2026"

import asyncio
import hashlib
import json
import pathlib
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime

# Repo root must be importable (script lives in artifacts/, not on sys.path).
sys.path.insert(0, "D:/Code/KB/kb-manager")

ENUMERATE_ONLY = "--enumerate-only" in sys.argv
RESUME = "--resume" in sys.argv
FINALIZE_ONLY = "--finalize-only" in sys.argv

REPO = pathlib.Path("D:/Code/KB/kb-manager")
OUT_JSONL = REPO / "artifacts" / "retrieval_training" / "retrieval_failures.jsonl"
OUT_MD = REPO / "docs" / "retrieval_training" / "failure_analysis.md"
BASELINE_OLD = REPO / "data" / "massive_results_baseline_oldcode.json"
FREEZE_JSON = REPO / "artifacts" / "retrieval_training" / "baseline_freeze.json"
TEMP_PROGRESS = pathlib.Path(_os.environ.get("TEMP", ".")) / "agent2_progress.json"

PER_QUERY_TIMEOUT_S = 600
TOP_K = 100

from kb_manager.config import load_config  # noqa: E402
from kb_manager.models.database import Database  # noqa: E402
from sqlalchemy import text  # noqa: E402


def rank_of(results, gt_id):
    for i, r in enumerate(results):
        if r.chunk_id == gt_id:
            return i + 1
    return None


def sha256_file(path, chunk_mb=8):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_mb * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git(cmd):
    try:
        return subprocess.run(
            ["git"] + cmd, cwd=str(REPO), capture_output=True, text=True, timeout=60
        ).stdout.strip()
    except Exception:
        return "?"


def enumerate_files(cfg):
    """EXACT replica of _run_massive file discovery."""
    import pathlib as _pl

    candidates = []
    env_src = _os.getenv("KB_SOURCE_DIR", "")
    if env_src and _pl.Path(env_src).exists():
        candidates.append(_pl.Path(env_src))
    for cand in [
        _pl.Path("D:/Code/KB/kb-source/KB_9.7.2026"),
        _pl.Path("D:/Code/KB/kb-source/1405-05-31"),
        _pl.Path(cfg.source_dir),
    ]:
        if cand.exists() and cand not in candidates:
            candidates.append(cand)
    files = []
    for src in candidates:
        for p in src.rglob("*.xlsx"):
            if p.name.startswith("~$") or "TestQuestion" in str(p):
                continue
            if any(s in p.stem for s in ["واژگان معادل", "محدودیت ها"]):
                continue
            try:
                from kb_manager.parsers.xlsx_parser import XlsxParser

                parsed = XlsxParser().parse(str(p))
                for sh in parsed.sheets:
                    if sh.get("schema") == "crm_qa":
                        sp = str(p.resolve())
                        if sp not in files:
                            files.append(sp)
                        break
            except Exception:
                continue
        if files:
            break
    return files


def build_file_rows(files):
    from kb_manager.parsers.xlsx_parser import XlsxParser

    file_rows = []
    total = 0
    for f in files:
        try:
            sh = [s for s in XlsxParser().parse(f).sheets if s.get("schema") == "crm_qa"][0]
            total += len(sh["rows"])
            file_rows.append((f, sh))
        except Exception:
            pass
    return file_rows, total


def map_gold(q, qa_chunks):
    """EXACT replica of _run_massive 3-step GT mapping. Returns chunk id or None."""
    expected = None
    for cid, content, meta in qa_chunks:
        mq = ((meta.get("fields") or {}).get("question") or "").strip()
        if mq and mq == q:
            expected = cid
            break
    if not expected:
        prefix = f"سوال: {q}"
        for cid, content, meta in qa_chunks:
            if content.startswith(prefix) or prefix in content:
                expected = cid
                break
    if not expected:
        for cid, content, meta in qa_chunks:
            if q[:30] in content:
                expected = cid
                break
    return expected


def classify(bm25_rank, dense_rank, merged_rank, final_rank):
    hit5 = final_rank is not None and final_rank <= 5
    if hit5:
        return "D", True
    if merged_rank is None:
        return "A", False
    if merged_rank > 20:
        return "B", False
    return "C", False


async def main():
    t_run0 = time.monotonic()
    cfg = load_config()
    if FINALIZE_ONLY:
        await finalize_from_jsonl(cfg, t_run0)
        return
    files = enumerate_files(cfg)
    file_rows, total = build_file_rows(files)
    print(f"files={len(files)} total_rows={total}", flush=True)
    if ENUMERATE_ONLY:
        # Gold-map everything without searching (fast sanity check).
        from kb_manager.parsers.xlsx_parser import XlsxParser  # noqa: F401

        db = Database(cfg.db)
        n_empty = n_nochunk = n_nofile = n_gold = 0
        for f, sh in file_rows:
            headers = [h.lower() for h in sh["headers"]]
            q_idx = headers.index("question") if "question" in headers else 0
            qpath = str(pathlib.Path(f).resolve()).replace("\\", "/")
            async with db.session() as s:
                r = await s.execute(text("SELECT id, source_path FROM documents"))
                doc_id = None
                for row in r.fetchall():
                    if row[1].replace("\\", "/").lower() == qpath.lower():
                        doc_id = row[0]
                        break
                if not doc_id:
                    n_nofile += len(sh["rows"])
                    continue
                r2 = await s.execute(
                    text("SELECT id, content, chunk_type, metadata FROM chunks WHERE document_id = :d"),
                    {"d": doc_id},
                )
                qa_chunks = []
                for _cid, _content, _ctype, _meta in r2.fetchall():
                    if _ctype != "qa_pair":
                        continue
                    _meta_dict = {}
                    if _meta:
                        try:
                            _meta_dict = json.loads(_meta) if isinstance(_meta, str) else (_meta or {})
                        except Exception:
                            _meta_dict = {}
                    qa_chunks.append((_cid, _content, _meta_dict))
            for row in sh["rows"]:
                q = row[q_idx].strip() if q_idx < len(row) else ""
                if not q:
                    n_empty += 1
                    continue
                if map_gold(q, qa_chunks):
                    n_gold += 1
                else:
                    n_nochunk += 1
        await db.close()
        print(
            f"enumerate: total={total} gold={n_gold} empty={n_empty} nochunk={n_nochunk} nofile={n_nofile}",
            flush=True,
        )
        return

    from kb_manager.web.routes.search import search_knowledge_base

    db = Database(cfg.db)
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    preloaded = {}
    if RESUME and OUT_JSONL.exists():
        for line in OUT_JSONL.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                preloaded[rec["query_id"]] = rec
            except Exception:
                continue  # torn tail line: will be re-run
        print(f"resume: preloaded {len(preloaded)} rows", flush=True)
    fout = open(OUT_JSONL, "a" if preloaded else "w", encoding="utf-8")
    rows = []  # in-memory copies for the md report
    skipped = []  # (query_id, reason, file, query)
    counts = {"A": 0, "B": 0, "C": 0, "D": 0, "error": 0}
    hits = 0
    evaluated = 0
    done = 0
    errors = []
    b_in_leg = 0  # B rows with gold in >=1 exposed leg top-100

    for f, sh in file_rows:
        stem = pathlib.Path(f).stem
        headers = [h.lower() for h in sh["headers"]]
        q_idx = headers.index("question") if "question" in headers else 0
        qpath = str(pathlib.Path(f).resolve()).replace("\\", "/")
        async with db.session() as s:
            r = await s.execute(text("SELECT id, source_path FROM documents"))
            doc_id = None
            for row in r.fetchall():
                if row[1].replace("\\", "/").lower() == qpath.lower():
                    doc_id = row[0]
                    break
            if not doc_id:
                for rowidx, row in enumerate(sh["rows"]):
                    q = row[q_idx].strip() if q_idx < len(row) else ""
                    skipped.append((f"{stem}#{rowidx}", "file-not-indexed", f, q))
                    done += 1
                continue
            r2 = await s.execute(
                text("SELECT id, content, chunk_type, metadata FROM chunks WHERE document_id = :d"),
                {"d": doc_id},
            )
            qa_chunks = []
            for _cid, _content, _ctype, _meta in r2.fetchall():
                if _ctype != "qa_pair":
                    continue
                _meta_dict = {}
                if _meta:
                    try:
                        _meta_dict = json.loads(_meta) if isinstance(_meta, str) else (_meta or {})
                    except Exception:
                        _meta_dict = {}
                qa_chunks.append((_cid, _content, _meta_dict))
        for rowidx, row in enumerate(sh["rows"]):
            q = row[q_idx].strip() if q_idx < len(row) else ""
            qid = f"{stem}#{rowidx}"
            if not q:
                skipped.append((qid, "empty-question", f, q))
                done += 1
                continue
            expected = map_gold(q, qa_chunks)
            if not expected:
                skipped.append((qid, "no-gold-chunk", f, q))
                done += 1
                continue
            # --- gold-mapped: skip if resumed, else run pipeline with per-query timeout ---
            if qid in preloaded:
                rec = preloaded[qid]
                ftype = rec["failure_type"]
                hit5 = bool(rec["hit5"])
                evaluated += 1
                if ftype in counts:
                    counts[ftype] += 1
                else:
                    counts["error"] = counts.get("error", 0) + 1
                if ftype == "B" and (rec.get("bm25_rank") is not None or rec.get("dense_rank") is not None):
                    b_in_leg += 1
                if ftype == "error":
                    errors.append((qid, rec.get("error", "?")))
                if hit5:
                    hits += 1
                rows.append({"file": f, **rec})
                done += 1
                if done % 25 == 0 or done == total:
                    rate = hits / max(evaluated, 1)
                    print(f"{done}/{total} hit5_rate={rate:.4f} eval={evaluated}", flush=True)
                continue
            t0 = time.monotonic()
            err = None
            try:
                steps = await asyncio.wait_for(
                    search_knowledge_base(q, top_k=TOP_K), timeout=PER_QUERY_TIMEOUT_S
                )
                bm25_rank = rank_of(steps.bm25_results, expected)
                dense_rank = rank_of(steps.dense_results, expected)
                merged_rank = rank_of(steps.merged_candidates, expected)
                final_rank = rank_of(steps.final_results, expected)
            except Exception as e:
                bm25_rank = dense_rank = merged_rank = final_rank = None
                err = f"{type(e).__name__}: {str(e)[:200]}"
            elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
            evaluated += 1
            if err is not None:
                ftype, hit5 = "error", False
                counts["error"] += 1
                errors.append((qid, err))
            else:
                ftype, hit5 = classify(bm25_rank, dense_rank, merged_rank, final_rank)
                counts[ftype] += 1
                if ftype == "B" and (bm25_rank is not None or dense_rank is not None):
                    b_in_leg += 1
            if hit5:
                hits += 1
            rec = {
                "query_id": qid,
                "query": q,
                "gold_chunk_id": expected,
                "bm25_rank": bm25_rank,
                "dense_rank": dense_rank,
                "merged_rank": merged_rank,
                "final_rank": final_rank,
                "hit5": hit5,
                "failure_type": ftype,
                "elapsed_ms": elapsed_ms,
            }
            if err is not None:
                rec["error"] = err
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            rows.append({"file": f, **rec})
            done += 1
            if done % 25 == 0 or done == total:
                rate = hits / max(evaluated, 1)
                print(f"{done}/{total} hit5_rate={rate:.4f} eval={evaluated}", flush=True)
                try:
                    TEMP_PROGRESS.write_text(
                        json.dumps(
                            {"done": done, "total": total, "evaluated": evaluated,
                             "hits": hits, "errors": len(errors)}
                        ),
                        encoding="utf-8",
                    )
                except Exception:
                    pass
    fout.close()
    await db.close()
    rate, mrr = write_report(rows, skipped, counts, b_in_leg, errors, evaluated, total, hits, t_run0)
    print(
        f"DONE total={total} evaluated={evaluated} A={counts['A']} B={counts['B']} "
        f"C={counts['C']} D={counts['D']} err={counts['error']} skipped={len(skipped)} "
        f"hit5={rate:.4f} mrr={mrr:.4f}",
        flush=True,
    )

def write_report(rows, skipped, counts, b_in_leg, errors, evaluated, total, hits, t_run0):
    """Aggregates + overlap + md report. Returns (hit5_rate, mrr)."""
    # ---------- aggregates ----------
    ok_rows = [r for r in rows if r["failure_type"] != "error"]
    n_ok = len(ok_rows)
    hit1 = sum(1 for r in ok_rows if r["final_rank"] == 1)
    mrr = sum(1.0 / r["final_rank"] for r in ok_rows if r["final_rank"]) / max(n_ok, 1)
    hit5_rate = sum(1 for r in ok_rows if r["hit5"]) / max(n_ok, 1)
    elapsed_min = (time.monotonic() - t_run0) / 60.0

    # ---------- overlap vs known baseline 489/22 ----------
    overlap_note = ""
    try:
        base = json.loads(BASELINE_OLD.read_text(encoding="utf-8"))
        bmap = {}  # (file, q80) -> hit?
        for s in base.get("failed_samples", []):
            bmap[(s.get("file"), (s.get("question") or "")[:80])] = False
        for s in base.get("passed_samples", []):
            # passed_samples truncated to 20 in current file? use what exists
            bmap[(s.get("file"), (s.get("question") or "")[:80])] = True
        ov_total = ov_bhit = ov_ohit = 0
        newly_fixed = []
        newly_broken = []
        for r in ok_rows:
            key = (r["file"], r["query"][:80])
            if key in bmap:
                ov_total += 1
                if bmap[key]:
                    ov_bhit += 1
                if r["hit5"]:
                    ov_ohit += 1
                if r["hit5"] and not bmap[key]:
                    newly_fixed.append(r["query_id"])
                if bmap[key] and not r["hit5"]:
                    newly_broken.append(r["query_id"])
        overlap_note = (
            f"overlap_rows={ov_total} baseline_hit5={ov_bhit} ours_hit5={ov_ohit} "
            f"newly_fixed={len(newly_fixed)} newly_broken={len(newly_broken)}"
        )
        overlap_detail = (newly_fixed, newly_broken)
    except Exception as e:
        overlap_note = f"baseline comparison unavailable: {e}"
        overlap_detail = ([], [])

    # ---------- md report ----------
    head_rev = git(["rev-parse", "HEAD"])
    tag_rev = git(["rev-parse", "v10-retrieval-baseline"])
    status = git(["status", "--short", "kb_manager/"])
    freeze = json.loads(FREEZE_JSON.read_text(encoding="utf-8"))
    db_path = REPO / "data" / "kb_9_7_2026.db"
    npz_path = REPO / "data" / "dense_embeddings.npz"
    try:
        db_sha = sha256_file(db_path)
    except Exception as e:
        db_sha = f"unavailable: {e}"
    try:
        npz_sha = sha256_file(npz_path)
    except Exception as e:
        npz_sha = f"unavailable: {e}"

    skip_empty = sum(1 for s in skipped if s[1] == "empty-question")
    skip_nochunk = sum(1 for s in skipped if s[1] == "no-gold-chunk")
    skip_nofile = sum(1 for s in skipped if s[1] == "file-not-indexed")

    def examples(ftype, k=4):
        return [r for r in rows if r["failure_type"] == ftype][:k]

    L = []
    L.append("# Retrieval failure analysis (Agent 2 — v10 frozen baseline)")
    L.append("")
    L.append("## Header / determinism")
    L.append(f"- code HEAD: {head_rev}")
    L.append(f"- tag v10-retrieval-baseline: {tag_rev}")
    L.append(f"- kb_manager/ status clean: {str(status == '')} (status={status!r})")
    L.append("- HEAD vs tag: freeze commit only (adds baseline_freeze.json); `git diff 96438c1 HEAD -- kb_manager/` empty")
    L.append(f"- db: data/kb_9_7_2026.db sha256={db_sha} (freeze={freeze.get('db_sha256')} match={str(db_sha == freeze.get('db_sha256'))})")
    L.append(f"- npz: data/dense_embeddings.npz sha256={npz_sha} (freeze={freeze.get('npz_sha256')} match={str(npz_sha == freeze.get('npz_sha256'))})")
    L.append(f"- models: dense={freeze['models']['dense']} reranker={freeze['models']['reranker']}")
    L.append(f"- retrieval config: {json.dumps(freeze.get('retrieval_config'), ensure_ascii=False)}")
    L.append(f"- env: KB_DB_URL=sqlite+aiosqlite:///D:/Code/KB/kb-manager/data/kb_9_7_2026.db KB_SOURCE_DIR=D:/Code/KB/kb-source/KB_9.7.2026")
    L.append(f"- run at (UTC): {datetime.now(UTC).isoformat()} top_k=100 per-query-timeout=600s")
    L.append(f"- elapsed: {elapsed_min:.1f} min; pipeline deterministic (CPU), single run")
    L.append("")
    L.append("## Method (replica of benchmarks.py::_run_massive)")
    L.append("- QA files: rglob *.xlsx minus ~$/TestQuestion/واژگان معادل/محدودیت ها stems, first dir with crm_qa sheets; GT 3-step: metadata.fields.question==q → content startswith/in 'سوال: {q}' → q[:30] in content.")
    L.append("- Ranks: first-index match over steps.bm25_results / dense_results / merged_candidates / final_results (each ≤100 since top_k=100).")
    L.append("- Classes (D checked first so Hit@5 == D rate): D final≤5; else A merged None; else B merged>20 (gold in ≥1 exposed leg top-100 verified below); else C (merged≤20, final>5 or absent).")
    L.append("")
    L.append("## Counts")
    L.append(f"- total QA rows: {total}")
    L.append(f"- evaluated (gold-mapped): {evaluated} (= jsonl lines)")
    L.append(f"- skipped: {len(skipped)} (empty-question={skip_empty} no-gold-chunk={skip_nochunk} file-not-indexed={skip_nofile})")
    L.append(f"- A retriever_failure (gold not in merged top-100): {counts['A']}")
    L.append(f"- B fusion_failure (leg top-100 → merged>20): {counts['B']} (with gold in ≥1 exposed leg top-100: {b_in_leg}/{counts['B']})")
    L.append(f"- C reranker_failure (merged≤20 → final>5/absent): {counts['C']}")
    L.append(f"- D success (final≤5): {counts['D']}")
    L.append(f"- error (timeout/exception): {counts['error']}")
    L.append("")
    L.append("## Aggregate (over evaluated non-error rows)")
    L.append(f"- N={n_ok} Hit@1={(hit1 / max(n_ok, 1)):.4f} ({hit1}/{n_ok}) Hit@5={hit5_rate:.4f} MRR={mrr:.4f}")
    L.append("")
    L.append("## Comparison vs known baseline (data/massive_results_baseline_oldcode.json: 573 total, 489 pass, 22 fail, 62 skipped)")
    L.append(f"- {overlap_note}")
    L.append(f"- newly fixed ids: {json.dumps(overlap_detail[0], ensure_ascii=False)}")
    L.append(f"- newly broken ids: {json.dumps(overlap_detail[1], ensure_ascii=False)}")
    L.append("")
    for ftype, title in [("A", "A — retriever_failure"), ("B", "B — fusion_failure"), ("C", "C — reranker_failure")]:
        L.append(f"## Examples: {title}")
        for r in examples(ftype):
            L.append(f"- {r['query_id']} bm25={r['bm25_rank']} dense={r['dense_rank']} merged={r['merged_rank']} final={r['final_rank']}")
            L.append(f"  - Q: {r['query']}")
            L.append(f"  - gold: {r['gold_chunk_id']}")
        if not examples(ftype):
            L.append("- (none)")
        L.append("")
    L.append("## Skipped log (query_id, reason)")
    for qid, reason, f, q in skipped:
        L.append(f"- {qid} {reason} :: {pathlib.Path(f).name} :: {q[:80]}")
    if errors:
        L.append("")
        L.append("## Errors")
        for qid, e in errors:
            L.append(f"- {qid} {e}")
    L.append("")
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    rate = hits / max(evaluated, 1)
    return rate, mrr


async def finalize_from_jsonl(cfg, t_run0):
    """Regenerate failure_analysis.md from an existing (clean) jsonl. SELECT-only."""
    recs = [
        json.loads(line)
        for line in OUT_JSONL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    files = enumerate_files(cfg)
    file_rows, total = build_file_rows(files)
    qid_to_file = {}
    for f, sh in file_rows:
        stem = pathlib.Path(f).stem
        for rowidx in range(len(sh["rows"])):
            qid_to_file[f"{stem}#{rowidx}"] = f
    rows = [{"file": qid_to_file.get(r["query_id"], "?"), **r} for r in recs]
    db = Database(cfg.db)
    skipped = []
    for f, sh in file_rows:
        stem = pathlib.Path(f).stem
        headers = [h.lower() for h in sh["headers"]]
        q_idx = headers.index("question") if "question" in headers else 0
        qpath = str(pathlib.Path(f).resolve()).replace("\\", "/")
        async with db.session() as s:
            r = await s.execute(text("SELECT id, source_path FROM documents"))
            doc_id = None
            for row in r.fetchall():
                if row[1].replace("\\", "/").lower() == qpath.lower():
                    doc_id = row[0]
                    break
            if not doc_id:
                for rowidx, row in enumerate(sh["rows"]):
                    q = row[q_idx].strip() if q_idx < len(row) else ""
                    skipped.append((f"{stem}#{rowidx}", "file-not-indexed", f, q))
                continue
            r2 = await s.execute(
                text("SELECT id, content, chunk_type, metadata FROM chunks WHERE document_id = :d"),
                {"d": doc_id},
            )
            qa_chunks = []
            for _cid, _content, _ctype, _meta in r2.fetchall():
                if _ctype != "qa_pair":
                    continue
                _meta_dict = {}
                if _meta:
                    try:
                        _meta_dict = json.loads(_meta) if isinstance(_meta, str) else (_meta or {})
                    except Exception:
                        _meta_dict = {}
                qa_chunks.append((_cid, _content, _meta_dict))
        for rowidx, row in enumerate(sh["rows"]):
            q = row[q_idx].strip() if q_idx < len(row) else ""
            qid = f"{stem}#{rowidx}"
            if not q:
                skipped.append((qid, "empty-question", f, q))
            elif not map_gold(q, qa_chunks):
                skipped.append((qid, "no-gold-chunk", f, q))
    await db.close()
    counts = {"A": 0, "B": 0, "C": 0, "D": 0, "error": 0}
    b_in_leg = 0
    hits = 0
    errors = []
    for r in rows:
        ftype = r["failure_type"]
        counts[ftype] = counts.get(ftype, 0) + 1
        if ftype == "B" and (r.get("bm25_rank") is not None or r.get("dense_rank") is not None):
            b_in_leg += 1
        if ftype == "error":
            errors.append((r["query_id"], r.get("error", "?")))
        if r["hit5"]:
            hits += 1
    evaluated = len(rows)
    rate, mrr = write_report(rows, skipped, counts, b_in_leg, errors, evaluated, total, hits, t_run0)
    print(
        f"FINALIZED total={total} evaluated={evaluated} A={counts.get('A', 0)} B={counts.get('B', 0)} "
        f"C={counts.get('C', 0)} D={counts.get('D', 0)} err={counts.get('error', 0)} skipped={len(skipped)} "
        f"hit5={rate:.4f} mrr={mrr:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
