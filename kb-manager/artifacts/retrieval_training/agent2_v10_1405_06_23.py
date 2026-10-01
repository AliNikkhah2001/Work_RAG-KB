"""Agent 2 (v10_1405-06-23 variant) — stage-level failure analysis over ALL QA rows.

Same A/B/C/D methodology as agent2_failure_analysis.py, retargeted at the
v10 corpus: data/kb_1405_06_23.db + data/v10_source/1405-06-23.
Reranker pinned to mmarco (same as frozen v10 baseline) via KB_RERANKER_MODEL.

Outputs (v10-suffixed, frozen baseline files untouched):
  artifacts/retrieval_training/retrieval_failures_v10_1405-06-23.jsonl
  docs/retrieval_training/failure_analysis_v10_1405-06-23.md

Usage:
  python artifacts/retrieval_training/agent2_v10_1405_06_23.py [--enumerate-only] [--resume] [--finalize-only]
"""

import os as _os

# --- MUST be set before any kb_manager import ---
_os.environ["KB_DB_URL"] = "sqlite+aiosqlite:///D:/Code/KB/kb-manager/data/kb_1405_06_23.db"
_os.environ["KB_SOURCE_DIR"] = "D:/Code/KB/kb-manager/data/v10_source/1405-06-23"
# Pin mmarco reranker (same as frozen v10 baseline) unless caller overrides.
_os.environ.setdefault("KB_RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

import asyncio
import gc
import hashlib
import json
import pathlib
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime

# Lean mode (KB_BENCH_LEAN=1): cap native threads + collect garbage per row.
# Changes NO retrieval math (same TOP_K/beam/model); only bounds host RAM
# after the Cheque-slice OOM cascade (55x MemoryError/paging-1455).
if _os.getenv("KB_BENCH_LEAN") == "1":
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        _os.environ.setdefault(_v, "1")
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

# Repo root must be importable (script lives in artifacts/, not on sys.path).
sys.path.insert(0, "D:/Code/KB/kb-manager")

ENUMERATE_ONLY = "--enumerate-only" in sys.argv
RESUME = "--resume" in sys.argv
FINALIZE_ONLY = "--finalize-only" in sys.argv

REPO = pathlib.Path("D:/Code/KB/kb-manager")
OUT_JSONL = REPO / "artifacts" / "retrieval_training" / "retrieval_failures_v10_1405-06-23.jsonl"
OUT_MD = REPO / "docs" / "retrieval_training" / "failure_analysis_v10_1405-06-23.md"
BASELINE_OLD = REPO / "data" / "massive_results_baseline_oldcode.json"
FREEZE_JSON = REPO / "artifacts" / "retrieval_training" / "baseline_freeze.json"
TEMP_PROGRESS = pathlib.Path(_os.environ.get("TEMP", ".")) / "agent2_v10_progress.json"

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


def top_k_list(results, k=10):
    """Serialize top-k results from a retrieval stage. Never returns -1."""
    out = []
    for r in results[:k]:
        entry = {
            "chunk_id": r.chunk_id,
            "doc_title": r.doc_title,
            "heading_path": r.heading_path,
            "content_preview": (r.content_preview or "")[:200],
        }
        if getattr(r, "bm25_score", 0):
            entry["bm25_score"] = round(r.bm25_score, 4)
        if getattr(r, "dense_score", 0):
            entry["dense_score"] = round(r.dense_score, 4)
        if getattr(r, "rerank_score", 0):
            entry["rerank_score"] = round(r.rerank_score, 4)
        if getattr(r, "hybrid_score", 0):
            entry["hybrid_score"] = round(r.hybrid_score, 4)
        out.append(entry)
    return out


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
        _pl.Path("D:/Code/KB/kb-manager/data/v10_source/1405-06-23"),
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
    """A/B/C/D/A' classification with correct A vs A' split.

    A  = retriever miss: gold absent from BOTH bm25 AND dense top-100
    A' = fusion loss:    gold found by ONE leg but RRF dropped it
    B  = reranker loss:  gold in merged top-100 but reranker drops from final top-5
    D  = success:        gold in final top-5
    """
    hit5 = final_rank is not None and final_rank <= 5
    if hit5:
        return "D", True
    gold_in_bm25 = bm25_rank is not None
    gold_in_dense = dense_rank is not None
    if not gold_in_bm25 and not gold_in_dense:
        return "A", False  # true retriever miss — neither leg found it
    if merged_rank is None:
        return "A'", False  # fusion loss — one leg found it, RRF killed it
    return "B", False  # gold in merged but reranker dropped it


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
    counts = {"A": 0, "A'": 0, "B": 0, "C": 0, "D": 0, "error": 0}
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
                # Ensure new fields exist for old-format rows (v1 schema)
                rec.setdefault("reranker_model", "unknown")
                rec.setdefault("bm25_top10", [])
                rec.setdefault("dense_top10", [])
                rec.setdefault("merged_top10", [])
                rec.setdefault("final_top5", [])
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
                bm25_top = top_k_list(steps.bm25_results, k=10)
                dense_top = top_k_list(steps.dense_results, k=10)
                merged_top = top_k_list(steps.merged_candidates, k=10)
                final_top = top_k_list(steps.final_results, k=5)
            except Exception as e:
                bm25_rank = dense_rank = merged_rank = final_rank = None
                bm25_top = dense_top = merged_top = final_top = []
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
                "reranker_model": _os.getenv("KB_RERANKER_MODEL", "unknown"),
                "bm25_rank": bm25_rank,
                "dense_rank": dense_rank,
                "merged_rank": merged_rank,
                "final_rank": final_rank,
                "bm25_top10": bm25_top,
                "dense_top10": dense_top,
                "merged_top10": merged_top,
                "final_top5": final_top,
                "hit5": hit5,
                "failure_type": ftype,
                "elapsed_ms": elapsed_ms,
            }
            if err is not None:
                rec["error"] = err
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if _os.getenv("KB_BENCH_LEAN") == "1":
                gc.collect()
            rows.append({"file": f, **rec})
            done += 1
            if done % 25 == 0 or done == total:
                rate = hits / max(evaluated, 1)
                # Stage-level quick stats
                ok_recs = [r for r in rows if r.get("failure_type") != "error"]
                n_ok_q = len(ok_recs)
                s_bm25 = sum(1 for r in ok_recs if r.get("bm25_rank") is not None and r["bm25_rank"] <= 5)
                s_dense = sum(1 for r in ok_recs if r.get("dense_rank") is not None and r["dense_rank"] <= 5)
                s_merged = sum(1 for r in ok_recs if r.get("merged_rank") is not None and r["merged_rank"] <= 5)
                print(f"{done}/{total} hit5={rate:.4f} eval={evaluated} stage_Hit@5: bm25={s_bm25} dense={s_dense} merged={s_merged} final={hits}", flush=True)
                try:
                    TEMP_PROGRESS.write_text(
                        json.dumps(
                            {"done": done, "total": total, "evaluated": evaluated,
                             "hits": hits, "errors": len(errors),
                             "stage_hit5": {"bm25": s_bm25, "dense": s_dense, "merged": s_merged, "final": hits}}
                        ),
                        encoding="utf-8",
                    )
                except Exception:
                    pass
    fout.close()
    await db.close()
    rate, mrr = write_report(rows, skipped, counts, b_in_leg, errors, evaluated, total, hits, t_run0)
    a_prime = counts.get("A'", 0)
    print(
        f"DONE total={total} evaluated={evaluated} A={counts['A']} A'={a_prime} B={counts['B']} "
        f"D={counts['D']} err={counts['error']} skipped={len(skipped)} "
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
    db_path = REPO / "data" / "kb_1405_06_23.db"
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
    L.append("# Retrieval failure analysis (Agent 2 — v10_1405-06-23 corpus)")
    L.append("")
    L.append("## Header / determinism")
    L.append(f"- code HEAD: {head_rev}")
    L.append(f"- corpus: v10_1405-06-23")
    L.append(f"- kb_manager/ status clean: {str(status == '')} (status={status!r})")
    L.append(f"- db: data/kb_1405_06_23.db sha256={db_sha}")
    L.append(f"- npz: data/dense_embeddings.npz sha256={npz_sha}")
    L.append(f"- models: dense=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 reranker={rows[0].get('reranker_model', 'unknown') if rows else 'unknown'}")
    L.append(f"- retrieval config: rrf_k=60 rerank_top_k=100 keyword_boost=3.0 synonym_beam=5 fusion_alpha=0.7")
    L.append(f"- env: KB_DB_URL=sqlite+aiosqlite:///D:/Code/KB/kb-manager/data/kb_1405_06_23.db KB_SOURCE_DIR=D:/Code/KB/kb-manager/data/v10_source/1405-06-23")
    L.append(f"- run at (UTC): {datetime.now(UTC).isoformat()} top_k=100 per-query-timeout=600s")
    L.append(f"- elapsed: {elapsed_min:.1f} min; pipeline deterministic (CPU), single run")
    L.append("")
    L.append("## Method (replica of benchmarks.py::_run_massive)")
    L.append("- QA files: rglob *.xlsx minus ~$/TestQuestion/واژگان معادل/محدودیت ها stems, first dir with crm_qa sheets; GT 3-step: metadata.fields.question==q → content startswith/in 'سوال: {q}' → q[:30] in content.")
    L.append("- Ranks: first-index match over steps.bm25_results / dense_results / merged_candidates / final_results (each ≤100 since top_k=100).")
    L.append("- Classes (D checked first so Hit@5 == D rate): D final≤5; else A neither leg found gold in top-100; else A' one leg found but RRF dropped; else B gold in merged top-100 but reranker dropped from final top-5.")
    L.append("")
    L.append("## Counts")
    L.append(f"- total QA rows: {total}")
    L.append(f"- evaluated (gold-mapped): {evaluated} (= jsonl lines)")
    L.append(f"- skipped: {len(skipped)} (empty-question={skip_empty} no-gold-chunk={skip_nochunk} file-not-indexed={skip_nofile})")
    L.append(f"- A retriever_failure (gold not in bm25 AND not in dense top-100): {counts['A']}")
    a_prime = counts.get("A'", 0)
    L.append(f"- A' fusion_loss (gold in ≥1 leg top-100 but RRF dropped it): {a_prime}")
    L.append(f"- B reranker_failure (gold in merged top-100 but final>5): {counts['B']}")
    L.append(f"- D success (final≤5): {counts['D']}")
    L.append(f"- error (timeout/exception): {counts['error']}")
    L.append("")
    L.append("## Aggregate (over evaluated non-error rows)")
    L.append(f"- N={n_ok} Hit@1={(hit1 / max(n_ok, 1)):.4f} ({hit1}/{n_ok}) Hit@5={hit5_rate:.4f} MRR={mrr:.4f}")
    # Stage-level hit rates
    bm25_hit5 = sum(1 for r in ok_rows if r.get("bm25_rank") is not None and r["bm25_rank"] <= 5)
    dense_hit5 = sum(1 for r in ok_rows if r.get("dense_rank") is not None and r["dense_rank"] <= 5)
    merged_hit5 = sum(1 for r in ok_rows if r.get("merged_rank") is not None and r["merged_rank"] <= 5)
    final_hit5 = sum(1 for r in ok_rows if r["hit5"])
    bm25_hit100 = sum(1 for r in ok_rows if r.get("bm25_rank") is not None)
    dense_hit100 = sum(1 for r in ok_rows if r.get("dense_rank") is not None)
    merged_hit100 = sum(1 for r in ok_rows if r.get("merged_rank") is not None)
    L.append(f"- Stage Hit@5:  bm25={bm25_hit5}/{n_ok} ({bm25_hit5/max(n_ok,1):.4f}) dense={dense_hit5}/{n_ok} ({dense_hit5/max(n_ok,1):.4f}) merged={merged_hit5}/{n_ok} ({merged_hit5/max(n_ok,1):.4f}) final={final_hit5}/{n_ok} ({final_hit5/max(n_ok,1):.4f})")
    L.append(f"- Stage recall@100: bm25={bm25_hit100}/{n_ok} ({bm25_hit100/max(n_ok,1):.4f}) dense={dense_hit100}/{n_ok} ({dense_hit100/max(n_ok,1):.4f}) merged={merged_hit100}/{n_ok} ({merged_hit100/max(n_ok,1):.4f})")
    L.append("")
    L.append("## Failure funnel")
    true_retriever_miss = counts["A"]
    fusion_loss = counts.get("A'", 0)
    reranker_loss = counts["B"]
    L.append(f"- True retriever miss (A): {true_retriever_miss} — neither BM25 nor Dense found gold in top-100")
    L.append(f"- Fusion loss (A'): {fusion_loss} — one leg found gold, RRF dropped it")
    L.append(f"- Reranker loss (B): {reranker_loss} — gold in merged top-100, reranker dropped from final top-5")
    L.append(f"- Retriever recall (any leg): {n_ok - true_retriever_miss}/{n_ok} ({(n_ok - true_retriever_miss)/max(n_ok,1):.4f})")
    L.append(f"- Fusion retention: {n_ok - true_retriever_miss - fusion_loss}/{n_ok - true_retriever_miss} ({(n_ok - true_retriever_miss - fusion_loss)/max(n_ok - true_retriever_miss,1):.4f})")
    L.append(f"- Reranker rescue: {final_hit5}/{n_ok - true_retriever_miss - fusion_loss} ({final_hit5/max(n_ok - true_retriever_miss - fusion_loss,1):.4f})")
    L.append("")
    # Per-file breakdown
    file_groups: dict[str, list] = {}
    for r in ok_rows:
        fname = pathlib.Path(r.get("file", "?")).stem
        file_groups.setdefault(fname, []).append(r)
    L.append("## Per-file breakdown")
    L.append(f"| File | n | A | A' | B | D | hit5 |")
    L.append(f"|------|---|---|----|----|----|----|")
    for fname in sorted(file_groups):
        fr = file_groups[fname]
        f_a = sum(1 for r in fr if r["failure_type"] == "A")
        f_ap = sum(1 for r in fr if r["failure_type"] == "A'")
        f_b = sum(1 for r in fr if r["failure_type"] == "B")
        f_d = sum(1 for r in fr if r["hit5"])
        f_n = len(fr)
        f_hit5 = f_d / max(f_n, 1)
        L.append(f"| {fname} | {f_n} | {f_a} | {f_ap} | {f_b} | {f_d} | {f_hit5:.3f} |")
    L.append("")
    L.append("## Comparison vs frozen v10 baseline (retrieval_failures.jsonl: 573 total, 489 pass)")
    L.append(f"- {overlap_note}")
    L.append(f"- newly fixed ids: {json.dumps(overlap_detail[0], ensure_ascii=False)}")
    L.append(f"- newly broken ids: {json.dumps(overlap_detail[1], ensure_ascii=False)}")
    L.append("")
    for ftype, title in [("A", "A — retriever_failure (neither leg)"), ("A'", "A' — fusion_loss (one leg, RRF killed)"), ("B", "B — reranker_failure")]:
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
    counts = {"A": 0, "A'": 0, "B": 0, "C": 0, "D": 0, "error": 0}
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
    a_prime = counts.get("A'", 0)
    print(
        f"FINALIZED total={total} evaluated={evaluated} A={counts.get('A', 0)} A'={a_prime} B={counts.get('B', 0)} "
        f"D={counts.get('D', 0)} err={counts.get('error', 0)} skipped={len(skipped)} "
        f"hit5={rate:.4f} mrr={mrr:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
