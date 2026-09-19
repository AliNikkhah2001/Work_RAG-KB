"""Dump all v10 failures with gold content + per-stage top-k into grouped markdown files for RCA."""
import os, sys, json, asyncio, pathlib
sys.stdout.reconfigure(encoding="utf-8")
os.environ["KB_DB_URL"] = "sqlite+aiosqlite:///D:/Code/KB/kb-manager/data/kb_1405_06_23.db"
os.environ["KB_SOURCE_DIR"] = "D:/Code/KB/kb-manager/data/v10_source/1405-06-23"
sys.path.insert(0, "D:/Code/KB/kb-manager")

from sqlalchemy import text
from kb_manager.models.database import Database
from kb_manager.config import load_config

REPO = pathlib.Path("D:/Code/KB/kb-manager")
OUT = REPO / "artifacts" / "retrieval_training" / "rca"
OUT.mkdir(parents=True, exist_ok=True)

JSONL = REPO / "artifacts" / "retrieval_training" / "retrieval_failures_v10_1405-06-23.jsonl"


def clip(s, n=200):
    s = (s or "").replace("\n", " | ")
    return s[:n] + ("…" if len(s) > n else "")


async def main():
    rows = [json.loads(l) for l in JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    fails = [r for r in rows if r["failure_type"] in ("A", "B", "C")]

    cfg = load_config()
    db = Database(cfg.db)

    # fetch gold chunk content + type for all gold ids
    gold_ids = list({r["gold_chunk_id"] for r in fails})
    gold = {}
    async with db.session() as s:
        for gid in gold_ids:
            r = await s.execute(
                text("SELECT content, chunk_type, document_id, metadata FROM chunks WHERE id = :i"),
                {"i": gid},
            )
            row = r.fetchone()
            if row:
                gold[gid] = {"content": row[0], "type": row[1], "doc": row[2], "meta": row[3]}
    # doc titles
    docs = {}
    async with db.session() as s:
        r = await s.execute(text("SELECT id, source_path FROM documents"))
        for did, sp in r.fetchall():
            docs[did] = sp.split("\\")[-1].split("/")[-1]
    await db.close()

    groups = {
        "A_company": [r for r in fails if r["failure_type"] == "A" and r["query_id"].startswith("Company_CRM_Questions")],
        "A_individual": [r for r in fails if r["failure_type"] == "A" and r["query_id"].startswith("IndividualCRMQuestions")],
        "A_public": [r for r in fails if r["failure_type"] == "A" and not r["query_id"].startswith(("Company_CRM_Questions", "IndividualCRMQuestions"))],
        "BC": [r for r in fails if r["failure_type"] in ("B", "C")],
    }

    def render(r, idx):
        L = []
        L.append(f"### [{idx}] {r['query_id']}  (type {r['failure_type']})")
        L.append(f"- **QUERY:** {r['query']}")
        g = gold.get(r["gold_chunk_id"], {})
        L.append(f"- **GOLD chunk** ({g.get('type','?')}, doc={docs.get(g.get('doc'),'?')}):")
        L.append(f"  > {clip(g.get('content','(not found)'), 500)}")
        L.append(f"- **Ranks:** bm25={r.get('bm25_rank')} dense={r.get('dense_rank')} merged={r.get('merged_rank')} final={r.get('final_rank')}")
        for stage, key, k in [("BM25", "bm25_top10", 10), ("DENSE", "dense_top10", 10), ("MERGED(RRF)", "merged_top10", 10), ("FINAL(rerank)", "final_top5", 5)]:
            items = r.get(key, [])
            L.append(f"- **{stage} top-{k}:**")
            for i, it in enumerate(items):
                sc = it.get("rerank_score", it.get("hybrid_score", it.get("bm25_score", it.get("dense_score", 0))))
                mark = "  <<< GOLD" if it["chunk_id"] == r["gold_chunk_id"] else ""
                L.append(f"  {i+1}. [{it.get('doc_title','?')}] score={sc} :: {clip(it.get('content_preview',''), 160)}{mark}")
        L.append("")
        return "\n".join(L)

    for name, items in groups.items():
        parts = [f"# v10 failure group: {name}  ({len(items)} samples)\n"]
        for i, r in enumerate(items, 1):
            parts.append(render(r, i))
        (OUT / f"group_{name}.md").write_text("\n".join(parts), encoding="utf-8")
        print(f"{name}: {len(items)} samples -> group_{name}.md", flush=True)

    # also a combined summary of failure ids
    print(f"TOTAL failures: {len(fails)}", flush=True)


asyncio.run(main())
