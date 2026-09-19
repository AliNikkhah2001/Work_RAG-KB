"""Build an LLM-judge corpus: for each failure, full gold text vs the actual
retrieved winner text, plus word-level overlap stats (token Jaccard, shared
content words, length ratio). Emits one markdown file per group.
"""
import os, sys, json, asyncio, pathlib, re
sys.stdout.reconfigure(encoding="utf-8")
os.environ["KB_DB_URL"] = "sqlite+aiosqlite:///D:/Code/KB/kb-manager/data/kb_1405_06_23.db"
sys.path.insert(0, "D:/Code/KB/kb-manager")
from sqlalchemy import text
from kb_manager.models.database import Database
from kb_manager.config import load_config

REPO = pathlib.Path("D:/Code/KB/kb-manager")
JSONL = REPO / "artifacts" / "retrieval_training" / "retrieval_failures_v10_1405-06-23.jsonl"
OUT = REPO / "artifacts" / "retrieval_training" / "rca_judge"
OUT.mkdir(parents=True, exist_ok=True)

STOP = {"و","در","به","از","که","این","را","با","است","برای","آن","یک","می","های","ها","یا","تا","هم","بر","شد","شده","بود","اگر","چه","چی","چرا","کدام","آیا","من","ما","شما","او","توی","تو","رو","هم","کن","کند","کنم","کنید","دارد","دارم","دارند","نیست","هست","باید","میشود","میشود","میتوان","توان","کند","بر","دیگر","همه","هر","خود"}


def toks(s):
    s = re.sub(r"[\u200c\u200f\u200e]", " ", s or "")
    s = re.sub(r"[^\w\s]", " ", s)
    return [w for w in s.split() if len(w) > 1]


def content_words(s):
    return set(w for w in toks(s) if w not in STOP)


async def main():
    rows = [json.loads(l) for l in JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    fails = [r for r in rows if r["failure_type"] in ("A", "B", "C")]
    cfg = load_config()
    db = Database(cfg.db)

    # full content for every chunk id we need
    need = set()
    for r in fails:
        need.add(r["gold_chunk_id"])
        for k in ("bm25_top10", "dense_top10", "merged_top10", "final_top5"):
            for it in r.get(k, [])[:2]:
                need.add(it["chunk_id"])
    content = {}
    ctype = {}
    async with db.session() as s:
        for cid in need:
            r = await s.execute(text("SELECT content, chunk_type FROM chunks WHERE id=:i"), {"i": cid})
            row = r.fetchone()
            if row:
                content[cid] = row[0] or ""
                ctype[cid] = row[1] or ""
    await db.close()

    groups = {
        "A_company": [r for r in fails if r["failure_type"] == "A" and r["query_id"].startswith("Company_CRM_Questions")],
        "A_individual": [r for r in fails if r["failure_type"] == "A" and r["query_id"].startswith("IndividualCRMQuestions")],
        "A_public": [r for r in fails if r["failure_type"] == "A" and not r["query_id"].startswith(("Company_CRM_Questions", "IndividualCRMQuestions"))],
        "BC": [r for r in fails if r["failure_type"] in ("B", "C")],
    }

    def winner(r):
        # the best actual retrieved chunk: final #1 if present else merged #1 else best leg #1
        for k in ("final_top5", "merged_top10", "dense_top10", "bm25_top10"):
            if r.get(k):
                return r[k][0]["chunk_id"], k
        return None, None

    for name, items in groups.items():
        L = [f"# LLM-judge corpus — {name} ({len(items)} failures)\n"]
        for n, r in enumerate(items, 1):
            gold = content.get(r["gold_chunk_id"], "(missing)")
            wid, wstage = winner(r)
            wtext = content.get(wid, "(missing)") if wid else "(none)"
            qw = content_words(r["query"])
            gw = content_words(gold)
            ww = content_words(wtext)
            jac_gw = len(qw & gw) / max(len(qw | gw), 1)
            jac_qw = len(qw & ww) / max(len(qw | ww), 1)
            L.append(f"## [{n}] {r['query_id']} (type {r['failure_type']})")
            L.append(f"**QUERY:** {r['query']}")
            L.append(f"")
            L.append(f"**GOLD** [{ctype.get(r['gold_chunk_id'],'?')}]  (ranks bm25={r.get('bm25_rank')} dense={r.get('dense_rank')} merged={r.get('merged_rank')} final={r.get('final_rank')})")
            L.append(f"```")
            L.append(gold)
            L.append(f"```")
            L.append(f"**ACTUAL RETRIEVED WINNER** (stage={wstage})")
            L.append(f"```")
            L.append(wtext)
            L.append(f"```")
            L.append(f"**Word stats:** query↔gold Jaccard={jac_gw:.3f} (shared {len(qw & gw)}: {sorted(qw & gw)[:15]}); query↔winner Jaccard={jac_qw:.3f} (shared {len(qw & ww)}: {sorted(qw & ww)[:15]}); gold_len={len(gold)} winner_len={len(wtext)}")
            L.append(f"**Gold content words absent from winner:** {sorted(gw - ww)[:25]}")
            L.append("")
        (OUT / f"judge_{name}.md").write_text("\n".join(L), encoding="utf-8")
        print(f"{name}: {len(items)} -> judge_{name}.md", flush=True)


asyncio.run(main())
