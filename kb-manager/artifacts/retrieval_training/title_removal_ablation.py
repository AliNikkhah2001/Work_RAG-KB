"""Isolate which contextual prefix component hurts dense retrieval:
full / no-title / no-heading / no-type / content-only."""
import os, sys, json, asyncio, pathlib
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
os.environ["KB_DB_URL"] = "sqlite+aiosqlite:///D:/Code/KB/kb-manager/data/kb_1405_06_23.db"
sys.path.insert(0, "D:/Code/KB/kb-manager")
from sqlalchemy import text
from kb_manager.models.database import Database
from kb_manager.config import load_config

REPO = pathlib.Path("D:/Code/KB/kb-manager")
JSONL = REPO / "artifacts" / "retrieval_training" / "retrieval_failures_v10_1405-06-23.jsonl"
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def ctx(content, title, heading, ctype, use_title=True, use_heading=True, use_type=True):
    parts = []
    if use_title and title:
        parts.append(f"Title: {title}")
    if use_heading and heading:
        parts.append(f"Heading: {heading}")
    if use_type and ctype == "qa_pair":
        parts.append("Type: Q&A")
    parts.append(f"Content: {content}")
    return "\n".join(parts)


async def main():
    rows = [json.loads(l) for l in JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    cfg = load_config()
    db = Database(cfg.db)
    ids, contents, titles, headings, ctypes = [], [], [], [], []
    async with db.session() as s:
        r = await s.execute(text(
            "SELECT c.id, c.content, c.heading_path, c.chunk_type, d.title "
            "FROM chunks c JOIN documents d ON d.id = c.document_id "
            "WHERE c.chunk_type NOT LIKE '%_parent'"))
        for cid, content, heading, ctype, title in r.fetchall():
            ids.append(cid); contents.append(content or ""); headings.append(heading or "")
            ctypes.append(ctype or ""); titles.append(title or "")
    await db.close()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL)
    Q = model.encode([r["query"] for r in rows], batch_size=32, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    id_to_idx = {cid: i for i, cid in enumerate(ids)}

    variants = {
        "full (Title+Heading+Type+Content)": dict(),
        "NO TITLE (Heading+Type+Content)": dict(use_title=False),
        "NO HEADING (Title+Type+Content)": dict(use_heading=False),
        "NO TYPE (Title+Heading+Content)": dict(use_type=False),
        "CONTENT ONLY": dict(use_title=False, use_heading=False, use_type=False),
    }

    def recall(ranks, k):
        vals = [x for x in ranks if x is not None]
        return sum(1 for x in vals if x <= k) / len(vals)

    results = {}
    for label, kw in variants.items():
        texts = [ctx(contents[i], titles[i], headings[i], ctypes[i], **kw) for i in range(len(ids))]
        M = model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        sims = Q @ M.T
        ranks = []
        for i, r in enumerate(rows):
            gi = id_to_idx.get(r["gold_chunk_id"])
            if gi is None:
                ranks.append(None); continue
            s = sims[i]
            ranks.append(int((s > s[gi]).sum()) + 1)
        results[label] = {k: recall(ranks, k) for k in [1, 5, 10, 20, 50, 100]}
        print(f"{label:38s} " + " ".join(f"@{k}={results[label][k]:.3f}" for k in [1, 5, 10, 100]), flush=True)

    print()
    print("=" * 78)
    print("DENSE GOLD RECALL — prefix component ablation")
    print("=" * 78)
    ks = [1, 5, 10, 20, 50, 100]
    print(f"{'variant':38s} " + " ".join(f"{('@'+str(k)):>8}" for k in ks))
    for label in variants:
        print(f"{label:38s} " + " ".join(f"{results[label][k]:>8.4f}" for k in ks))

    (REPO / "artifacts" / "retrieval_training" / "title_removal_ablation.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nsaved title_removal_ablation.json", flush=True)


asyncio.run(main())
