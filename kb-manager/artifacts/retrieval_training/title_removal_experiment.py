"""A/B experiment: does removing the doc title (and heading) from the dense
embedding text change gold recall?

Compares, for all 714 gold-mapped queries:
  - dense rank of the gold with contextual prefix  (Title/Heading/Type/Content)  [current]
  - dense rank of the gold with content only                                     [proposed]
Reports recall@1/@5/@10/@20/@50/@100 for each and per-file deltas.
"""
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


def build_context_text(content, title, heading, ctype):
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if heading:
        parts.append(f"Heading: {heading}")
    if ctype == "qa_pair":
        parts.append("Type: Q&A")
    parts.append(f"Content: {content}")
    return "\n".join(parts)


async def main():
    rows = [json.loads(l) for l in JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"loaded {len(rows)} evaluated rows", flush=True)

    cfg = load_config()
    db = Database(cfg.db)
    chunk_ids, contents, titles, headings, ctypes = [], [], [], [], []
    async with db.session() as s:
        r = await s.execute(text(
            "SELECT c.id, c.content, c.heading_path, c.chunk_type, d.title "
            "FROM chunks c JOIN documents d ON d.id = c.document_id "
            "WHERE c.chunk_type NOT LIKE '%_parent'"
        ))
        for cid, content, heading, ctype, title in r.fetchall():
            chunk_ids.append(cid)
            contents.append(content or "")
            headings.append(heading or "")
            ctypes.append(ctype or "")
            titles.append(title or "")
    await db.close()
    print(f"indexed chunks: {len(chunk_ids)}", flush=True)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL)

    print("encoding chunks WITH context ...", flush=True)
    ctx_texts = [build_context_text(contents[i], titles[i], headings[i], ctypes[i]) for i in range(len(chunk_ids))]
    M_ctx = model.encode(ctx_texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    print("encoding chunks CONTENT-ONLY ...", flush=True)
    M_raw = model.encode(contents, batch_size=32, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    print("encoding queries ...", flush=True)
    qs = [r["query"] for r in rows]
    Q = model.encode(qs, batch_size=32, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    id_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}

    def gold_rank(mat, qi, gold_id):
        gi = id_to_idx.get(gold_id)
        if gi is None:
            return None
        sims = mat @ mat[qi]  # unused; placeholder
        return None

    # proper: query row i dot all chunk rows
    def ranks_for(mat):
        # mat: (n_chunks, d); Q: (n_queries, d)
        sims = Q @ mat.T  # (n_queries, n_chunks)
        out = []
        for i, r in enumerate(rows):
            gi = id_to_idx.get(r["gold_chunk_id"])
            if gi is None:
                out.append(None)
                continue
            # rank = 1 + number of chunks with strictly greater sim
            s = sims[i]
            rank = int((s > s[gi]).sum()) + 1
            out.append(rank)
        return out

    print("scoring ...", flush=True)
    r_ctx = ranks_for(M_ctx)
    r_raw = ranks_for(M_raw)

    def recall(ranks, k):
        vals = [x for x in ranks if x is not None]
        return sum(1 for x in vals if x <= k) / len(vals)

    print()
    print("=" * 66)
    print("DENSE GOLD RECALL — context prefix vs content-only")
    print("=" * 66)
    print(f"{'k':>6} {'WITH context':>16} {'CONTENT-ONLY':>16} {'delta':>8}")
    for k in [1, 5, 10, 20, 50, 100]:
        a, b = recall(r_ctx, k), recall(r_raw, k)
        print(f"{k:>6} {a:>16.4f} {b:>16.4f} {b-a:>+8.4f}")

    # per-file for recall@100
    print()
    print("Per-file recall@100 (context -> content-only):")
    files = sorted({r["query_id"].split("#")[0] for r in rows})
    for f in files:
        idx = [i for i, r in enumerate(rows) if r["query_id"].startswith(f + "#")]
        a = recall([r_ctx[i] for i in idx], 100)
        b = recall([r_raw[i] for i in idx], 100)
        print(f"  {f:42s} {a:.3f} -> {b:.3f} ({b-a:+.3f})")

    # cheque-bias: among failures, how often does a Cheque doc top the dense list?
    print()
    print("Top-1 dense doc-title distribution for the 54 failures (context vs raw):")
    fail_idx = [i for i, r in enumerate(rows) if r["failure_type"] in ("A", "B", "C")]
    for label, mat in [("context", M_ctx), ("content-only", M_raw)]:
        sims = Q @ mat.T
        from collections import Counter
        c = Counter()
        for i in fail_idx:
            j = int(np.argmax(sims[i]))
            c[titles[j]] += 1
        print(f"  [{label}] {dict(c.most_common(6))}")

    # save ranks for the report
    out = {
        "recall": {k: {"context": recall(r_ctx, k), "content_only": recall(r_raw, k)} for k in [1, 5, 10, 20, 50, 100]},
        "per_query": [
            {"query_id": rows[i]["query_id"], "failure_type": rows[i]["failure_type"],
             "rank_ctx": r_ctx[i], "rank_raw": r_raw[i]}
            for i in range(len(rows))
        ],
    }
    (REPO / "artifacts" / "retrieval_training" / "title_removal_experiment.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nsaved artifacts/retrieval_training/title_removal_experiment.json", flush=True)


asyncio.run(main())
