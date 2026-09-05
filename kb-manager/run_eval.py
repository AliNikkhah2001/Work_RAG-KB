import json
import sqlite3

# Load dataset
import os, pathlib
_BASE = pathlib.Path(__file__).resolve().parent
ds = os.getenv("KB_EVAL_JSON", str(_BASE / "kb_manager" / "evaluation" / "datasets" / "eval_full.json"))
if not pathlib.Path(ds).exists():
    ds = str(_BASE / "data" / "test_questions.json")
with open(ds, "r", encoding="utf-8") as f:
    dataset = json.load(f)

dbp = os.getenv("KB_DB_PATH", str(_BASE / "data" / "kb_test.db"))
conn = sqlite3.connect(dbp)
cur = conn.cursor()

results = []
for item in dataset:
    query = item["query"]
    expected_ids = item["expected_chunk_ids"]
    expected_set = set(expected_ids)
    terms = query.split()
    
    # Search
    conditions = " OR ".join(["content LIKE ?" for _ in terms])
    params = [f"%{t}%" for t in terms]
    cur.execute(f"SELECT id, content FROM chunks WHERE {conditions} LIMIT ?", params + [20])
    rows = cur.fetchall()
    
    # Score
    scored = []
    for cid, content in rows:
        content_lower = content.lower()
        score = sum(1 for t in terms if t.lower() in content_lower)
        scored.append((cid, score / len(terms) if terms else 0.0))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    retrieved = [cid for cid, _ in scored[:10]]
    
    # Check if expected in top 10
    found = any(eid in retrieved for eid in expected_set)
    results.append({
        "query": query,
        "expected": expected_ids,
        "retrieved": retrieved,
        "found": found
    })

# Compute metrics
found_count = sum(1 for r in results if r["found"])
total = len(results)
hit_rate = found_count / total if total > 0 else 0

with open("eval_results.txt", "w", encoding="utf-8") as f:
    f.write(f"Total queries: {total}\n")
    f.write(f"Hit rate@10: {hit_rate:.4f}\n")
    f.write(f"Found: {found_count}/{total}\n\n")
    
    # Show first 10 failures
    failures = [r for r in results if not r["found"]]
    f.write(f"Failures: {len(failures)}\n")
    for r in failures[:10]:
        f.write(f"  Query: {r['query'][:80]}\n")
        f.write(f"  Expected: {r['expected']}\n")
        f.write(f"  Retrieved: {r['retrieved'][:5]}\n\n")

conn.close()
print("Done")