import sqlite3

conn = sqlite3.connect(r"C:\Users\10225\Downloads\KB\kb-manager\data\kb_test.db")
cur = conn.cursor()

query = "استانداردهای کدگذاری نوع شرکت، جنسیت و وضعیت تاهل چیست؟"
expected_id = "504bb823-4cdd-4266-9e0d-08d56bd4a14c"
k = 10

terms = query.split()
conditions = " OR ".join(["content LIKE ?" for _ in terms])
params = [f"%{t}%" for t in terms]

sql = f"SELECT id, content FROM chunks WHERE {conditions} LIMIT ?"
cur.execute(sql, params + [k * 2])
rows = cur.fetchall()

with open("debug_search.txt", "w", encoding="utf-8") as f:
    f.write(f"Query: {query}\n")
    f.write(f"Terms: {terms}\n")
    f.write(f"Expected ID: {expected_id}\n")
    f.write(f"Found {len(rows)} candidates\n\n")
    
    scored = []
    for cid, content in rows:
        content_lower = content.lower()
        score = sum(1 for t in terms if t.lower() in content_lower)
        scored.append((cid, score / len(terms) if terms else 0.0))
        if cid == expected_id:
            f.write(f"*** FOUND EXPECTED CHUNK! Score: {score / len(terms)}\n")
    
    scored.sort(key=lambda x: x[1], reverse=True)
    f.write(f"\nTop {k} results:\n")
    for i, (cid, score) in enumerate(scored[:k]):
        marker = " ***" if cid == expected_id else ""
        f.write(f"  {i+1}. {cid[:8]}... score={score:.3f}{marker}\n")

conn.close()