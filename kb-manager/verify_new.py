import sqlite3

conn = sqlite3.connect(r"C:\Users\10225\Downloads\KB\kb-manager\data\kb_test.db")
cur = conn.cursor()

# Sample QA chunks with new format
cur.execute("SELECT content, token_count, metadata FROM chunks WHERE chunk_type = 'qa_pair' LIMIT 3")
rows = cur.fetchall()

results = []
results.append("=== NEW QA CHUNK FORMAT ===")
for content, tc, meta in rows:
    results.append(f"TOKENS: {tc}")
    results.append(f"CONTENT:\n{content[:500]}")
    results.append("")

# Verify incomplete rows were skipped
cur.execute("SELECT COUNT(*) FROM chunks WHERE chunk_type = 'qa_pair' AND content NOT LIKE '%\u067e\u0627\u0633\u062e%'")
no_answer = cur.fetchone()[0]
results.append(f"QA chunks without answer: {no_answer} (should be 0)")

# Check parent chunks
cur.execute("SELECT COUNT(*) FROM chunks WHERE metadata LIKE '%is_parent%'")
parents = cur.fetchone()[0]
results.append(f"Parent chunks: {parents}")

# Check hierarchy
cur.execute("SELECT COUNT(*) FROM chunks WHERE parent_id IS NOT NULL")
with_parent = cur.fetchone()[0]
results.append(f"Chunks with parent_id: {with_parent}")

# Stats
cur.execute("SELECT chunk_type, COUNT(*) FROM chunks GROUP BY chunk_type")
for ct, cnt in cur.fetchall():
    results.append(f"Type {ct}: {cnt}")

with open(r"C:\Users\10225\Downloads\KB\kb-manager\verify_new_format.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results))
conn.close()
print("done")
