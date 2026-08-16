import sqlite3

conn = sqlite3.connect(r"C:\Users\10225\Downloads\KB\kb-manager\data\kb_test.db")
cur = conn.cursor()

# Find QA chunks that are missing answer fields
cur.execute("SELECT id, content, token_count, heading_path, metadata FROM chunks WHERE chunk_type = 'qa_pair' ORDER BY token_count ASC LIMIT 10")
short_chunks = cur.fetchall()

results = []
results.append("=== SHORTEST QA CHUNKS (likely broken) ===")
for cid, content, tc, hp, meta in short_chunks:
    results.append(f"ID: {cid[:8]}... TOKENS: {tc} HEADING: {hp}")
    results.append(f"CONTENT: {content[:300]}")
    results.append(f"METADATA: {meta}")
    results.append("")

# Find QA chunks that don't contain 'Answer:' or Persian equivalent
cur.execute("SELECT id, content, token_count FROM chunks WHERE chunk_type = 'qa_pair' AND content NOT LIKE '%Answer:%' AND content NOT LIKE '%\u067e\u0627\u0633\u062e%' LIMIT 10")
no_answer = cur.fetchall()
results.append(f"=== QA CHUNKS WITHOUT 'Answer:' field: {len(no_answer)} ===")
for cid, content, tc in no_answer:
    results.append(f"ID: {cid[:8]}... TOKENS: {tc}")
    results.append(f"CONTENT: {content[:400]}")
    results.append("")

# Check how many QA chunks have all fields vs partial
cur.execute("SELECT COUNT(*) FROM chunks WHERE chunk_type = 'qa_pair' AND content LIKE '%Question:%' AND content LIKE '%Answer:%' AND content LIKE '%Keyword:%'")
complete = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM chunks WHERE chunk_type = 'qa_pair' AND content LIKE '%Question:%' AND content LIKE '%Answer:%' AND content NOT LIKE '%Keyword:%'")
no_kw = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM chunks WHERE chunk_type = 'qa_pair' AND content LIKE '%Question:%' AND content NOT LIKE '%Answer:%'")
no_ans = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM chunks WHERE chunk_type = 'qa_pair' AND content NOT LIKE '%Question:%'")
no_q = cur.fetchone()[0]

results.append("=== FIELD COMPLETENESS ===")
results.append(f"Complete (Q+A+K): {complete}")
results.append(f"Missing Keyword: {no_kw}")
results.append(f"Missing Answer: {no_ans}")
results.append(f"Missing Question: {no_q}")

# Check what the short chunks look like
cur.execute("SELECT content, token_count, metadata FROM chunks WHERE chunk_type = 'qa_pair' AND token_count < 80 ORDER BY token_count")
very_short = cur.fetchall()
results.append("")
results.append(f"=== VERY SHORT QA CHUNKS (<80 tokens): {len(very_short)} ===")
for content, tc, meta in very_short[:5]:
    results.append(f"TOKENS: {tc}")
    results.append(f"CONTENT: {content[:500]}")
    results.append(f"METADATA: {meta}")
    results.append("")

conn.close()

with open(r"C:\Users\10225\Downloads\KB\kb-manager\chunk_analysis.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results))
print("done")
