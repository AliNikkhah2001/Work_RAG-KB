import sqlite3
import json

conn = sqlite3.connect(r"C:\Users\10225\Downloads\KB\kb-manager\data\kb_test.db")
cur = conn.cursor()

cur.execute(
    "SELECT id, content, metadata FROM chunks "
    "WHERE chunk_type = 'qa_pair' "
    "ORDER BY RANDOM() LIMIT 5",
)
rows = cur.fetchall()
conn.close()

with open("debug_gen_out.txt", "w", encoding="utf-8") as f:
    for chunk_id, content, meta_json in rows:
        f.write(f"ID: {chunk_id}\n")
        f.write(f"Content preview: {content[:200]}\n")
        f.write(f"Metadata JSON type: {type(meta_json)}\n")
        f.write(f"Metadata JSON preview: {str(meta_json)[:200]}\n")
        
        if meta_json:
            try:
                meta = json.loads(meta_json)
                fields = meta.get("fields", {})
                question = fields.get("question", "")
                f.write(f"Question: {question}\n")
            except Exception as e:
                f.write(f"JSON parse error: {e}\n")
        f.write("\n---\n\n")

print("Done")