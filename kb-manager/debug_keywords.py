import sqlite3
import json

conn = sqlite3.connect('data/kb_test.db')
cur = conn.cursor()
cur.execute('SELECT id, keywords, metadata FROM chunks WHERE chunk_type=? LIMIT 10', ('qa_pair',))

with open('keywords_output.txt', 'w', encoding='utf-8') as f:
    for row in cur.fetchall():
        f.write(f'ID: {row[0][:8]}\n')
        f.write(f'Keywords (from chunks table): {row[1]}\n')
        meta = json.loads(row[2]) if row[2] else {}
        fields = meta.get('fields', {})
        f.write(f'keyword field: {fields.get("keyword", "NOT FOUND")}\n')
        f.write(f'keywords field: {fields.get("keywords", "NOT FOUND")}\n')
        f.write('---\n')

print("Done, check keywords_output.txt")