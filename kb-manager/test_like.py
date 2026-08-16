import sqlite3

conn = sqlite3.connect(r"C:\Users\10225\Downloads\KB\kb-manager\data\kb_test.db")
cur = conn.cursor()

cur.execute("SELECT id FROM chunks WHERE content LIKE '%استانداردهای%' LIMIT 5")
rows = cur.fetchall()
with open("debug_out.txt", "w", encoding="utf-8") as f:
    f.write(f"Found {len(rows)} matches for 'استانداردهای'\n")

cur.execute("SELECT id FROM chunks WHERE content LIKE '%کدگذاری%' LIMIT 5")
rows = cur.fetchall()
with open("debug_out.txt", "a", encoding="utf-8") as f:
    f.write(f"Found {len(rows)} matches for 'کدگذاری'\n")

cur.execute("SELECT id FROM chunks WHERE content LIKE '%نوع%' LIMIT 5")
rows = cur.fetchall()
with open("debug_out.txt", "a", encoding="utf-8") as f:
    f.write(f"Found {len(rows)} matches for 'نوع'\n")

cur.execute("SELECT id FROM chunks WHERE content LIKE '%شرکت%' LIMIT 5")
rows = cur.fetchall()
with open("debug_out.txt", "a", encoding="utf-8") as f:
    f.write(f"Found {len(rows)} matches for 'شرکت'\n")

conn.close()