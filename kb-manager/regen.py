import sqlite3, json, pathlib, sys
DB_PATH='data/kb_test.db'
import pathlib, json, random, re, sqlite3
from kb_manager.evaluation.query_formats import apply_format
p=pathlib.Path(DB_PATH)
print('DB exists', p.exists())
conn=sqlite3.connect(DB_PATH)
cur=conn.cursor()
cur.execute("SELECT count(*) FROM chunks WHERE chunk_type='qa_pair'")
print(cur.fetchone())
# now run generate with patched DB path
import generate_test_questions as gt
gt.DB_PATH='data/kb_test.db'
gt.main(num_bases=20, seed=42)
print('done')
