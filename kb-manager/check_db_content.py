import sqlite3

conn = sqlite3.connect(r"C:\Users\10225\Downloads\KB\kb-manager\data\kb_test.db")
cur = conn.cursor()

cur.execute("SELECT content FROM chunks WHERE chunk_type = 'qa_pair' LIMIT 1")
row = cur.fetchone()

if row:
    content = row[0]
    # Check raw bytes
    b = content[:200].encode('utf-8')
    with open("check_bytes.txt", "wb") as f:
        f.write(b)
    
    # Check if valid UTF-8
    try:
        content[:200].encode('utf-8').decode('utf-8')
        with open("check_result.txt", "w", encoding="utf-8") as f:
            f.write("Valid UTF-8\n")
            f.write(content[:200] + "\n")
    except Exception as e:
        with open("check_result.txt", "w", encoding="utf-8") as f:
            f.write(f"Invalid UTF-8: {e}\n")
    
    # Check actual bytes
    with open("check_raw.txt", "w", encoding="utf-8") as f:
        f.write(f"Length: {len(content)}\n")
        f.write(f"First 200 chars: {repr(content[:200])}\n")

conn.close()