import json
import sqlite3
from pathlib import Path

DB_PATH = Path("data/kb_test.db")
TEST_JSON = Path("data/test_questions.json")

def main():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        with open(TEST_JSON, "r") as f:
            data = json.load(f)
            
        remapped_count = 0
        
        for item in data:
            ans = item["expected_answer"].strip()
            
            # Find the new chunk ID based on exact matching of the answer
            cur.execute("SELECT id FROM chunks WHERE content LIKE ?", ('%' + ans + '%',))
            rows = cur.fetchall()
            
            if rows:
                new_id = rows[0]["id"]
                item["expected_chunk_ids"] = [new_id]
                item["relevance_scores"] = {new_id: 1.0}
                remapped_count += 1
            else:
                # Fallback: find by question
                q = item["gt"].strip()
                cur.execute("SELECT id FROM chunks WHERE content LIKE ?", ('%' + q + '%',))
                rows = cur.fetchall()
                if rows:
                    new_id = rows[0]["id"]
                    item["expected_chunk_ids"] = [new_id]
                    item["relevance_scores"] = {new_id: 1.0}
                    remapped_count += 1
                    
        with open(TEST_JSON, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"Remapped {remapped_count}/{len(data)} questions to new UUIDs.")

if __name__ == "__main__":
    main()
