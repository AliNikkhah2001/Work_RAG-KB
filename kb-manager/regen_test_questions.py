"""Regenerate test_questions.json from the current KB."""
import json
import random
import sqlite3
from pathlib import Path

from kb_manager.evaluation.query_formats import (
    apply_format,
    default_difficulty,
    format_list,
)

DB_PATH = Path(__file__).resolve().parent / "data" / "kb_test.db"
OUTPUT = Path(__file__).resolve().parent / "data" / "test_questions.json"
FORMATS = format_list()
QUERIES_PER_FORMAT = 20  # 20 questions x 6 formats = 120 total
SEED = 42


def main():
    random.seed(SEED)
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    # Get QA chunks with question + keyword fields
    cur.execute(
        "SELECT id, content, metadata FROM chunks "
        "WHERE chunk_type = 'qa_pair' "
        "ORDER BY RANDOM()"
    )
    rows = cur.fetchall()
    conn.close()

    # Extract questions from QA chunks
    base_questions = []
    for chunk_id, content, meta_json in rows:
        meta = json.loads(meta_json) if meta_json else {}
        fields = meta.get("fields", {})
        question = fields.get("question", "").strip()
        keywords = fields.get("keyword", "")
        answer = fields.get("answer", fields.get("briefanswer", ""))
        if not question or len(question) < 5:
            continue
        base_questions.append({
            "chunk_id": chunk_id,
            "question": question,
            "keywords": keywords,
            "answer": answer,
        })

    print(f"Found {len(base_questions)} QA chunks with valid questions")

    # Take up to QUERIES_PER_FORMAT unique base questions
    selected = base_questions[:QUERIES_PER_FORMAT]
    if len(selected) < QUERIES_PER_FORMAT:
        print(f"Warning: only {len(selected)} questions available, need {QUERIES_PER_FORMAT}")

    dataset = []
    for fmt in FORMATS:
        for item in selected:
            query_text = apply_format(fmt, item["question"], item["keywords"])
            dataset.append({
                "query": query_text,
                "expected_chunk_ids": [item["chunk_id"]],
                "expected_answer": item["answer"],
                "relevance_scores": {item["chunk_id"]: 1.0},
                "category": "factual",
                "difficulty": default_difficulty(fmt),
                "format": fmt,
                "gt": item["question"],
                "keywords": item["keywords"],
                "gt_similarity": 1.0 if fmt == "verbatim" else round(random.uniform(0.3, 0.8), 3),
            })

    random.shuffle(dataset)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(dataset)} queries ({len(selected)} questions x {len(FORMATS)} formats)")
    print(f"Saved to {OUTPUT}")

    # Summary by format
    for fmt in FORMATS:
        count = sum(1 for d in dataset if d["format"] == fmt)
        print(f"  {fmt}: {count}")


if __name__ == "__main__":
    main()
