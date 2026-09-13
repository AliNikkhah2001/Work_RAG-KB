import json

with open('data/test_questions.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('_tq_debug.txt', 'w', encoding='utf-8') as out:
    out.write(f"Total questions: {len(data)}\n\n")
    for i, q in enumerate(data[:3]):
        out.write(f"--- Question {i} ---\n")
        out.write(json.dumps(q, ensure_ascii=False, indent=2) + "\n\n")
    # Check what keys exist
    if data:
        out.write(f"Keys in first item: {list(data[0].keys())}\n")
        # Check for expected_chunk_ids
        has_expected = sum(1 for q in data if q.get('expected_chunk_ids'))
        out.write(f"Questions with expected_chunk_ids: {has_expected}/{len(data)}\n")
        # Check empty/missing fields
        for field in ['question', 'expected_chunk_ids', 'format', 'difficulty']:
            missing = sum(1 for q in data if not q.get(field))
            out.write(f"Questions missing '{field}': {missing}/{len(data)}\n")
