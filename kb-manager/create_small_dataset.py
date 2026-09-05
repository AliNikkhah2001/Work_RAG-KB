import json

with open("data/test_questions.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Take first 20 queries (covers all 6 formats for ~3-4 base questions)
subset = data[:20]

with open("data/test_questions_small.json", "w", encoding="utf-8") as f:
    json.dump(subset, f, ensure_ascii=False, indent=2)

print(f"Created small dataset with {len(subset)} queries")
for i, q in enumerate(subset):
    print(f"  {i+1}. [{q['format']}] {q['query'][:60]}...")