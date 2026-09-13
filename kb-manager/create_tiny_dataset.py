import json

with open("data/test_questions_small.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Take first 5 queries
subset = data[:5]

with open("data/test_questions_tiny.json", "w", encoding="utf-8") as f:
    json.dump(subset, f, ensure_ascii=False, indent=2)

print(f"Created tiny dataset with {len(subset)} queries")