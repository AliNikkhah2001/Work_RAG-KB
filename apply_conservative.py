import json, pathlib

tsv = pathlib.Path(r'Work_Credit-RAG_Phase1/components/guardrails/kb/hurtlex_FA.tsv').read_text(encoding='utf-8')
lines = tsv.splitlines()

conservative = set()
for line in lines[1:]:  # skip header
    parts = line.split('\t')
    if len(parts) >= 6 and parts[5].lower() == 'conservative':
        lemma = parts[4]
        conservative.add(lemma)

out = sorted(conservative)
output_path = pathlib.Path(r'Work_Credit-RAG_Phase1/hurtlex_fa_conservative.json')
output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print('Saved', len(out), 'conservative lemmas to', output_path)
print('First 10:', out[:10])