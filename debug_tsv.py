import pathlib

tsv = pathlib.Path(r'Work_Credit-RAG_Phase1/components/guardrails/kb/hurtlex_FA.tsv').read_text(encoding='utf-8')
lines = tsv.splitlines()

# Print first 3 lines raw - write to file to avoid encoding issues
with open('debug_out.txt', 'w', encoding='utf-8') as f:
    for i in range(min(3, len(lines))):
        f.write(f"Line {i}: {lines[i]}\n")
    if len(lines) > 1:
        parts = lines[1].split('\t')
        f.write(f"\nLine 1 parts count: {len(parts)}\n")
        for idx, p in enumerate(parts):
            f.write(f"  part[{idx}]: {p}\n")