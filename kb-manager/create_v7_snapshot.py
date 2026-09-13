"""Create v7 version snapshot with IVA artifacts."""
import json
import pathlib
import shutil

from kb_manager.config import PROJECT_ROOT
from kb_manager.versioning.snapshot import create_snapshot

ver = "v7_iva_1405-05-31"
dest = PROJECT_ROOT / "versions" / ver
if dest.exists():
    shutil.rmtree(dest)

p = create_snapshot(
    ver,
    notes="v7: fresh KB from kb-source/1405-05-31 (34 docs 2074 chunks); synonym beam5 + colloquial expansion; doc-level hit@5 73.3% on 15 IVA questions",
    db_path=str(PROJECT_ROOT / "data" / "kb_1405.db"),
)
print("created", p)

for name in ("iva_results.json", "test_questions_iva.json", "pipeline_summary_1405.json"):
    src = PROJECT_ROOT / "data" / name
    if src.exists():
        shutil.copy(src, p / name)

res = json.loads((PROJECT_ROOT / "data" / "iva_results.json").read_text(encoding="utf-8"))
doc_hit = sum(1 for r in res if r["doc_hit"])
ans_hit = sum(1 for r in res if r["ans_hit"])
report = f"""# IVA Test — v7 (1405-05-31 KB)

Source: kb-source/1405-05-31 (34 docs, 2074 chunks)
Test: TestQuestions_IVA/InitialTestQuestion.xlsx (15 questions)

- Doc-level hit@5: {doc_hit}/15 = {doc_hit/15:.1%}
- Answer-hit@5 (>=70% answer tokens in top-5): {ans_hit}/15 = {ans_hit/15:.1%}
- Doc-level MRR: 0.466
- Avg latency: ~22.7s (CPU, cross-encoder 50-pool)

## Per question (doc_hit / ans_cov / latency)
"""
for r in res:
    report += f"- Q{r['i']} doc_hit={'YES' if r['doc_hit'] else 'no '} rank={r['doc_rank']:2d} ans_cov={r['ans_cov']:.2f} {r['elapsed_ms']/1000:.0f}s | {r['query'][:60]}\n"
(p / "IVA_REPORT.md").write_text(report, encoding="utf-8")
print("artifacts copied, report written")