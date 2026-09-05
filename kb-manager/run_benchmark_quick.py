import asyncio, json, pathlib
from kb_manager.evaluation.benchmark import AsyncBenchmarkRunner
from kb_manager.web.routes.search import search_knowledge_base
import time

async def _search(q,k):
    steps=await search_knowledge_base(q,k)
    return [(r.chunk_id, r.hybrid_score) for r in steps.final_results]

# use small dataset
dataset=json.loads(pathlib.Path('data/test_questions.json').read_text(encoding='utf-8'))
# take 20 samples
dataset=dataset[:20]
print('dataset', len(dataset))
runner=AsyncBenchmarkRunner(_search, top_k=5, version='live-kb')
result=asyncio.run(runner._run_async(dataset, progress=lambda d,t: print(f"progress {d}/{t}")))
print('overall', result.overall)
print('by_format', result.by_format)
result_dict=result.to_dict()
pathlib.Path('data/benchmark_results.json').write_text(json.dumps(result_dict, ensure_ascii=False, indent=2), encoding='utf-8')
print('written')
# also regenerate plots
from kb_manager.evaluation.plots import render_benchmark_plots, plot_duplicate_stats
from kb_manager.versioning.snapshot import export_compact
from kb_manager.config import PROJECT_ROOT
import json as js
export=export_compact(str(PROJECT_ROOT / "data" / "kb_test.db"))
open('data/qa_duplication.json','w',encoding='utf-8').write(json.dumps(export['counts'], ensure_ascii=False, indent=2))
render_benchmark_plots('data/benchmark_results.json','data/plots')
plot_duplicate_stats('data/qa_duplication.json','data/plots/qa_duplication.png')
print('plots done')
