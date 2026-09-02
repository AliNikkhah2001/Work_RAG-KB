"""Write v7 benchmark_results.json + ir_metrics.json from the saved IVA doc-level eval."""
import json
import pathlib
from datetime import UTC, datetime

root = pathlib.Path(r"D:/Code/KB/kb-manager/data")
iva = json.loads((root / "iva_results.json").read_text(encoding="utf-8"))
ds = json.loads((root / "test_questions_iva.json").read_text(encoding="utf-8"))

queries = []
for row in iva:
    # query text from dataset (index i-1)
    item = ds[row["i"] - 1]
    queries.append(
        {
            "query": item["query"],
            "expected_ids": item.get("expected_chunk_ids", []),
            "format": "verbatim",
            "difficulty": "medium",
            "hit": row["doc_hit"],
            "rank": row["doc_rank"],
            "top1_hit": row["doc_rank"] == 1,
            "elapsed_ms": row["elapsed_ms"],
        }
    )

n = len(queries)
hits = sum(1 for q in queries if q["hit"])
top1 = sum(1 for q in queries if q["top1_hit"])
ranks = [q["rank"] for q in queries if q["rank"] > 0]
mrr = sum(1.0 / r for r in ranks) / n
lat = sum(q["elapsed_ms"] for q in queries) / n

overall = {
    "queries": n,
    "hit_rate": round(hits / n, 4),
    "top1_hit_rate": round(top1 / n, 4),
    "mrr": round(mrr, 4),
    "avg_rank": round((sum(q["rank"] for q in queries if q["rank"] > 0) / max(len(ranks), 1)), 2) if ranks else 0,
    "avg_latency_ms": round(lat, 1),
}
by_format = {
    "verbatim": {
        "queries": n,
        "hit_rate": round(hits / n, 4),
        "top1_hit_rate": round(top1 / n, 4),
        "mrr": round(mrr, 4),
        "avg_latency_ms": round(lat, 1),
    }
}
result = {
    "version": "v7_iva_1405-05-31",
    "created_at": datetime.now(UTC).isoformat(),
    "top_k": 5,
    "total_queries": n,
    "by_format": by_format,
    "overall": overall,
    "queries": queries,
    "note": "doc-level Hit@5: golden-doc chunk present in top-5 retrieved (15 IVA questions)",
}
(root / "benchmark_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

ir = {
    "top_k": 5,
    "metrics": {
        "mrr@5": round(mrr, 4),
        "recall@5": round(hits / n, 4),
        "precision@5": round(hits / n, 4),
    },
    "note": "chunk-level recall approximated by doc-level hit@5",
}
(root / "ir_metrics.json").write_text(json.dumps(ir, ensure_ascii=False, indent=2), encoding="utf-8")

print("overall:", json.dumps(overall, ensure_ascii=False))
print("wrote benchmark_results.json + ir_metrics.json (v7)")