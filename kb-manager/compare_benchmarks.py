"""Compare BEFORE vs AFTER benchmark results (IVA + full)."""
import json, pathlib

def load(p):
    p = pathlib.Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

iva_before = load("data/iva_results_BEFORE.json")
iva_after = load("data/iva_results.json")
bench_before = load("data/benchmark_results_BEFORE.json")
bench_after = load("data/benchmark_results.json")
ir_before = load("data/ir_metrics_BEFORE.json")
ir_after = load("data/ir_metrics.json")

print("="*70)
print("IVA 15 — BEFORE vs AFTER (fixed Persian preprocessor)")
print("="*70)
if iva_before and iva_after:
    def stats(rows):
        hits = sum(1 for r in rows if r["doc_hit"])
        ans = sum(1 for r in rows if r["ans_hit"])
        mrr = sum( (1.0/r["doc_rank"] if r["doc_rank"]>0 else 0) for r in rows)/len(rows)
        avg = sum(r["elapsed_ms"] for r in rows)/len(rows)
        return hits, ans, mrr, avg
    bh, ba, bm, bv = stats(iva_before)
    ah, aa, am, av = stats(iva_after)
    print(f"BEFORE: doc_hit {bh}/15={bh/15:.1%} ans_hit {ba}/15 mrr {bm:.3f} avg {bv:.0f}ms")
    print(f"AFTER : doc_hit {ah}/15={ah/15:.1%} ans_hit {aa}/15 mrr {am:.3f} avg {av:.0f}ms")
    print(f"Delta : doc_hit {ah-bh:+d} ans_hit {aa-ba:+d} mrr {am-bm:+.3f} latency {av-bv:+.0f}ms")
else:
    print("IVA files missing — run run_iva_eval.py first")

print()
print("="*70)
print("Full benchmark — BEFORE vs AFTER")
print("="*70)
if bench_before and bench_after:
    for key in ["overall","by_format"]:
        print(f"\n{key}:")
        b = bench_before.get(key, {}) if key=="overall" else bench_before.get("by_format",{})
        a = bench_after.get(key, {}) if key=="overall" else bench_after.get("by_format",{})
        if key=="overall":
            print(f"  BEFORE hit {b.get('hit_rate',0):.1%} mrr {b.get('mrr',0):.3f} lat {b.get('avg_latency_ms',0):.0f}")
            print(f"  AFTER  hit {a.get('hit_rate',0):.1%} mrr {a.get('mrr',0):.3f} lat {a.get('avg_latency_ms',0):.0f}")
        else:
            for fmt in sorted(set(b)|set(a)):
                bv = b.get(fmt,{}); av = a.get(fmt,{})
                print(f"  {fmt:15} BEFORE {bv.get('hit_rate',0):.1%} -> AFTER {av.get('hit_rate',0):.1%} (Δ {av.get('hit_rate',0)-bv.get('hit_rate',0):+.1%})")
else:
    print("benchmark_results missing")

# Save comparison json
out = {
    "iva_before": iva_before[:2] if iva_before else None,
    "iva_after": iva_after[:2] if iva_after else None,
    "bench_before_overall": bench_before.get("overall") if bench_before else None,
    "bench_after_overall": bench_after.get("overall") if bench_after else None,
}
pathlib.Path("data/benchmark_comparison_before_after.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nSaved data/benchmark_comparison_before_after.json")
