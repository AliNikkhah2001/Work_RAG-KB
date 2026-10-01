"""Live benchmark dashboard - run in a separate terminal window."""
import json, os, sys, time, pathlib

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

JSONL = pathlib.Path("artifacts/retrieval_training/retrieval_failures_v10_1405-06-23.jsonl")
PROGRESS = pathlib.Path(os.environ.get("TEMP", ".")) / "agent2_v10_progress.json"
TOTAL = 1659

def load_recs():
    recs = []
    if not JSONL.exists():
        return recs
    for line in JSONL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                recs.append(json.loads(line))
            except Exception:
                continue
    return recs

def main():
    while True:
        recs = load_recs()
        n = len(recs)
        labels = {}
        hits = hit1 = 0
        mrr_s = 0.0
        file_q = {}
        for r in recs:
            ft = r.get("failure_type", "?")
            labels[ft] = labels.get(ft, 0) + 1
            if r.get("hit5"): hits += 1
            fr = r.get("final_rank")
            if fr and isinstance(fr, int) and fr > 0:
                mrr_s += 1.0 / fr
                if fr == 1: hit1 += 1
            qid = r.get("query_id", "?")
            fn = qid.rsplit("#", 1)[0] if "#" in qid else qid
            file_q.setdefault(fn, []).append(r)

        ok = [r for r in recs if r.get("failure_type") != "error"]
        n_ok = len(ok)
        bm25_h5 = sum(1 for r in ok if r.get("bm25_rank") is not None and r["bm25_rank"] <= 5)
        dense_h5 = sum(1 for r in ok if r.get("dense_rank") is not None and r["dense_rank"] <= 5)
        merged_h5 = sum(1 for r in ok if r.get("merged_rank") is not None and r["merged_rank"] <= 5)
        bm25_r = sum(1 for r in ok if r.get("bm25_rank") is not None)
        dense_r = sum(1 for r in ok if r.get("dense_rank") is not None)
        merged_r = sum(1 for r in ok if r.get("merged_rank") is not None)

        h5r = hits / max(n_ok, 1)
        mrr = mrr_s / max(n_ok, 1)

        pf = {}
        for fn, fr in file_q.items():
            a = sum(1 for r in fr if r.get("failure_type") == "A")
            ap = sum(1 for r in fr if r.get("failure_type") == "A'")
            b = sum(1 for r in fr if r.get("failure_type") == "B")
            d = sum(1 for r in fr if r.get("hit5"))
            pf[fn] = (len(fr), a, ap, b, d)

        os.system("cls" if os.name == "nt" else "clear")
        pct = n / TOTAL
        filled = int(40 * pct)
        pb = "X" * filled + "-" * (40 - filled)

        L = []
        L.append("=" * 72)
        L.append("  v10 BENCHMARK LIVE DASHBOARD (post-fix: overlap + keyword + prefix removal)")
        L.append("=" * 72)
        L.append("")
        L.append("  Progress:  [%s] %d/%d (%.1f%%)" % (pb, n, TOTAL, pct * 100))
        L.append("  Evaluated: %d   Errors: %d" % (n_ok, labels.get("error", 0)))
        L.append("")
        L.append("  Hit@1:  %d/%d (%.4f)" % (hit1, n_ok, hit1 / max(n_ok, 1)))
        L.append("  Hit@5:  %d/%d (%.4f)" % (hits, n_ok, h5r))
        L.append("  MRR:    %.4f" % mrr)
        L.append("")
        L.append("  STAGE FUNNEL")
        L.append("  %-10s %12s %12s" % ("Stage", "Hit@5", "recall@100"))
        L.append("  %-10s %12s %12s" % ("-" * 10, "-" * 12, "-" * 12))
        L.append("  %-10s %5d/%-5d (%.4f)  %5d/%-5d (%.4f)" % ("BM25", bm25_h5, n_ok, bm25_h5/max(n_ok,1), bm25_r, n_ok, bm25_r/max(n_ok,1)))
        L.append("  %-10s %5d/%-5d (%.4f)  %5d/%-5d (%.4f)" % ("Dense", dense_h5, n_ok, dense_h5/max(n_ok,1), dense_r, n_ok, dense_r/max(n_ok,1)))
        L.append("  %-10s %5d/%-5d (%.4f)  %5d/%-5d (%.4f)" % ("Merged", merged_h5, n_ok, merged_h5/max(n_ok,1), merged_r, n_ok, merged_r/max(n_ok,1)))
        L.append("  %-10s %5d/%-5d (%.4f)" % ("Final", hits, n_ok, h5r))
        L.append("")
        L.append("  FAILURE FUNNEL")
        L.append("  A  (retriever miss):   %4d  neither leg found gold in top-100" % labels.get("A", 0))
        L.append("  A' (fusion loss):      %4d  one leg found, RRF dropped it" % labels.get("A'", 0))
        L.append("  B  (reranker loss):    %4d  gold in merged, reranker dropped from top-5" % labels.get("B", 0))
        L.append("  D  (success):          %4d  gold in final top-5" % labels.get("D", 0))
        L.append("")
        L.append("  PER-FILE BREAKDOWN")
        L.append("  %-33s %4s %3s %3s %3s %4s %6s" % ("File", "n", "A", "Ap", "B", "D", "hit5"))
        L.append("  " + "-" * 33 + " " + "-" * 4 + " " + "-" * 3 + " " + "-" * 3 + " " + "-" * 3 + " " + "-" * 4 + " " + "-" * 6)
        for fn in sorted(pf):
            f_n, f_a, f_ap, f_b, f_d = pf[fn]
            fh = f_d / max(f_n, 1)
            short = fn[:31] if len(fn) > 31 else fn
            L.append("  %-33s %4d %3d %3d %3d %4d %6.3f" % (short, f_n, f_a, f_ap, f_b, f_d, fh))
        L.append("")
        L.append("  Updated: %s | Refreshes every 5s | Ctrl+C to quit" % time.strftime("%H:%M:%S"))

        sys.stdout.write("\n".join(L) + "\n")
        sys.stdout.flush()
        time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
