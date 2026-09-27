import json
import urllib.request

def get(url, timeout=20):
    return urllib.request.urlopen(url, timeout=timeout).read()

# 1. Pipeline / stale job marked interrupted
r = get("http://127.0.0.1:8000/pipeline")
print("PIPELINE page bytes", len(r))
import re
html = r.decode(errors="ignore")
for m in re.findall(r"(running|interrupted|completed|failed)", html)[:6]:
    print("  job status token:", m)

# 2. Versions page
r = get("http://127.0.0.1:8000/versions")
print("VERSIONS page bytes", len(r))

# 3. Snapshot plot images now serve (was 404)
for url in [
    "http://127.0.0.1:8000/benchmarks/snapshots/v7_iva_1405-05-31/file/plots/hit_rate_by_format.png",
    "http://127.0.0.1:8000/benchmarks/snapshots/v7_iva_1405-05-31/file/benchmarks/benchmark_results.json",
    "http://127.0.0.1:8000/benchmarks/plot/hit_rate_by_format.png",
    "http://127.0.0.1:8000/benchmarks/snapshots/v6/file/plots/hit_rate_by_format.png",
]:
    try:
        resp = urllib.request.urlopen(url, timeout=15)
        print(f"{'OK ' if resp.status==200 else resp.status} {resp.headers.get('Content-Type')}  {resp.read(20)[:10]}  <- {url.split('/benchmarks/')[1][:60]}")
    except Exception as e:
        print(f"ERR {e}  <- {url.split('/benchmarks/')[1][:60]}")

# 4. Comparison page + data
r = get("http://127.0.0.1:8000/benchmarks/comparison")
print("COMPARISON page bytes", len(r))
html = r.decode(errors="ignore")
# check v7 default button present
print("  comparison has v7 button:", "v7" in html and 'data-version' not in html)
print("  comparison embeds comparison_data:", "comparisonData" in html)
d = json.loads(get("http://127.0.0.1:8000/benchmarks/comparison/data"))
print("  comparison/data versions:", list(d.get("versions", {}).keys()))