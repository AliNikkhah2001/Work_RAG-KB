import requests

endpoints = [
    "/",
    "/documents",
    "/chunks",
    "/pipeline",
    "/versions",
    "/monitoring",
    "/cleanup/qa",
    "/search",
    "/transparency",
    "/benchmarks",
    "/benchmarks/massive",
    "/benchmarks/comparison",
    "/ingestion"
]

base_url = "http://127.0.0.1:8000"
results = {}

for ep in endpoints:
    try:
        resp = requests.get(base_url + ep, timeout=5)
        results[ep] = resp.status_code
        if resp.status_code != 200:
            print(f"FAILED: {ep} - Status {resp.status_code}")
    except Exception as e:
        results[ep] = str(e)
        print(f"ERROR: {ep} - {e}")

print("--- ENDPOINT TEST RESULTS ---")
for ep, status in results.items():
    print(f"{ep:<25} {status}")
