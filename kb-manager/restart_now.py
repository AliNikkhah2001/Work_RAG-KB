"""Restart KB server - run with: python restart_now.py"""
import subprocess, time, sys, os
from pathlib import Path

ROOT = Path(__file__).parent
os.chdir(ROOT)

print("Finding process on :8000...")
try:
    out = subprocess.check_output("netstat -ano | findstr :8000", shell=True, text=True, errors="ignore")
    print(out)
    for line in out.splitlines():
        parts = line.strip().split()
        if parts and parts[-1].isdigit():
            pid = parts[-1]
            print(f"Killing PID {pid}...")
            subprocess.run(f"taskkill /F /PID {pid}", shell=True)
except Exception as e:
    print(f"netstat failed: {e}")

time.sleep(2)
print("Starting server detached...")
# use the detached starter
subprocess.Popen([sys.executable, "start_server_detached.py"], cwd=str(ROOT),
                 creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
                 close_fds=True)
print("Started. Waiting 8s for pre-warm...")
time.sleep(8)
try:
    import urllib.request
    print("GET /transparency ->", urllib.request.urlopen("http://127.0.0.1:8000/transparency", timeout=5).status)
    print("GET / ->", urllib.request.urlopen("http://127.0.0.1:8000/", timeout=5).status)
except Exception as e:
    print(f"Check failed: {e} - see server.log / server_err.log")
    if Path("server_err.log").exists():
        print(Path("server_err.log").read_text(encoding="utf-8", errors="ignore")[-2000:])
