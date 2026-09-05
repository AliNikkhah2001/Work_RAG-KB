"""Start the KB server as a detached process (cross-platform, config-driven)."""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CMD = [sys.executable, "run_server.py"]

# Windows vs POSIX detached
kwargs = {}
if sys.platform == "win32":
    kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
else:
    kwargs["start_new_session"] = True

with open(HERE / "server.log", "w", encoding="utf-8") as out, open(HERE / "server_err.log", "w", encoding="utf-8") as err:
    p = subprocess.Popen(CMD, cwd=str(HERE), stdout=out, stderr=err, close_fds=True, **kwargs)
print("detached pid", p.pid)