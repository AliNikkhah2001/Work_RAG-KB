"""Start the KB server as a detached process that survives the spawning shell."""
import subprocess
import sys

CMD = [sys.executable, "run_server.py"]
with open("server.log", "w", encoding="utf-8") as out, open("server_err.log", "w", encoding="utf-8") as err:
    p = subprocess.Popen(
        CMD,
        cwd=r"D:/Code/KB/kb-manager",
        stdout=out,
        stderr=err,
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        close_fds=True,
    )
print("detached pid", p.pid)