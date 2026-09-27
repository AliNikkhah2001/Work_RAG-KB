@echo off
REM Launcher for one mining shard (detached via `start`).
REM Usage: launch_mining_shard.bat SHARD NUM_SHARDS OUTFILE LOGFILE
setlocal
set "PYTHONHASHSEED=0"
set "PYTHONIOENCODING=utf-8"
set "KB_DB_URL=sqlite+aiosqlite:///D:/Code/KB/kb-manager/data/kb_9_7_2026.db"
set "KB_SOURCE_DIR=D:/Code/KB/kb-source/KB_9.7.2026"
set "OMP_NUM_THREADS=6"
set "MKL_NUM_THREADS=6"
cd /d D:\Code\KB\kb-manager
python scripts\retrieval_training\run_mining.py --seed 42 --shard %1 --num-shards %2 --output %3 > %4 2>&1
