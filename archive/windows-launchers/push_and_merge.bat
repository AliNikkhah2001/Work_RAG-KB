@echo off
REM Push and merge KB changes via CMD (bypasses blocked PowerShell 0x800704ec)
REM Run by double-click or: cmd /c push_and_merge.bat

cd /d "%~dp0\.."
echo === Repo: %CD% ===
echo Branch: master  Remote: origin/master
echo.

echo --- git status ---
git status --short
echo.

echo --- git diff --stat (staged+unstaged) ---
git diff --stat
echo.

echo --- staging ---
git add -A
git status --short
echo.

echo --- committing (if needed) ---
git diff --cached --quiet || git commit -m "feat: Transparency tab + Persian RTL fix + route ordering + QA experiment docs + troubleshooting 0x800704ec"
echo.

echo --- pull --rebase (fast-forward) ---
git fetch origin
git pull --rebase origin master || (echo REBASE FAILED - resolve manually & pause & exit /b 1)
echo.

echo --- push master ---
git push origin master
if errorlevel 1 (echo PUSH FAILED & pause & exit /b 1)
echo.

echo --- tags ---
git push origin --tags 2>nul
echo.

echo --- kb-source submodule (if changed) ---
cd /d "%~dp0\..\kb-source" 2>nul && (
  git status --short | findstr . >nul && (
    echo kb-source has local changes - pushing submodule...
    git add -A
    git diff --cached --quiet || git commit -m "chore: update 1405-05-31 transparency assets"
    git push origin main
    cd /d "%~dp0\.."
    git add kb-source
    git diff --cached --quiet || git commit -m "chore: bump kb-source gitlink"
    git push origin master
  ) || echo kb-source clean
) || echo no kb-source submodule
cd /d "%~dp0\.."

echo.
echo --- log (last 5) ---
git log --oneline -5
echo.
echo --- DONE: check https://github.com/AliNikkhah2001/Work_RAG-KB ---
pause
