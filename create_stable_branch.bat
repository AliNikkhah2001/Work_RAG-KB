@echo off
REM Create stable branch from current master (fixed persian.py) and publish to GitHub
cd /d "%~dp0"
echo Current branch:
git branch --show-current
echo.
echo Creating / checking out stable branch...
git checkout -b stable/v7.1-zwnj-fix 2>nul || git checkout stable/v7.1-zwnj-fix
if errorlevel 1 (echo checkout failed & pause & exit /b 1)

echo Adding all changes...
git add -A
git status --short

echo Committing if needed...
git diff --cached --quiet || git commit -m "stable: v7.1 fix ZWNJ mid-word corruption (persian.py), transparency zip browser, 2174 chunks clean"

echo Pushing branch to origin...
git push -u origin stable/v7.1-zwnj-fix
if errorlevel 1 (echo push branch failed & pause & exit /b 1)

echo Tagging stable version...
git tag -a v7.1-stable -m "stable v7.1: ZWNJ fix, transparency zip, 2174 chunks, 73.3%% IVA" 2>nul
git push origin v7.1-stable 2>nul
if errorlevel 1 echo tag exists or push failed

echo Also updating master if not already...
git checkout master
git merge --ff-only stable/v7.1-zwnj-fix 2>nul
git push origin master 2>nul

echo.
echo === DONE ===
echo Branch: stable/v7.1-zwnj-fix pushed
echo Tag: v7.1-stable
echo Check: https://github.com/AliNikkhah2001/Work_RAG-KB/tree/stable/v7.1-zwnj-fix
git log --oneline -5
pause
