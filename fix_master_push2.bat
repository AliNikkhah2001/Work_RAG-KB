@echo off
cd /d "%~dp0"
echo Stashing unstaged (iva_debug.html, kb-source)...
git add -A
git status --short
git diff --cached --quiet || git commit -m "chore: update iva_debug and massive test fix"
echo Fetching...
git fetch origin
echo Rebasing master...
git checkout master
git pull --rebase origin master
if errorlevel 1 (
  echo Rebase failed, trying merge...
  git pull origin master --no-rebase
)
git push origin master
if errorlevel 1 echo push failed
else echo master pushed OK
git push origin stable/v7.2 2>nul
git push origin v7.2-stable 2>nul
git log --oneline -5
pause
