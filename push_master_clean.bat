@echo off
cd /d "%~dp0"
echo Staging...
git add -A >nul 2>&1
git diff --cached --quiet || git commit -m "chore: sync massive benchmark UI and iva debug" >nul
echo Fetching origin...
git fetch origin --prune >nul 2>&1
echo Switching to master...
git checkout master >nul 2>&1
echo Pulling --rebase...
git pull --rebase origin master >nul 2>&1
if errorlevel 1 (
  echo Rebase failed, trying merge...
  git rebase --abort >nul 2>&1
  git pull origin master --no-rebase >nul 2>&1
)
echo Pushing master...
git push origin master
if errorlevel 1 (
  echo Push failed
) else (
  echo Master pushed OK
)
echo Pushing stable...
git push origin stable/v7.2 >nul 2>&1
git push origin v7.2-stable >nul 2>&1
echo Done
git log --oneline -5
pause
