@echo off
REM Fix master push rejected - fetch and rebase, then push
cd /d "%~dp0"
echo Fetching origin...
git fetch origin
echo.
echo Current master log:
git log --oneline -10
echo.
echo Rebasing master onto origin/master...
git checkout master
git pull --rebase origin master
if errorlevel 1 (
  echo Rebase failed - manual resolve needed
  pause
  exit /b 1
)
echo.
echo Pushing master...
git push origin master
if errorlevel 1 (
  echo Push still failed
  pause
  exit /b 1
)
echo.
echo Pushing stable tag/branch...
git push origin v7.2-stable 2>nul
git push -u origin stable/v7.2 2>nul
echo.
echo Done - master pushed to origin
git log --oneline -10
pause
