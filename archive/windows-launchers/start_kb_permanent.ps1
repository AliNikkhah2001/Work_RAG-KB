$process = Start-Process python -ArgumentList "run_server.py" -NoNewWindow -PassThru -WorkingDirectory "D:\Code\KB\kb-manager"
Start-Sleep -Seconds 3
Write-Host "KB Manager PID: $($process.Id)"