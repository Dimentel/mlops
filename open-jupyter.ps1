# open-jupyter.ps1
Write-Host "=== Jupyter Lab Auto-Opener ===" -ForegroundColor Green

# open-jupyter-simple.ps1
Write-Host "Jupyter Lab Opener" -ForegroundColor Green

# Simple Docker check - just try to run a command
docker version 2>&1 > $null
if ($LASTEXITCODE -eq 0) {
    Write-Host "Docker: OK" -ForegroundColor Green
} else {
    Write-Host "Please start Docker Desktop first!" -ForegroundColor Red
    exit 1
}

# Try to get token
Write-Host "Getting access token..." -ForegroundColor Yellow
$token = $null

# Method 1: access file
# Get the file content
$accessContent = docker-compose exec jupyter cat /home/jovyan/ACCESS.md 2>$null
if ($accessContent) {
    # Find the token line
    $tokenLine = $accessContent -split "`n" | Where-Object { $_ -match "Token\*\*:" }
    if ($tokenLine) {
        # Просто делим строку по двоеточию и берём вторую часть
        $parts = $tokenLine.Split(":", 2)  # Делим на 2 части максимум
        if ($parts.Length -eq 2) {
            $token = $parts[1].Trim()
            Write-Host "Token found in access file: $token" -ForegroundColor Green
        }
    }
}

# Method 2: From logs
if (-not $token) {
    $tokenLine = docker-compose logs jupyter 2>$null |
        Where-Object { $_ -match "token=" } |
        Select-Object -First 1
    if ($tokenLine -match "token=([a-f0-9]+)") {
        $token = $matches[1].Trim()
        Write-Host "Token found in logs: $token" -ForegroundColor Green
    }
}

# Construct URL
if ($token) {
    $url = "http://localhost:8855/lab?token={0}" -f $token
    Write-Host "Token found, url constructed" -ForegroundColor Green
    Write-Host $url
} else {
    $url = "http://localhost:8855/lab"
    Write-Host "No token found, opening without authentication" -ForegroundColor Yellow
}

Write-Host "Opening: $url" -ForegroundColor Cyan
Start-Process $url

Write-Host "If page doesn't load, check: docker-compose ps" -ForegroundColor Yellow

Write-Host "`nUseful commands:" -ForegroundColor Magenta
Write-Host "  Check logs: docker-compose logs jupyter" -ForegroundColor Gray
Write-Host "  Check status: docker-compose ps" -ForegroundColor Gray
Write-Host "  Restart: docker-compose restart jupyter" -ForegroundColor Gray