# Client Demo Webapp Launch Script
# Starts FastAPI backend (port 8000) and Next.js frontend (port 3050)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   AIMDEA - Client Demo Presentation Launcher             " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "[ERROR] Virtualenv python not found at $venvPython" -ForegroundColor Red
    exit 1
}

# 1. Start FastAPI Backend
Write-Host "`n[1/3] Starting FastAPI backend on http://localhost:8000..." -ForegroundColor Yellow
$backendProcess = Start-Process -FilePath $venvPython -ArgumentList "-m", "uvicorn", "src.server:app", "--port", "8000", "--host", "0.0.0.0" -PassThru -NoNewWindow

# 2. Start Next.js Frontend
Write-Host "[2/3] Starting Next.js frontend on http://localhost:3050..." -ForegroundColor Yellow
$frontendDir = Join-Path $PSScriptRoot "frontend"
$frontendProcess = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "pnpm", "dev" -WorkingDirectory $frontendDir -PassThru -NoNewWindow

# 3. Wait for services to be ready
Write-Host "`n[3/3] Waiting for servers to initialize..." -ForegroundColor Yellow
$backendReady = $false
$maxTries = 90
for ($i = 0; $i -lt $maxTries; $i++) {
    Start-Sleep -Seconds 1
    try {
        $res = Invoke-RestMethod -Uri "http://localhost:8000/api/health" -Method Get -TimeoutSec 2 -ErrorAction Stop
        if ($res.status -eq "ok" -and $res.rag_ready -eq $true) {
            $backendReady = $true
            break
        }
    } catch {
        # waiting for server boot
    }
}

if ($backendReady) {
    Write-Host "  Backend & DINOv2 Pipeline is ONLINE at http://localhost:8000" -ForegroundColor Green
} else {
    Write-Host "  [WARNING] Backend startup is still finalizing in the background." -ForegroundColor DarkYellow
}

# Wait for frontend port 3050 to accept connections
$frontendReady = $false
for ($i = 0; $i -lt 15; $i++) {
    try {
        $res = Invoke-WebRequest -Uri "http://localhost:3050" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($res.StatusCode -eq 200) {
            $frontendReady = $true
            break
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}

Write-Host "`nDemo services running! Press Ctrl+C in this terminal to shut down both servers." -ForegroundColor Green
Write-Host "Opening http://localhost:3050 in your browser...`n" -ForegroundColor Cyan
Start-Process "http://localhost:3050"

try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
} finally {
    Write-Host "`nShutting down demo services..." -ForegroundColor Yellow
    if ($backendProcess -and -not $backendProcess.HasExited) {
        Stop-Process -Id $backendProcess.Id -Force -ErrorAction SilentlyContinue
    }
    if ($frontendProcess -and -not $frontendProcess.HasExited) {
        Stop-Process -Id $frontendProcess.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Clean shutdown complete." -ForegroundColor Green
}
