<#
.SYNOPSIS
    ASD-Facial-CNN: Full-Stack Multi-Service Local Development Orchestrator.
.DESCRIPTION
    Launches and manages the complete stack locally on Windows:
      1. PostgreSQL Database on port 5432 (with automatic H2 in-memory fallback)
      2. Python FastAPI PyTorch CNN Vision Engine on port 8000
      3. Java Spring Boot 3 Gateway on port 8080
      4. React 18 + Vite Clinician Dashboard on port 5173
    Streams color-coded logs concurrently from all application tiers.
#>

[CmdletBinding()]
param(
    [switch]$ForceH2 = $false
)

$ErrorActionPreference = "Continue"

Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host "      ASD-Facial-CNN: Pediatric Autism Diagnostic Platform                " -ForegroundColor Cyan
Write-Host "               Local Multi-Service Orchestrator                           " -ForegroundColor DarkCyan
Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host "Target Directory: $PSScriptRoot" -ForegroundColor DarkGray
Write-Host ""

# ------------------------------------------------------------------------------
# 1. Database Check (Port 5432)
# ------------------------------------------------------------------------------
Write-Host "[Database] Checking PostgreSQL availability on port 5432..." -ForegroundColor Yellow

$pgActive = $false

function Test-PostgresPort {
    try {
        $tcpClient = New-Object System.Net.Sockets.TcpClient
        $asyncResult = $tcpClient.BeginConnect("127.0.0.1", 5432, $null, $null)
        $success = $asyncResult.AsyncWaitHandle.WaitOne(1000, $false)
        if ($success -and $tcpClient.Connected) {
            $tcpClient.EndConnect($asyncResult)
            $tcpClient.Close()
            return $true
        }
        $tcpClient.Close()
        return $false
    } catch {
        return $false
    }
}

if (-not $ForceH2) {
    if (Test-PostgresPort) {
        $pgActive = $true
        Write-Host "[Database] PostgreSQL is actively running locally on port 5432." -ForegroundColor Green
    } else {
        Write-Host "[Database] PostgreSQL not detected on port 5432. Defaulting to H2 in-memory database." -ForegroundColor Yellow
    }
} else {
    Write-Host "[Database] -ForceH2 flag specified. Using H2 in-memory profile." -ForegroundColor Yellow
}

# ------------------------------------------------------------------------------
# 2. Launch Python FastAPI CNN Vision Engine (Port 8000)
# ------------------------------------------------------------------------------
Write-Host "`n[Python Engine] Starting FastAPI CNN Engine on port 8000..." -ForegroundColor Yellow

$pythonCmd = "python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload"
$pythonProcess = Start-Process -FilePath "powershell.exe" `
    -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; Write-Host '--- Python FastAPI CNN Engine (:8000) ---' -ForegroundColor Magenta; $pythonCmd" `
    -PassThru

Write-Host "[Python Engine] Process started (PID: $($pythonProcess.Id)). URL: http://localhost:8000/docs" -ForegroundColor Green

# ------------------------------------------------------------------------------
# 3. Launch Spring Boot 3 Gateway (Port 8080)
# ------------------------------------------------------------------------------
Write-Host "`n[Spring Boot] Starting Java Spring Boot 3 Gateway on port 8080..." -ForegroundColor Yellow

$springProfile = if ($pgActive -and -not $ForceH2) { "default" } else { "h2" }
$springDir = Join-Path $PSScriptRoot "backend-spring"
$springCmd = "cd '$springDir'; .\mvnw.cmd spring-boot:run -Dspring-boot.run.profiles=$springProfile"

$springProcess = Start-Process -FilePath "powershell.exe" `
    -ArgumentList "-NoExit", "-Command", "Write-Host '--- Spring Boot 3 Gateway (:8080, Profile: $springProfile) ---' -ForegroundColor Cyan; $springCmd" `
    -PassThru

Write-Host "[Spring Boot] Process started (PID: $($springProcess.Id)). URL: http://localhost:8080/api/v1/health" -ForegroundColor Green

# ------------------------------------------------------------------------------
# 4. Launch React 18 Frontend Dashboard (Port 5173 / 3000)
# ------------------------------------------------------------------------------
Write-Host "`n[Frontend] Starting React 18 Clinician Dashboard..." -ForegroundColor Yellow

$frontendDir = Join-Path $PSScriptRoot "frontend"
$frontendCmd = "cd '$frontendDir'; npm run dev"

$frontendProcess = Start-Process -FilePath "powershell.exe" `
    -ArgumentList "-NoExit", "-Command", "Write-Host '--- React 18 Clinician Dashboard ---' -ForegroundColor Green; $frontendCmd" `
    -PassThru

Write-Host "[Frontend] Process started (PID: $($frontendProcess.Id)). URL: http://localhost:5173" -ForegroundColor Green

# ------------------------------------------------------------------------------
# 5. Service Summary & Live Status
# ------------------------------------------------------------------------------
Write-Host "`n==========================================================================" -ForegroundColor Cyan
Write-Host "                    ALL SERVICES SUCCESSFULLY LAUNCHED!                   " -ForegroundColor Green
Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host "  - React Clinician Dashboard:  http://localhost:5173" -ForegroundColor White
Write-Host "  - Spring Boot REST Gateway:   http://localhost:8080" -ForegroundColor White
Write-Host "  - FastAPI Swagger Docs:       http://localhost:8000/docs" -ForegroundColor White
Write-Host "  - Database Mode:              $(if ($pgActive -and -not $ForceH2) { 'PostgreSQL (Port 5432)' } else { 'H2 In-Memory' })" -ForegroundColor White
Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host "Press Ctrl+C or close opened PowerShell terminal windows to stop services.`n" -ForegroundColor DarkGray
