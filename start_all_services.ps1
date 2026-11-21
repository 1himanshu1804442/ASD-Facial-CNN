# ASD-Facial-CNN Full-Stack Launch Script
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "                 ASD-FACIAL-CNN: LAUNCHING FULL-STACK PLATFORM" -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "  1. PyTorch FastAPI Engine : http://127.0.0.1:8000 (Swagger: /docs)" -ForegroundColor Yellow
Write-Host "  2. Spring Boot Gateway    : http://localhost:8080" -ForegroundColor Yellow
Write-Host "  3. React Vite UI          : http://localhost:5173" -ForegroundColor Yellow
Write-Host "==============================================================================" -ForegroundColor Cyan

$root = $PSScriptRoot

Write-Host "`n[*] Starting PyTorch FastAPI Inference Engine on Port 8000..." -ForegroundColor Cyan
Start-Process cmd -ArgumentList "/k cd /d `"$root`" && python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --reload"

Start-Sleep -Seconds 2

Write-Host "[*] Starting Spring Boot Gateway on Port 8080..." -ForegroundColor Cyan
Start-Process cmd -ArgumentList "/k cd /d `"$root\backend-spring`" && mvnw.cmd spring-boot:run -Dspring-boot.run.profiles=h2"

Start-Sleep -Seconds 2

Write-Host "[*] Starting React 18 Frontend on Port 5173..." -ForegroundColor Cyan
Start-Process cmd -ArgumentList "/k cd /d `"$root\frontend`" && npm run dev"

Start-Sleep -Seconds 3
Write-Host "[*] Launching browser at http://localhost:5173/..." -ForegroundColor Green
Start-Process "http://localhost:5173/"

Write-Host "`nAll 3 services successfully launched!" -ForegroundColor Green
