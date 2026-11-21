@echo off
title ASD-Facial-CNN Multi-Service Launcher
color 0B
echo ==============================================================================
echo                 ASD-FACIAL-CNN: LAUNCHING FULL-STACK PLATFORM
echo ==============================================================================
echo  1. PyTorch FastAPI Engine : http://127.0.0.1:8000 (Swagger: /docs)
echo  2. Spring Boot Gateway   : http://localhost:8080
echo  3. React Vite Web UI     : http://localhost:5173
echo ==============================================================================
echo.

set ROOT_DIR=%~dp0

echo [*] Step 1/3: Starting PyTorch FastAPI Inference Engine on Port 8000...
start "1. PyTorch AI Engine (Port 8000)" cmd /k "cd /d %ROOT_DIR% && python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --reload"

timeout /t 2 /nobreak >nul

echo [*] Step 2/3: Starting Spring Boot Enterprise Gateway on Port 8080...
start "2. Spring Boot Gateway (Port 8080)" cmd /k "cd /d %ROOT_DIR%backend-spring && mvnw.cmd spring-boot:run -Dspring-boot.run.profiles=h2"

timeout /t 2 /nobreak >nul

echo [*] Step 3/3: Starting React 18 Frontend on Port 5173...
start "3. React 18 Frontend (Port 5173)" cmd /k "cd /d %ROOT_DIR%frontend && npm run dev"

echo.
echo [*] Waiting for services to initialize...
timeout /t 4 /nobreak >nul

echo [*] Launching web browser at http://localhost:5173/ ...
start http://localhost:5173/

echo ==============================================================================
echo  SUCCESS: All three services are running in their respective command windows.
echo  To shut down, simply close each of the 3 command windows.
echo ==============================================================================
pause
