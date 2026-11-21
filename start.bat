@echo off
title ASD-Facial-CNN 1-Click Launcher
echo ==========================================================================
echo               Starting ASD-Facial-CNN Full-Stack Platform
echo ==========================================================================
cd /d "%~dp0"
echo.
echo Select Database Mode:
echo   [1] H2 In-Memory DB  (Recommended for Instant Demo / Viva - Zero Setup)
echo   [2] PostgreSQL DB    (Requires local PostgreSQL with user 'neurograph')
echo.
choice /C 12 /T 3 /D 1 /M "Choose option (default is [1] in 3 seconds)"
if errorlevel 2 goto POSTGRES
goto H2

:POSTGRES
echo Starting with PostgreSQL Profile...
powershell -NoProfile -ExecutionPolicy Bypass -File ".\dev.ps1"
goto END

:H2
echo Starting with H2 In-Memory Profile...
powershell -NoProfile -ExecutionPolicy Bypass -File ".\dev.ps1" -ForceH2
goto END

:END
pause
