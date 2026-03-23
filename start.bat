@echo off
title Project Vesi

echo Starting backend...
:: Add your conda env name and path
start "Vesi Backend" /min cmd /k "call C:\Users\Aaro\anaconda3\condabin\conda.bat activate llama-python && cd /d %~dp0server && python main.py"

echo Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

echo Starting frontend...
start "Vesi Frontend" /min cmd /k "cd /d %~dp0client && python -m http.server 5500"

echo.
echo Vesi is running!
echo   Backend:  http://localhost:8000
echo   Frontend: http://[::]:5500
echo.
echo Press any key to shut down...
pause >nul
echo Shutting down...
taskkill /FI "WINDOWTITLE eq Vesi Backend*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Vesi Frontend*" /F >nul 2>&1

taskkill /FI "WINDOWTITLE eq Vesi*" /F >nul 2>&1
echo Done.
timeout /t 2 /nobreak >nul
