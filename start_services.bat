@echo off
setlocal
cd /d "%~dp0"

echo ===================================================
echo   AI Board Game Portal
echo ===================================================

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found.
    echo Install Python 3.10 or later and run:
    echo   python -m pip install -r requirements.txt
    pause
    exit /b 1
)

if not exist "web\index.html" (
    echo [ERROR] web\index.html was not found.
    echo Build the frontend with:
    echo   cd frontend
    echo   npm ci
    echo   npm run build
    pause
    exit /b 1
)

if not exist "build\Release\play_monte_carlo_cuda.exe" (
    echo [WARNING] CUDA executable was not found.
    echo Build the project in Release mode before using the GPU AI.
)

echo Starting FastAPI at http://127.0.0.1:8889/
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8889/"
python -m uvicorn app:app --host 127.0.0.1 --port 8889

if errorlevel 1 (
    echo.
    echo [ERROR] Server startup failed.
    echo Run: python -m pip install -r requirements.txt
    pause
)

endlocal
