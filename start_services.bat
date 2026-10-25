@echo off
cd /d "%~dp0"
echo ======================================================================
echo   DANG KHOI CHAY HE THONG AI BANKING ASSISTANT VA ARIZE PHOENIX
echo ======================================================================

echo [1/2] Dang bat Arize Phoenix Tracing (Port 6006)...
start "Arize Phoenix" /min cmd /k "cd /d ""%~dp0"" && .venv\Scripts\python.exe scripts\run_phoenix.py"

echo [2/2] Dang bat Web Server Chat (Port 8000)...
start "AI Banking Assistant" /min cmd /k "cd /d ""%~dp0"" && .venv\Scripts\python.exe main.py"

echo.
echo ======================================================================
echo   DA KHOI CHAY THANH CONG!
echo   * Web Frontend : http://127.0.0.1:8000
echo   * Phoenix Trace: http://127.0.0.1:6006
echo ======================================================================
