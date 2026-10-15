@echo off
chcp 65001 >nul
echo ======================================================================
echo    VIETCOMBANK AI BANKING ASSISTANT - CHUNKING & DLA INSPECTOR
echo ======================================================================

cd /d "%~dp0"

REM 1. Kiem tra uv run
where uv >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [*] Tim thay uv! Dang khoi chay: uv run python serve.py ...
    uv run python serve.py
    goto :done
)

REM 2. Kiem tra python trong .venv o thu muc goc project
if exist "..\..\..\.venv\Scripts\python.exe" (
    echo [*] Tim thay .venv! Dang khoi chay qua Python .venv ...
    "..\..\..\.venv\Scripts\python.exe" serve.py
    goto :done
)

REM 3. Kiem tra python3.14
where python3.14 >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [*] Tim thay python3.14! Dang khoi chay...
    python3.14 serve.py
    goto :done
)

REM 4. Kiem tra python3.11
where python3.11 >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [*] Tim thay python3.11! Dang khoi chay...
    python3.11 serve.py
    goto :done
)

REM 5. Fallback: Mo truc tiep index.html tren trinh duyet
echo [!] Khong tim thay Python hop le tren he thong.
echo [*] Dang mo truc tiep giao dien qua trinh duyet mac dinh...
start "" "index.html"

:done
