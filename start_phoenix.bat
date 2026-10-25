@echo off
chcp 65001 > nul
echo ======================================================================
echo    🔥 KHỞI CHẠY ARIZE PHOENIX TRACING & OBSERVABILITY SERVER
echo ======================================================================
echo Đang khởi động máy chủ Arize Phoenix...
echo Web Dashboard: http://127.0.0.1:6006
echo OTLP Endpoint: http://127.0.0.1:6006/v1/traces
echo ======================================================================
.venv\Scripts\python.exe scripts\run_phoenix.py
pause
