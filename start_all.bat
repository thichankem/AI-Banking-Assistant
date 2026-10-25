@echo off
chcp 65001 > nul
echo ======================================================================
echo    🚀 KHỞI ĐỘNG ĐỒNG THỜI FRONTEND VÀ ARIZE PHOENIX TRACING
echo ======================================================================
echo [1/2] Đang mở Arize Phoenix Tracing Server (Port 6006)...
start "Arize Phoenix Server (Tracing)" cmd /k ".venv\Scripts\python.exe scripts\run_phoenix.py"

echo Chờ 3 giây để Phoenix Collector sẵn sàng...
timeout /t 3 /nobreak > nul

echo [2/2] Đang mở AI Banking Assistant Web Server (Port 8000)...
start "AI Banking Assistant (Web App)" cmd /k ".venv\Scripts\python.exe main.py"

echo ======================================================================
echo ✅ ĐÃ KHỞI CHẠY THÀNH CÔNG 2 DỊCH VỤ:
echo   - Giao diện Web (Frontend) : http://127.0.0.1:8000
echo   - Giám sát Tracing (Phoenix): http://127.0.0.1:6006
echo   - Tài liệu API (Swagger UI): http://127.0.0.1:8000/docs
echo ======================================================================
pause
