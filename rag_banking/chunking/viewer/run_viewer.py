#!/usr/bin/env python3
"""
Launcher khởi chạy Frontend Trực quan hóa Bounding Box & Chunking
Chạy trực tiếp từ thư mục `rag_banking/chunking/viewer`:
    python run_viewer.py
hoặc:
    uv run python run_viewer.py
"""
import sys
import subprocess
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SERVE_SCRIPT = CURRENT_DIR / "serve.py"

if __name__ == "__main__":
    if not SERVE_SCRIPT.exists():
        print(f"[Error] Không tìm thấy file máy chủ tại: {SERVE_SCRIPT}")
        sys.exit(1)
    
    subprocess.run([sys.executable, str(SERVE_SCRIPT)])
