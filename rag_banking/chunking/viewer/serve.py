#!/usr/bin/env python3
"""
Lightweight HTTP Server Launcher cho Chunker & Bounding Box Viewer
Sử dụng thư viện chuẩn của Python (không cần cài thêm bất kỳ package nào).
"""
import sys
import os
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PORT = 8501
VIEWER_DIR = Path(__file__).resolve().parent

DATASET_DIR = (VIEWER_DIR.parents[2] / "dataset" / "chunking_dataset").resolve()

class CustomHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(VIEWER_DIR), **kwargs)

    def translate_path(self, path):
        import urllib.parse
        clean_path = urllib.parse.unquote(path.split('?', 1)[0].split('#', 1)[0])
        if clean_path.startswith('/dataset_raw/'):
            rel = clean_path[len('/dataset_raw/'):].lstrip('/')
            target = DATASET_DIR / rel
            return str(target)
        return super().translate_path(path)

    def end_headers(self):
        # Enable CORS and disable aggressive caching for local development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else PORT
    for p in range(port, port + 20):
        try:
            server = HTTPServer(('127.0.0.1', p), CustomHandler)
            port = p
            break
        except OSError:
            continue

    url = f"http://127.0.0.1:{port}/index.html"
    print("=" * 72)
    print("   VIETCOMBANK AI BANKING ASSISTANT - DLA & CHUNKING INSPECTOR")
    print("=" * 72)
    print(f"[*] Máy chủ giao diện đang chạy tại: {url}")
    print("[*] Đang mở trình duyệt mặc định...")
    print("[*] Nhấn Ctrl+C để dừng máy chủ.")
    print("=" * 72)

    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Đã dừng máy chủ giao diện.")
        server.server_close()

if __name__ == '__main__':
    main()
