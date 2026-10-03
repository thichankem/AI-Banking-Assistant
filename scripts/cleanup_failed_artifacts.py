"""
Cleanup Failed Artifacts and Corrupted Downloads
Tự động dọn dẹp các báo cáo, tạo tác bị lỗi không tải được trên Google NotebookLM và bộ nhớ cục bộ.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# Cấu hình UTF-8 cho stdout/stderr trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

NOTEBOOKS = {
    "1938151d-f91d-47df-b395-4b27aac0a211": "AI Agent banking",
    "f07ce8fc-f536-4624-af36-3f614e6637ca": "AI Agent",
    "880c1277-4ac1-4b2b-9565-54913bda81df": "RAG",
    "d3f1452a-ba79-4ca9-875e-a5d7bb20ca74": "Strategic Document Chunking",
    "f3f23d9f-31db-4025-873e-fd47fec79a87": "Khảo sát về những gian lận trong ngân hàng",
}

LOCAL_DIRS = [
    Path(r"d:\Desktop\GitHub\AI Banking Asistant\docs\research"),
    Path(r"C:\Users\ADMIN\Downloads\gemini-notebook"),
]

def clean_local_storage():
    """Quét và xóa các tệp tải bị lỗi, tệp rỗng (0 bytes), tệp tải dở (.tmp, .crdownload, .part) hoặc tệp lỗi JSON."""
    print("=== [1/2] KIỂM TRA & DỌN DẸP TỆP HỎNG TẠI BỘ NHỚ CỤC BỘ ===")
    deleted_count = 0

    for base_dir in LOCAL_DIRS:
        if not base_dir.exists():
            continue
        
        for file_path in base_dir.rglob("*"):
            if not file_path.is_file():
                continue
            
            should_delete = False
            reason = ""

            # 1. Tệp rỗng (0 bytes)
            try:
                size = file_path.stat().st_size
                if size == 0:
                    should_delete = True
                    reason = "Tệp rỗng (0 bytes)"
            except Exception:
                continue

            # 2. Tệp tải dở / tạm thời
            if not should_delete and file_path.suffix.lower() in [".tmp", ".crdownload", ".part", ".partial"]:
                should_delete = True
                reason = f"Tệp tải dở ({file_path.suffix})"

            # 3. Tệp chứa thông báo lỗi tải về từ MCP / API
            if not should_delete and file_path.suffix.lower() in [".json", ".txt", ".md"] and size < 1024:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read().strip()
                        if '"status": "error"' in content or '"status":"error"' in content or '"error_reason"' in content:
                            should_delete = True
                            reason = "Tệp chứa payload báo lỗi (error payload)"
                except Exception:
                    pass

            if should_delete:
                try:
                    file_path.unlink()
                    print(f"  [ĐÃ XÓA] {file_path.name} | Lý do: {reason} | Đường dẫn: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"  [LỖI XÓA] {file_path.name}: {e}")

    if deleted_count == 0:
        print("  -> Không có tệp rác hoặc tệp lỗi nào trong thư mục cục bộ.")
    else:
        print(f"  -> Đã dọn dẹp thành công {deleted_count} tệp lỗi cục bộ.")
    return deleted_count

def clean_notebooklm_studio():
    """Kiểm tra Studio trên Google NotebookLM và xóa vĩnh viễn các tạo tác bị lỗi (failed / error)."""
    print("\n=== [2/2] KIỂM TRA & XÓA CÁC TẠO TÁC BỊ LỖI TRÊN NOTEBOOKLM CLOUD ===")
    cloud_deleted_count = 0

    for nb_id, nb_name in NOTEBOOKS.items():
        print(f"Đang kiểm tra sổ tay: {nb_name} ({nb_id})...")
        try:
            # Gọi nlm studio status --json với UTF-8 encoding
            res = subprocess.run(
                ["nlm", "studio", "status", nb_id, "--json"],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                check=False
            )
            if res.returncode != 0:
                print(f"  [CẢNH BÁO] Không thể lấy trạng thái Studio cho {nb_name}: {res.stderr.strip()}")
                continue

            raw_output = res.stdout.strip()
            if not raw_output:
                print(f"  -> Không có tạo tác nào trên {nb_name}.")
                continue

            data = json.loads(raw_output)
            # data có thể là list hoặc dict tùy phiên bản CLI
            if isinstance(data, dict):
                artifacts = data.get("artifacts", [])
            elif isinstance(data, list):
                artifacts = data
            else:
                artifacts = []

            failed_artifacts = []
            for art in artifacts:
                status = str(art.get("status", "")).lower()
                error_reason = art.get("error_reason")
                if status in ["failed", "error"] or error_reason is not None:
                    failed_artifacts.append(art)

            if not failed_artifacts:
                print(f"  -> Tất cả {len(artifacts)} tạo tác trên {nb_name} đều hợp lệ (COMPLETED/IN_PROGRESS).")
                continue

            for failed in failed_artifacts:
                art_id = failed.get("artifact_id") or failed.get("id")
                title = failed.get("title", "Không tên")
                reason = failed.get("error_reason") or failed.get("status")
                print(f"  -> Phát hiện tạo tác lỗi: '{title}' (ID: {art_id}) - Lý do: {reason}")
                
                # Gọi nlm studio delete --confirm
                del_res = subprocess.run(
                    ["nlm", "studio", "delete", nb_id, art_id, "--confirm"],
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False
                )
                if del_res.returncode == 0:
                    print(f"    [ĐÃ XÓA TRÊN CLOUD] Xóa thành công tạo tác lỗi: {title}")
                    cloud_deleted_count += 1
                else:
                    print(f"    [LỖI XÓA CLOUD] {del_res.stderr.strip()}")

        except Exception as e:
            print(f"  [LỖI XỬ LÝ] Sổ tay {nb_name}: {e}")

    if cloud_deleted_count == 0:
        print("  -> Không có tạo tác lỗi (failed) nào trên cả 5 sổ tay NotebookLM.")
    else:
        print(f"  -> Đã xóa thành công {cloud_deleted_count} tạo tác lỗi trên NotebookLM Cloud.")
    return cloud_deleted_count

if __name__ == "__main__":
    print("Khởi động quy trình tự động dọn dẹp báo cáo & tạo tác lỗi...")
    local_count = clean_local_storage()
    cloud_count = clean_notebooklm_studio()
    print(f"\n[KẾT QUẢ TỔNG QUAN] Hoàn tất dọn dẹp: {local_count} tệp cục bộ, {cloud_count} tạo tác cloud.")
