"""
Kịch bản chạy phân đoạn chuẩn xác cho toàn bộ Dataset 1 (AI Banking Assistant)
Xử lý toàn bộ các tệp Word DOCX nghiệp vụ số ngân hàng.
Xuất kết quả từng tệp ra từng file JSON độc lập trong thư mục chunks_per_file.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Đảm bảo mã hóa UTF-8 cho console trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path("d:/Desktop/GitHub/AI Banking Asistant")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.chunking.unified_chunk.layout_aware_chunker import LayoutAwareChunker

DATASET_1_DIR = PROJECT_ROOT / "dataset" / "dataset_1"
CHUNK_DATASET_1_DIR = PROJECT_ROOT / "agent_banking" / "chunking" / "chunking_dataset" / "dataset_1"

OUT_DIR_1 = DATASET_1_DIR / "chunks_per_file"
OUT_DIR_2 = CHUNK_DATASET_1_DIR / "chunks_per_file"

OUT_DIR_1.mkdir(parents=True, exist_ok=True)
OUT_DIR_2.mkdir(parents=True, exist_ok=True)


def run_dataset_1_pipeline():
    start_total = time.time()

    print("=" * 80)
    print("      CHẠY PHÂN ĐOẠN DATASET 1 - TOÀN BỘ TỆP WORD DOCX NGHIỆP VỤ")
    print("=" * 80)

    # Khởi tạo bộ phân đoạn LayoutAwareChunker
    print("[*] Đang khởi tạo LayoutAwareChunker (0% Regex, OpenXML AST + Typographic Flow)...")
    chunker = LayoutAwareChunker(
        max_chunk_size=800,
        min_section_size=120
    )

    docx_dir = DATASET_1_DIR / "docx"
    docx_files = sorted(list(docx_dir.glob("*.docx")))

    print(f"[*] Thư mục DOCX: {docx_dir} ({len(docx_files)} tệp)")
    print(f"[*] Thư mục xuất JSON: {OUT_DIR_1}")
    print(f"[*] Thư mục đồng bộ  : {OUT_DIR_2}\n")

    all_chunks: list[dict] = []
    file_reports: list[dict] = []

    print("-" * 80)
    print(f"BẮT ĐẦU PHÂN ĐOẠN {len(docx_files)} TỆP DOCX NGHIỆP VỤ NGÂN HÀNG")
    print("-" * 80)

    docx_start = time.time()
    for idx, docx_path in enumerate(docx_files, 1):
        t0 = time.time()
        fn = docx_path.name
        stem = docx_path.stem

        try:
            chunks = chunker.chunk_docx(docx_path)
            dt = time.time() - t0

            # Ghi ra từng file JSON độc lập
            json_name = f"{stem}.json"
            p1 = OUT_DIR_1 / json_name
            p2 = OUT_DIR_2 / json_name

            with open(p1, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            with open(p2, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

            all_chunks.extend(chunks)

            total_words = sum(c.get("word_count", len(c.get("content", "").split())) for c in chunks)
            report = {
                "file": fn,
                "format": "docx",
                "chunks": len(chunks),
                "words": total_words,
                "duration_sec": round(dt, 2),
                "status": "SUCCESS"
            }
            file_reports.append(report)

            print(f"  [{idx:02d}/{len(docx_files)}] DOCX: {fn[:55]:<55} -> {len(chunks):>3} chunks ({total_words:>5} từ, {dt:>5.2f}s)")

        except Exception as e:
            print(f"  [{idx:02d}/{len(docx_files)}] DOCX: {fn} -> LỖI: {e}")
            file_reports.append({
                "file": fn,
                "format": "docx",
                "chunks": 0,
                "words": 0,
                "duration_sec": round(time.time() - t0, 2),
                "status": f"FAILED: {e}"
            })

    docx_duration = time.time() - docx_start
    print(f"\n>> ĐÃ HOÀN TẤT PHÂN ĐOẠN {len(docx_files)} TỆP DOCX TRONG {docx_duration:.2f} GIÂY!\n")

    # Lưu file tổng hợp all_chunks.json và báo cáo thống kê
    total_elapsed = time.time() - start_total
    all_chunks_p1 = DATASET_1_DIR / "all_chunks.json"
    all_chunks_p2 = CHUNK_DATASET_1_DIR / "all_chunks.json"

    with open(all_chunks_p1, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    with open(all_chunks_p2, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    total_words_count = sum(r["words"] for r in file_reports)

    summary_meta = {
        "dataset_name": "Dataset 1 (Ultra-Clean Baseline Text Subset)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_files": len(file_reports),
        "total_docx": len(docx_files),
        "total_chunks": len(all_chunks),
        "total_words": total_words_count,
        "average_words_per_chunk": round(total_words_count / max(len(all_chunks), 1), 1),
        "docx_elapsed_sec": round(docx_duration, 2),
        "total_elapsed_sec": round(total_elapsed, 2),
        "output_chunks_per_file_dir": str(OUT_DIR_1),
        "file_reports": file_reports
    }

    summary_path = DATASET_1_DIR / "chunking_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_meta, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("                     TỔNG KẾT TOÀN DIỆN DATASET 1")
    print("=" * 80)
    print(f" Tổng số tệp đã phân đoạn : {len(file_reports)} / {len(docx_files)} tệp DOCX")
    print(f" Tổng số Chunks tạo ra     : {len(all_chunks)} chunks")
    print(f" Tổng số từ trích xuất     : {total_words_count:,} từ")
    print(f" Trung bình từ / chunk     : {summary_meta['average_words_per_chunk']} từ/chunk")
    print(f" Tổng thời gian thực thi   : {total_elapsed:.2f} giây")
    print(f" Đã xuất {len(file_reports)} file JSON vào : {OUT_DIR_1}")
    print(f" Đã đồng bộ sang          : {OUT_DIR_2}")
    print(f" Tệp tổng hợp             : {all_chunks_p1}")
    print("=" * 80)


if __name__ == "__main__":
    run_dataset_1_pipeline()
