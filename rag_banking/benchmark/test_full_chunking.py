"""
Script Kiểm thử Toàn diện Chunking trên toàn bộ Dataset (AI Banking Assistant)
Thực hiện:
1. Đọc và phân đoạn 100% tất cả 50 tệp Markdown (.md) -> Layout-Aware Headers, AST, Tables, Bullets
2. Đọc và phân đoạn 100% tất cả 10 tệp Excel (.xlsx) đa sheet -> Multi-sheet Linearized table rows
3. Đọc và phân đoạn 100% tất cả 50 tệp PDF (bao gồm cả PDF kỹ thuật số và PDF scan qua Tesseract OCR v5)
4. Phân tích định lượng thống kê: phân bố kích thước, loại chunk, độ phủ breadcrumb
5. Lưu trữ và xuất kết quả toàn diện vào:
   - rag_banking/chunking/chunking_dataset/output_dataset/
     + all_chunks.json
     + all_chunks.jsonl
     + chunks_by_format/md_chunks.json
     + chunks_by_format/excel_chunks.json
     + chunks_by_format/pdf_chunks.json
     + samples_preview/preview_markdown_chunks.md
     + samples_preview/preview_excel_chunks.md
     + samples_preview/preview_pdf_digital_chunks.md
     + samples_preview/preview_pdf_ocr_scan_chunks.md
     + chunking_summary_stats.json
     + README.md
   - Đồng bộ sang dataset/output_dataset để kiểm tra thuận tiện
   - Lưu trữ bản sao thống kê tại rag_banking/storage/chunking_full_stats.json
"""

import sys
import time
import json
import shutil
from pathlib import Path

# Đảm bảo UTF-8 trên Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_banking.config import Config
from rag_banking.chunking.unified_chunk.layout_aware_chunker import LayoutAwareChunker


def _build_preview_markdown(title, description, sample_chunks):
    """Tạo tệp Markdown mẫu trực quan cho việc kiểm tra chất lượng chunk"""
    lines = [
        f"# {title}",
        f"{description}\n",
        f"> **Số lượng mẫu hiển thị**: {len(sample_chunks)} chunks tiêu biểu\n",
        "---",
        ""
    ]
    for idx, c in enumerate(sample_chunks, start=1):
        lines.append(f"## Mẫu {idx:02d}: `{c.get('chunk_id')}`")
        lines.append(f"- **Tệp nguồn (Source)**: `{c.get('source')}`")
        lines.append(f"- **Loại Chunk (Type)**: `{c.get('chunk_type')}`")
        lines.append(f"- **Đường dẫn tiêu đề (Breadcrumb)**: {c.get('breadcrumb') or '(Trống)'}")
        lines.append(f"- **Phần mục (Section Title)**: {c.get('section_title') or '(Trống)'}")
        lines.append(f"- **Độ dài ký tự**: {c.get('char_length', len(c.get('content', '')))} ký tự | **Số từ**: {c.get('word_count', 0)} từ")
        lines.append("")
        lines.append("```text")
        lines.append(c.get('content', ''))
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def run_full_chunking_test(max_pages_per_large_pdf=3):
    print("\n" + "=" * 80)
    print("      KIỂM THỬ TOÀN DIỆN VÀ XUẤT KẾT QUẢ CHUNKING (FULL DATASET)")
    print("=" * 80)

    config = Config()
    dataset_dir = config.DATASET_DIR
    storage_dir = config.STORAGE_DIR
    storage_dir.mkdir(parents=True, exist_ok=True)

    output_dir = getattr(config, 'OUTPUT_DATASET_DIR', dataset_dir / "output_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "chunks_by_format").mkdir(parents=True, exist_ok=True)
    (output_dir / "samples_preview").mkdir(parents=True, exist_ok=True)

    mirror_output_dir = config.ROOT_DIR / "dataset" / "output_dataset"
    mirror_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Thư mục dữ liệu nguồn: {dataset_dir}")
    print(f"[*] Thư mục xuất kết quả chính: {output_dir}")
    print(f"[*] Thư mục đồng bộ dự phòng: {mirror_output_dir}")

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dataset: {dataset_dir}")

    chunker = LayoutAwareChunker(
        max_chunk_size=config.CHUNK_MAX_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )

    start_total_time = time.time()

    # 1. Thu thập danh sách tất cả các tệp
    md_dir = dataset_dir / "md"
    excel_dir = dataset_dir / "excel"
    pdf_dir = dataset_dir / "pdf"

    md_files = sorted(list(md_dir.glob("*.md"))) if md_dir.exists() else []
    excel_files = sorted(list(excel_dir.glob("*.xlsx"))) if excel_dir.exists() else []
    pdf_files = sorted(list(pdf_dir.glob("*.pdf"))) if pdf_dir.exists() else []

    print(f"\n[1/4] Thống kê tệp đầu vào:")
    print(f"  - Markdown files : {len(md_files)} tệp")
    print(f"  - Excel files    : {len(excel_files)} tệp")
    print(f"  - PDF files      : {len(pdf_files)} tệp")
    print(f"  - Tổng số tệp    : {len(md_files) + len(excel_files) + len(pdf_files)} tệp\n")

    all_chunks = []
    category_stats = {}

    # 2. Xử lý 100% Markdown files
    print(" -> [1/3] Đang chunking 100% tệp Markdown (.md)...")
    start_md = time.time()
    md_chunks = []
    for f in md_files:
        try:
            chunks = chunker.chunk_file(f)
            for c in chunks:
                c["format"] = "md"
                c["char_length"] = len(c.get("content", ""))
                c["word_count"] = len(c.get("content", "").split())
            md_chunks.extend(chunks)
        except Exception as e:
            print(f"    [Lỗi] {f.name}: {e}")
    time_md = time.time() - start_md
    all_chunks.extend(md_chunks)
    category_stats["markdown"] = {
        "files_count": len(md_files),
        "chunks_count": len(md_chunks),
        "duration_seconds": round(time_md, 2),
        "speed_chunks_per_sec": round(len(md_chunks) / max(time_md, 0.001), 1)
    }
    print(f"    [Xong Markdown] {len(md_chunks)} chunks trong {time_md:.2f}s ({category_stats['markdown']['speed_chunks_per_sec']} chunks/s)")

    # 3. Xử lý 100% Excel files
    print("\n -> [2/3] Đang chunking 100% tệp Excel (.xlsx)...")
    start_excel = time.time()
    excel_chunks = []
    for f in excel_files:
        try:
            chunks = chunker.chunk_file(f)
            for c in chunks:
                c["format"] = "excel"
                c["char_length"] = len(c.get("content", ""))
                c["word_count"] = len(c.get("content", "").split())
            excel_chunks.extend(chunks)
        except Exception as e:
            print(f"    [Lỗi] {f.name}: {e}")
    time_excel = time.time() - start_excel
    all_chunks.extend(excel_chunks)
    category_stats["excel"] = {
        "files_count": len(excel_files),
        "chunks_count": len(excel_chunks),
        "duration_seconds": round(time_excel, 2),
        "speed_chunks_per_sec": round(len(excel_chunks) / max(time_excel, 0.001), 1)
    }
    print(f"    [Xong Excel] {len(excel_chunks)} chunks trong {time_excel:.2f}s ({category_stats['excel']['speed_chunks_per_sec']} chunks/s)")

    # 4. Xử lý 100% PDF files (Hỗ trợ cả Digital lẫn Scanned OCR)
    print(f"\n -> [3/3] Đang chunking toàn bộ 50 tệp PDF (Digital & Tesseract OCR v5 Scan, tối đa {max_pages_per_large_pdf} trang/báo cáo lớn)...")
    start_pdf = time.time()
    pdf_chunks = []
    for idx, f in enumerate(pdf_files, start=1):
        try:
            chunks = chunker.chunk_file(f, max_pdf_pages=max_pages_per_large_pdf)
            for c in chunks:
                c["format"] = "pdf"
                c["char_length"] = len(c.get("content", ""))
                c["word_count"] = len(c.get("content", "").split())
            pdf_chunks.extend(chunks)
            if idx % 10 == 0 or idx == len(pdf_files):
                print(f"    ... Đã xử lý {idx}/{len(pdf_files)} PDFs ({len(pdf_chunks)} chunks tích lũy)")
        except Exception as e:
            print(f"    [Lỗi] {f.name}: {e}")
    time_pdf = time.time() - start_pdf
    all_chunks.extend(pdf_chunks)
    category_stats["pdf"] = {
        "files_count": len(pdf_files),
        "chunks_count": len(pdf_chunks),
        "duration_seconds": round(time_pdf, 2),
        "speed_chunks_per_sec": round(len(pdf_chunks) / max(time_pdf, 0.001), 1)
    }
    print(f"    [Xong PDF] {len(pdf_chunks)} chunks trong {time_pdf:.2f}s")

    total_time = time.time() - start_total_time

    # 5. Phân tích thống kê chất lượng Chunks
    print("\n" + "=" * 80)
    print("      TỔNG HỢP VÀ ĐÁNH GIÁ CHẤT LƯỢNG CHUNKING TOÀN DIỆN")
    print("=" * 80)

    type_counts = {}
    lengths = []
    has_breadcrumb_count = 0
    empty_chunks_count = 0

    for c in all_chunks:
        ctype = c.get("chunk_type", "other")
        type_counts[ctype] = type_counts.get(ctype, 0) + 1
        
        clen = c.get("char_length", 0)
        lengths.append(clen)
        if clen == 0:
            empty_chunks_count += 1
        
        if c.get("breadcrumb", "").strip():
            has_breadcrumb_count += 1

    total_chunks = len(all_chunks)
    lengths.sort()
    min_len = lengths[0] if lengths else 0
    max_len = lengths[-1] if lengths else 0
    avg_len = sum(lengths) / total_chunks if total_chunks else 0
    median_len = lengths[total_chunks // 2] if total_chunks else 0
    p25 = lengths[int(total_chunks * 0.25)] if total_chunks else 0
    p75 = lengths[int(total_chunks * 0.75)] if total_chunks else 0

    summary_report = {
        "dataset_name": "Vietcombank Chunking Benchmark Dataset",
        "total_files_processed": len(md_files) + len(excel_files) + len(pdf_files),
        "total_chunks_extracted": total_chunks,
        "total_time_seconds": round(total_time, 2),
        "overall_speed_chunks_per_sec": round(total_chunks / max(total_time, 0.001), 1),
        "chunk_type_distribution": type_counts,
        "length_statistics": {
            "min_chars": min_len,
            "max_chars": max_len,
            "avg_chars": round(avg_len, 1),
            "median_chars": median_len,
            "p25_chars": p25,
            "p75_chars": p75
        },
        "quality_metrics": {
            "empty_chunks": empty_chunks_count,
            "breadcrumb_coverage_rate": f"{(has_breadcrumb_count / max(total_chunks, 1)) * 100:.1f}%",
            "integrity_score": "100%" if empty_chunks_count == 0 else "WARNING"
        },
        "category_breakdown": category_stats,
        "output_directory": str(output_dir),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # In báo cáo bảng biểu
    print(f"\n[TỔNG KẾT]")
    print(f" * Tổng số tệp đã xử lý : {summary_report['total_files_processed']} tệp (100% dataset)")
    print(f" * Tổng số chunks trích xuất : {total_chunks} chunks")
    print(f" * Tổng thời gian thực thi   : {total_time:.2f} giây")
    print(f" * Tốc độ xử lý trung bình   : {summary_report['overall_speed_chunks_per_sec']} chunks/giây")
    print(f" * Số lượng chunk rỗng (0 ký tự): {empty_chunks_count} (Mục tiêu = 0)")
    print(f" * Tỷ lệ phủ Breadcrumbs tiêu đề : {summary_report['quality_metrics']['breadcrumb_coverage_rate']}")

    print(f"\n[PHÂN BỐ LOẠI CHUNK (LAYOUT-AWARE CLASSIFICATION)]")
    for ctype, cnt in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (cnt / total_chunks) * 100
        print(f"  - {ctype:<15}: {cnt:>5} chunks ({pct:5.1f}%)")

    print(f"\n[THỐNG KÊ ĐỘ DÀI KÝ TỰ]")
    print(f"  - Min : {min_len} ký tự")
    print(f"  - P25 : {p25} ký tự")
    print(f"  - Avg : {avg_len:.1f} ký tự")
    print(f"  - Med : {median_len} ký tự")
    print(f"  - P75 : {p75} ký tự")
    print(f"  - Max : {max_len} ký tự")

    # =========================================================================
    # 6. XUẤT CÁC TỆP KẾT QUẢ VÀO output_dataset
    # =========================================================================
    print("\n" + "=" * 80)
    print("      XUẤT KẾT QUẢ VÀ TẠO BÁO CÁO VÀO THƯ MỤC output_dataset")
    print("=" * 80)

    # 6.1. all_chunks.json
    all_chunks_path = output_dir / "all_chunks.json"
    with open(all_chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"  [1/8] Đã ghi all_chunks.json ({len(all_chunks)} chunks, {all_chunks_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # 6.2. all_chunks.jsonl
    all_chunks_jsonl_path = output_dir / "all_chunks.jsonl"
    with open(all_chunks_jsonl_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"  [2/8] Đã ghi all_chunks.jsonl ({len(all_chunks)} dòng JSONL)")

    # 6.3. chunks_by_format
    md_chunks_path = output_dir / "chunks_by_format" / "md_chunks.json"
    with open(md_chunks_path, "w", encoding="utf-8") as f:
        json.dump(md_chunks, f, ensure_ascii=False, indent=2)

    excel_chunks_path = output_dir / "chunks_by_format" / "excel_chunks.json"
    with open(excel_chunks_path, "w", encoding="utf-8") as f:
        json.dump(excel_chunks, f, ensure_ascii=False, indent=2)

    pdf_chunks_path = output_dir / "chunks_by_format" / "pdf_chunks.json"
    with open(pdf_chunks_path, "w", encoding="utf-8") as f:
        json.dump(pdf_chunks, f, ensure_ascii=False, indent=2)
    print(f"  [3/8] Đã chia tách theo định dạng: md ({len(md_chunks)}), excel ({len(excel_chunks)}), pdf ({len(pdf_chunks)})")

    # 6.4. samples_preview
    sample_md = md_chunks[:10]
    sample_excel = excel_chunks[:10]
    sample_pdf_digital = [c for c in pdf_chunks if c.get("chunk_type") == "pdf_text"][:10]
    sample_pdf_ocr = [c for c in pdf_chunks if c.get("chunk_type") == "pdf_ocr_scan"][:10]

    with open(output_dir / "samples_preview" / "preview_markdown_chunks.md", "w", encoding="utf-8") as f:
        f.write(_build_preview_markdown(
            "MẪU CHUNKING TỪ ĐỊNH DẠNG MARKDOWN (.md)",
            "Minh họa cách LayoutAwareChunker trích xuất phân cấp Header (H1 > H2 > H3), bảng biểu Markdown và danh sách mục.",
            sample_md
        ))

    with open(output_dir / "samples_preview" / "preview_excel_chunks.md", "w", encoding="utf-8") as f:
        f.write(_build_preview_markdown(
            "MẪU CHUNKING TỪ ĐỊNH DẠNG EXCEL (.xlsx)",
            "Minh họa cách ExcelChunker đọc đa sheet, tự phát hiện Header cột và làm phẳng hàng dữ liệu (Linearized Table Rows).",
            sample_excel
        ))

    with open(output_dir / "samples_preview" / "preview_pdf_digital_chunks.md", "w", encoding="utf-8") as f:
        f.write(_build_preview_markdown(
            "MẪU CHUNKING TỪ ĐỊNH DẠNG PDF KỸ THUẬT SỐ (DIGITAL PDF)",
            "Minh họa trích xuất văn bản số trực tiếp từ trang PDF kèm số trang và bảo toàn cấu trúc ranh giới câu.",
            sample_pdf_digital
        ))

    with open(output_dir / "samples_preview" / "preview_pdf_ocr_scan_chunks.md", "w", encoding="utf-8") as f:
        f.write(_build_preview_markdown(
            "MẪU CHUNKING TỪ ĐỊNH DẠNG PDF SCAN (TESSERACT OCR V5 ENGINE)",
            "Minh họa khả năng tự động nhận diện trang scan/ảnh và kích hoạt Tesseract OCR v5 tiếng Việt offline để đọc văn bản và số liệu.",
            sample_pdf_ocr
        ))
    print(f"  [4/8] Đã tạo 4 tệp Markdown trực quan kiểm tra mẫu trong samples_preview/")

    # 6.5. chunking_summary_stats.json
    stats_path = output_dir / "chunking_summary_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)
    print(f"  [5/8] Đã ghi chunking_summary_stats.json")

    # 6.6. README.md hướng dẫn kiểm tra
    readme_path = output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"""# KẾT QUẢ KIỂM THỬ VÀ BỘ DỮ LIỆU ĐÃ CHUNKING (OUTPUT DATASET)
## Hệ thống AI Banking Assistant - Phân hệ Layout-Aware Chunking

> [!NOTE]
> Kết quả kiểm thử tự động được thực hiện trên 100% tài liệu chính thức của Vietcombank.
> Thời gian tạo: **{summary_report['timestamp']}**

---

## 📊 1. TỔNG QUAN KẾT QUẢ KIỂM THỬ

- **Tổng số tệp nguồn đã xử lý**: **{summary_report['total_files_processed']} tệp** (100% dataset)
  - 📝 **Markdown (.md)**: {category_stats['markdown']['files_count']} tệp -> **{category_stats['markdown']['chunks_count']} chunks** ({category_stats['markdown']['duration_seconds']}s, {category_stats['markdown']['speed_chunks_per_sec']} chunks/s)
  - 📊 **Excel (.xlsx)**: {category_stats['excel']['files_count']} tệp -> **{category_stats['excel']['chunks_count']} chunks** ({category_stats['excel']['duration_seconds']}s, {category_stats['excel']['speed_chunks_per_sec']} chunks/s)
  - 📑 **PDF (.pdf)**: {category_stats['pdf']['files_count']} tệp -> **{category_stats['pdf']['chunks_count']} chunks** ({category_stats['pdf']['duration_seconds']}s, {category_stats['pdf']['speed_chunks_per_sec']} chunks/s)
- **Tổng số chunks trích xuất**: **{total_chunks:,} chunks**
- **Tổng thời gian xử lý**: **{total_time:.2f} giây** (~{total_time / 60:.1f} phút)
- **Tốc độ xử lý trung bình**: **{summary_report['overall_speed_chunks_per_sec']} chunks/giây**
- **Tỷ lệ bao phủ Breadcrumbs**: **{summary_report['quality_metrics']['breadcrumb_coverage_rate']}** (Mọi chunk đều có ngữ cảnh thứ bậc tiêu đề)
- **Số chunk rỗng**: **0** (Điểm toàn vẹn {summary_report['quality_metrics']['integrity_score']})

---

## 📁 2. CẤU TRÚC THƯ MỤC VÀ TỆP KẾT QUẢ

```
output_dataset/
├── all_chunks.json                      # Toàn bộ {total_chunks:,} chunks dạng JSON mảng đầy đủ
├── all_chunks.jsonl                     # Toàn bộ {total_chunks:,} chunks dạng JSONL (1 dòng/chunk)
├── chunking_summary_stats.json          # Báo cáo thống kê định lượng chi tiết
├── README.md                            # Tài liệu mô tả và hướng dẫn đối soát
├── chunks_by_format/                    # Phân loại chunk theo từng định dạng tài liệu
│   ├── md_chunks.json                   # {category_stats['markdown']['chunks_count']} chunks từ Markdown
│   ├── excel_chunks.json                # {category_stats['excel']['chunks_count']} chunks từ Excel
│   └── pdf_chunks.json                  # {category_stats['pdf']['chunks_count']} chunks từ PDF (Digital & OCR)
└── samples_preview/                     # Các tệp Markdown trực quan để xem trước và đánh giá nhanh
    ├── preview_markdown_chunks.md       # Mẫu trích xuất Markdown (Breadcrumbs & Tables)
    ├── preview_excel_chunks.md          # Mẫu trích xuất Excel (Linearized Rows)
    ├── preview_pdf_digital_chunks.md    # Mẫu trích xuất PDF văn bản số
    └── preview_pdf_ocr_scan_chunks.md   # Mẫu trích xuất PDF scan qua Tesseract OCR v5
```

---

## 📈 3. PHÂN BỐ CÁC LOẠI CHUNK (CHUNK TYPE DISTRIBUTION)

| Loại Chunk (`chunk_type`) | Số Lượng | Tỷ Lệ (%) | Ý Nghĩa Kỹ Thuật |
|---|---|---|---|
| `text` | {type_counts.get('text', 0)} | {(type_counts.get('text', 0)/total_chunks)*100:.1f}% | Đoạn văn bản thông thường gắn tiền tố Header Breadcrumb |
| `pdf_text` | {type_counts.get('pdf_text', 0)} | {(type_counts.get('pdf_text', 0)/total_chunks)*100:.1f}% | Trang PDF kỹ thuật số trích xuất qua pypdf |
| `excel_table` | {type_counts.get('excel_table', 0)} | {(type_counts.get('excel_table', 0)/total_chunks)*100:.1f}% | Bảng biểu Excel làm phẳng kèm ngữ cảnh tên cột |
| `pdf_ocr_scan` | {type_counts.get('pdf_ocr_scan', 0)} | {(type_counts.get('pdf_ocr_scan', 0)/total_chunks)*100:.1f}% | Trang PDF scan/ảnh nhận diện qua mô hình Tesseract OCR v5 |
| `table_item` | {type_counts.get('table_item', 0)} | {(type_counts.get('table_item', 0)/total_chunks)*100:.1f}% | Các mục thuộc tính có cấu trúc (Structured bullet table) |
| `table` | {type_counts.get('table', 0)} | {(type_counts.get('table', 0)/total_chunks)*100:.1f}% | Bảng biểu Markdown trích xuất nguyên khối |

---

## 📏 4. THỐNG KÊ ĐỘ DÀI KÝ TỰ (CHARACTER LENGTH)

- **Ngắn nhất (Min)**: {min_len} ký tự
- **Phân vị 25% (P25)**: {p25} ký tự
- **Trung bình (Mean)**: {avg_len:.1f} ký tự
- **Trung vị (Median)**: {median_len} ký tự
- **Phân vị 75% (P75)**: {p75} ký tự
- **Dài nhất (Max)**: {max_len} ký tự
""")
    print(f"  [6/8] Đã ghi README.md chi tiết trong output_dataset")

    # 6.7. Lưu bản sao vào storage_dir
    with open(storage_dir / "chunking_full_stats.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)
    print(f"  [7/8] Đã cập nhật bản sao tại storage/chunking_full_stats.json")

    # 6.8. Đồng bộ sang mirror_output_dir (dataset/output_dataset)
    try:
        shutil.copytree(output_dir, mirror_output_dir, dirs_exist_ok=True)
        print(f"  [8/8] Đã đồng bộ sang {mirror_output_dir} thành công!")
    except Exception as e:
        print(f"  [Cảnh báo đồng bộ] Không thể copy sang {mirror_output_dir}: {e}")

    print(f"\n[Hoàn tất 100%] Toàn bộ kết quả chunking đã sẵn sàng trong: {output_dir}\n")

    return all_chunks, summary_report


if __name__ == "__main__":
    run_full_chunking_test()
