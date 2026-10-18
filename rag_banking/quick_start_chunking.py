"""
HƯỚNG DẪN SỬ DỤNG NHANH BỘ CHUNKING (QUICK START)
AI Banking Assistant - Phân hệ Phân đoạn Tài liệu Nhận thức Bố cục (DLA)

Ví dụ mẫu siêu đơn giản, siêu dễ hiểu cho 2 luồng cốt lõi: DOCX và PDF.
"""

import sys
from pathlib import Path

# Đảm bảo đường dẫn gốc dự án luôn sẵn sàng
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Đảm bảo in UTF-8 trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rag_banking.chunking.unified_chunk.layout_aware_chunker import LayoutAwareChunker

def demo_quick_start():
    # 1. Khởi tạo bộ phân đoạn hợp nhất
    chunker = LayoutAwareChunker(
        max_chunk_size=500,     # Ngân sách độ dài tối ưu (~350-500 từ)
        chunk_overlap=0         # Không gối đầu nội dung (0% Content Overlap Law)
    )

    print("=== DEMO PHÂN ĐOẠN ĐA ĐỊNH DẠNG (DOCX, PDF) ===\n")

    # 2. Phân đoạn tệp Word DOCX
    docx_file = Path("dataset/chunking_dataset/docx/vcb_md_01__faq__the_tin_dung_va_the_ghi_no_ngan_hang.docx")
    if docx_file.exists():
        docx_chunks = chunker.chunk_docx(docx_file)
        print(f"[1] DOCX: {docx_file.name}")
        print(f"    -> Sinh ra: {len(docx_chunks)} chunks")
        print(f"    -> Mẫu chunk đầu tiên: {docx_chunks[0]['content'][:120]}...\n")

    # 3. Phân đoạn tệp Báo cáo tài chính PDF (Tự động kích hoạt OCR tiếng Việt nếu là scan)
    pdf_file = Path("dataset/chunking_dataset/pdf/vcb_pdf_01_bctc_2024_hop_nhat_kiem_toan.pdf")
    if pdf_file.exists():
        pdf_chunks = chunker.chunk_pdf(pdf_file, max_pages=2)
        print(f"[2] PDF: {pdf_file.name}")
        print(f"    -> Sinh ra: {len(pdf_chunks)} chunks")
        print(f"    -> Mẫu chunk đầu tiên: {pdf_chunks[0]['content'][:120]}...\n")

    # 4. Phân đoạn tổng quát bằng 1 dòng lệnh duy nhất cho bất kỳ tệp nào:
    # chunks = chunker.chunk_file("path/to/any_document.docx")

if __name__ == "__main__":
    demo_quick_start()
