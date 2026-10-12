"""
PDF Layout-Aware & OCR Chunker
Phân đoạn tài liệu PDF ngân hàng có nhận thức bố cục và tự động OCR:
- Tự động nhận diện PDF văn bản số (digital) hay PDF scan hình ảnh.
- Nếu trang PDF là bản scan (0 ký tự số), tự động kích hoạt RapidOCR để trích xuất nội dung và số liệu.
- Giữ nguyên thông tin số trang và nguồn tài liệu cho từng chunk.
"""

import re
from pathlib import Path
import pypdf
from .ocr_extractor import OCRExtractor

class PDFChunker:
    def __init__(self, ocr_extractor=None, max_chunk_size=800, chunk_overlap=100):
        self.ocr_extractor = ocr_extractor or OCRExtractor()
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pdf(self, pdf_path, max_pages=None):
        """
        Đọc và phân đoạn 1 tệp PDF ngân hàng:
        Hỗ trợ cả PDF kỹ thuật số lẫn PDF scan hình ảnh (qua OCR).
        """
        pdf_path = Path(pdf_path)
        chunks = []

        try:
            reader = pypdf.PdfReader(str(pdf_path))
            total_pages = len(reader.pages)
            pages_to_process = min(total_pages, max_pages) if max_pages else total_pages
        except Exception as e:
            print(f"[PDF Chunker] Không thể mở file {pdf_path.name}: {e}")
            return []

        chunk_counter = 0

        for page_idx in range(pages_to_process):
            page_num = page_idx + 1
            page = reader.pages[page_idx]

            # 1. Thử trích xuất văn bản kỹ thuật số thông thường
            digital_text = page.extract_text() or ""
            digital_text = digital_text.strip()
            is_scanned = len(digital_text) < 40  # Nếu dưới 40 ký tự thì trang là scan/ảnh

            page_content = ""
            chunk_type = "pdf_text"

            if not is_scanned:
                page_content = digital_text
            else:
                # 2. Trang scan -> Tự động kích hoạt OCR
                if self.ocr_extractor.is_available():
                    ocr_text = self.ocr_extractor.extract_text_from_pdf_page(str(pdf_path), page_idx=page_idx)
                    page_content = ocr_text.strip()
                    chunk_type = "pdf_ocr_scan"

            if not page_content:
                continue

            breadcrumb = f"{pdf_path.stem} > Trang {page_num}/{total_pages}"
            section_title = f"Trang {page_num}"

            # 3. Phân đoạn trang nếu nội dung quá dài
            if len(page_content) <= self.max_chunk_size:
                full_content = f"[{breadcrumb}]\n{page_content}"
                chunks.append({
                    "chunk_id": f"{pdf_path.stem}_p{page_num}_c{chunk_counter}",
                    "source": pdf_path.name,
                    "breadcrumb": breadcrumb,
                    "section_title": section_title,
                    "chunk_type": chunk_type,
                    "content": full_content,
                    "raw_content": page_content
                })
                chunk_counter += 1
            else:
                sub_parts = self._split_text(page_content, self.max_chunk_size, self.chunk_overlap)
                for part in sub_parts:
                    full_content = f"[{breadcrumb}]\n{part}"
                    chunks.append({
                        "chunk_id": f"{pdf_path.stem}_p{page_num}_c{chunk_counter}",
                        "source": pdf_path.name,
                        "breadcrumb": breadcrumb,
                        "section_title": section_title,
                        "chunk_type": chunk_type,
                        "content": full_content,
                        "raw_content": part
                    })
                    chunk_counter += 1

        return chunks

    def _split_text(self, text, max_size, overlap):
        """Tách văn bản dài theo câu hoặc xuống dòng"""
        sentences = re.split(r"(?<=[.!?\n])\s+", text)
        parts = []
        curr = ""

        for s in sentences:
            if not s.strip():
                continue
            if len(curr) + len(s) + 1 <= max_size:
                curr = f"{curr} {s}".strip() if curr else s
            else:
                if curr:
                    parts.append(curr)
                curr = s

        if curr:
            parts.append(curr)

        return parts or [text]
