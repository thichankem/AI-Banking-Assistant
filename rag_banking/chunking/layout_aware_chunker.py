"""
Document Layout-Aware Chunking (Table / Header Extraction)
Giải thuật phân đoạn tài liệu ngân hàng có nhận thức bố cục:
1. Header Hierarchy Extraction: Theo dõi thứ bậc tiêu đề (Breadcrumbs H1 > H2 > H3)
2. Table Extraction: Nhận diện và trích xuất nguyên vẹn bảng biểu (Markdown pipe table & Structured bullet table)
3. Text Contextualization: Gắn ngữ cảnh tiêu đề vào từng đoạn nội dung để tăng độ chính xác truy xuất.
"""

import re
from pathlib import Path
from .ocr_extractor import OCRExtractor
from .pdf_chunker import PDFChunker
from .excel_chunker import ExcelChunker

class LayoutAwareChunker:
    def __init__(self, max_chunk_size=800, chunk_overlap=100):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.ocr_extractor = OCRExtractor()
        self.pdf_chunker = PDFChunker(
            ocr_extractor=self.ocr_extractor,
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.excel_chunker = ExcelChunker()

    def chunk_document(self, text, source_name="document.md"):
        """
        Phân đoạn một tài liệu theo cấu trúc bố cục (Header & Table aware).
        """
        lines = text.split("\n")
        chunks = []

        # Lưu trữ trạng thái tiêu đề theo phân cấp: {1: "Tiêu đề 1", 2: "Tiêu đề 2", ...}
        active_headers = {}
        
        # Buffer cho nội dung văn bản thông thường
        current_text_lines = []
        chunk_counter = 0

        def get_current_breadcrumb():
            """Trả về chuỗi đường dẫn thứ bậc tiêu đề: H1 > H2 > H3"""
            sorted_levels = sorted(active_headers.keys())
            titles = [active_headers[lvl] for lvl in sorted_levels if active_headers[lvl]]
            return " > ".join(titles)

        def get_current_section_title():
            """Trả về tiêu đề gần nhất"""
            if not active_headers:
                return "Tổng quan"
            max_lvl = max(active_headers.keys())
            return active_headers[max_lvl]

        def flush_text_buffer():
            """Lưu trữ các dòng văn bản thông thường tích lũy trong buffer thành chunk"""
            nonlocal chunk_counter
            if not current_text_lines:
                return
            
            raw_text = "\n".join(current_text_lines).strip()
            current_text_lines.clear()

            if not raw_text:
                return

            breadcrumb = get_current_breadcrumb()
            section = get_current_section_title()

            # Nếu văn bản ngắn hơn max_chunk_size, tạo 1 chunk đơn giản
            if len(raw_text) <= self.max_chunk_size:
                full_content = f"[{breadcrumb}]\n{raw_text}" if breadcrumb else raw_text
                chunks.append({
                    "chunk_id": f"{Path(source_name).stem}_chunk_{chunk_counter}",
                    "source": source_name,
                    "breadcrumb": breadcrumb,
                    "section_title": section,
                    "chunk_type": "text",
                    "content": full_content,
                    "raw_content": raw_text
                })
                chunk_counter += 1
            else:
                # Nếu văn bản dài, chia nhỏ theo câu/đoạn nhưng luôn giữ tiền tố breadcrumb
                sub_parts = self._split_text_into_subparts(raw_text, self.max_chunk_size, self.chunk_overlap)
                for part in sub_parts:
                    full_content = f"[{breadcrumb}]\n{part}" if breadcrumb else part
                    chunks.append({
                        "chunk_id": f"{Path(source_name).stem}_chunk_{chunk_counter}",
                        "source": source_name,
                        "breadcrumb": breadcrumb,
                        "section_title": section,
                        "chunk_type": "text",
                        "content": full_content,
                        "raw_content": part
                    })
                    chunk_counter += 1

        i = 0
        total_lines = len(lines)

        while i < total_lines:
            line = lines[i]
            stripped = line.strip()

            # 1. KIỂM TRA HEADER MARKDOWN (# H1, ## H2, ### H3, ...)
            header_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
            if header_match:
                # Đẩy văn bản trước header vào chunk
                flush_text_buffer()

                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                # Cập nhật thứ bậc: Xóa bỏ các cấp con bên dưới level hiện tại
                active_headers[level] = title
                levels_to_remove = [lvl for lvl in active_headers.keys() if lvl > level]
                for lvl in levels_to_remove:
                    del active_headers[lvl]

                i += 1
                continue

            # 2. KIỂM TRA BẢNG BIỂU MARKDOWN CHUẨN (| Cột 1 | Cột 2 | ...)
            if stripped.startswith("|") and stripped.endswith("|") and i + 1 < total_lines:
                next_stripped = lines[i + 1].strip()
                # Kiểm tra dòng tiếp theo có phải dòng kẻ |---|---| không
                if re.match(r"^\|(\s*:?-+:?\s*\|)+$", next_stripped):
                    flush_text_buffer()

                    table_lines = [line]
                    i += 1
                    while i < total_lines and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                        table_lines.append(lines[i])
                        i += 1

                    # Trích xuất bảng và tạo chunk bảng nguyên vẹn
                    table_chunk = self._extract_markdown_table(
                        table_lines=table_lines,
                        source_name=source_name,
                        breadcrumb=get_current_breadcrumb(),
                        section=get_current_section_title(),
                        chunk_id=f"{Path(source_name).stem}_chunk_{chunk_counter}"
                    )
                    chunks.append(table_chunk)
                    chunk_counter += 1
                    continue

            # 3. KIỂM TRA BẢNG DẠNG STRUCTURED BULLETS (* **Mục:** \n  - **Thuộc tính:** ...)
            structured_match = re.match(r"^\*\s+\*\*([^*]+)\*\*[:\s]*(.*)$", stripped)
            if structured_match:
                # Kiểm tra các dòng sau có phải là thuộc tính con "- **Thuộc tính:**" không
                is_structured_table_item = False
                j = i + 1
                while j < total_lines and (lines[j].strip() == "" or lines[j].strip().startswith("- **") or lines[j].strip().startswith("- ")):
                    if lines[j].strip().startswith("- **"):
                        is_structured_table_item = True
                        break
                    j += 1
                
                if is_structured_table_item:
                    flush_text_buffer()
                    item_lines = [line]
                    i += 1
                    while i < total_lines:
                        nxt = lines[i]
                        nxt_strip = nxt.strip()
                        # Dừng khi gặp header, item mới, hoặc đường phân cách
                        if nxt_strip.startswith("#") or nxt_strip.startswith("---") or re.match(r"^\*\s+\*\*", nxt_strip):
                            break
                        if nxt_strip.startswith("- **") or nxt_strip.startswith("- ") or nxt_strip == "" or nxt.startswith("   "):
                            item_lines.append(nxt)
                            i += 1
                        else:
                            break

                    item_text = "\n".join(item_lines).strip()
                    breadcrumb = get_current_breadcrumb()
                    section = get_current_section_title()
                    full_content = f"[{breadcrumb} > Bảng tra cứu]\n{item_text}" if breadcrumb else item_text

                    chunks.append({
                        "chunk_id": f"{Path(source_name).stem}_chunk_{chunk_counter}",
                        "source": source_name,
                        "breadcrumb": f"{breadcrumb} > Bảng tra cứu" if breadcrumb else "Bảng tra cứu",
                        "section_title": section,
                        "chunk_type": "table_item",
                        "content": full_content,
                        "raw_content": item_text
                    })
                    chunk_counter += 1
                    continue

            # 4. CÁC DÒNG VĂN BẢN THÔNG THƯỜNG
            if stripped == "---":
                # Dấu gạch ngang phân cách trang/phần
                flush_text_buffer()
            else:
                current_text_lines.append(line)

            i += 1

        # Lưu phần văn bản còn lại trong buffer
        flush_text_buffer()

        return chunks

    def _extract_markdown_table(self, table_lines, source_name, breadcrumb, section, chunk_id):
        """
        Trích xuất và làm phẳng bảng biểu Markdown thành văn bản ngữ cảnh rõ ràng.
        """
        raw_table = "\n".join(table_lines)
        
        # Parse tiêu đề cột
        header_cells = [c.strip() for c in table_lines[0].strip("|").split("|")]
        
        # Gom các hàng dữ liệu (bỏ dòng gạch ngang ở index 1)
        data_rows = []
        for line in table_lines[2:]:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if any(cells):
                row_str = " | ".join([f"{header_cells[idx]}: {cells[idx]}" for idx in range(min(len(header_cells), len(cells)))])
                data_rows.append(row_str)

        linearized_content = "\n".join(data_rows)
        full_content = (
            f"[{breadcrumb} > Bảng biểu dữ liệu]\n"
            f"Cột: {', '.join(header_cells)}\n"
            f"Chi tiết từng hàng:\n{linearized_content}"
        ) if breadcrumb else f"Bảng biểu dữ liệu:\n{linearized_content}"

        return {
            "chunk_id": chunk_id,
            "source": source_name,
            "breadcrumb": f"{breadcrumb} > Bảng biểu" if breadcrumb else "Bảng biểu",
            "section_title": section,
            "chunk_type": "table",
            "content": full_content,
            "raw_content": raw_table
        }

    def _split_text_into_subparts(self, text, max_size, overlap):
        """Chia văn bản dài thành các đoạn nhỏ hơn theo ranh giới câu"""
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
                # Giữ overlap từ cuối đoạn trước nếu cần
                curr = s

        if curr:
            parts.append(curr)

        return parts or [text]

    def chunk_file(self, file_path, max_pdf_pages=None):
        """
        Phân đoạn một tệp bất kỳ dựa theo định dạng:
        - .md / .txt -> Layout-aware Markdown chunking
        - .pdf -> PDF chunking kết hợp tự động OCR cho trang scan
        - .xlsx -> Bảng biểu Excel đa sheet
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix in [".md", ".txt"]:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return self.chunk_document(content, source_name=file_path.name)
        elif suffix == ".pdf":
            return self.pdf_chunker.chunk_pdf(file_path, max_pages=max_pdf_pages)
        elif suffix in [".xlsx", ".xls"]:
            return self.excel_chunker.chunk_excel(file_path)
        else:
            return []

    def chunk_directory(self, dir_path, max_files=None, filenames=None, max_pdf_pages=None):
        """
        Đọc và chunking toàn bộ các tệp trong thư mục (hỗ trợ .md, .pdf, .xlsx).
        Tìm kiếm cả trong các thư mục con (vd: pdf/, md/, excel/).
        """
        dir_path = Path(dir_path)
        all_chunks = []
        files = []

        if filenames:
            for name in filenames:
                # Tìm trong thư mục hiện tại hoặc các thư mục con
                matched = list(dir_path.glob(f"**/{name}"))
                if matched:
                    files.append(matched[0])
                elif (dir_path / name).exists():
                    files.append(dir_path / name)
        else:
            # Tự động gom các tệp .md, .pdf, .xlsx
            for ext in ["*.md", "*.pdf", "*.xlsx"]:
                files.extend(list(dir_path.glob(f"**/{ext}")))
            if max_files:
                files = files[:max_files]

        for file_path in files:
            try:
                file_chunks = self.chunk_file(file_path, max_pdf_pages=max_pdf_pages)
                all_chunks.extend(file_chunks)
            except Exception as e:
                print(f"[Cảnh báo] Lỗi khi xử lý file {file_path.name}: {e}")

        return all_chunks
