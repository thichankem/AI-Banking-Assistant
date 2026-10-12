"""
PDF Layout-Aware & Advanced Table/OCR Chunker (DLA - Document Layout Analysis)
Bộ phân đoạn tài liệu PDF ngân hàng nhận thức bố cục, bảng biểu và OCR thế hệ mới:
1. Table Extraction (pdf_table): Tự động phát hiện bảng bằng PyMuPDF find_tables / pdfplumber,
   tuyến tính hóa từng hàng (* Hàng X: Cột A: Val A | Cột B: Val B), bảo toàn tính nguyên tử của bảng.
2. Smart Scan & Image Area Ratio: Nhận diện trang scan dựa trên tỷ lệ diện tích ảnh (image area ratio > 40%)
   kết hợp kiểm tra chất lượng font tiếng Việt (chống lỗi font VNI/TCVN3).
3. Scan Structure Reconstruction: Tích hợp ScanStructureReconstructor với render 300 DPI, Deskew,
   lọc mộc đỏ HSV, phân loại Heading 2 tín hiệu (Regex OCR + Bố cục) và State Machine BreadcrumbTracker.
4. Cross-Page Continuity (Luồng văn bản liên tục toàn tài liệu): Không cắt cứng theo trang;
   lưu vết page_start, page_end và danh sách page_numbers.
5. Legal Structure-First Chunking: Phân cấp theo Chương > Mục > Điều > Khoản, bảo vệ các thực thể tài chính.
6. Dehyphenation & Header/Footer Removal: Tự động loại bỏ header/footer lặp lại và hàn gắn dòng ngắt cứng.
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

try:
    import pymupdf as fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    try:
        import fitz
        PYMUPDF_AVAILABLE = True
    except ImportError:
        PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

from .ocr_extractor import OCRExtractor
from .scan_structure_reconstructor import ScanStructureReconstructor, KHOAN_REGEX, DIEM_REGEX
from .canonical_resolver import (
    count_words,
    count_syllables_vietnamese,
    extract_canonical_legal_ref,
    split_sentences_banking,
    group_sentences_banking,
    clean_and_dehyphenate_text,
    is_valid_vietnamese_text,
    detect_and_remove_headers_footers
)


class PDFChunker:
    """
    Bộ phân đoạn tài liệu PDF chuyên dụng cho ngân hàng theo giải thuật DLA & Advanced Table/OCR.
    """

    def __init__(
        self,
        ocr_extractor: Optional[OCRExtractor] = None,
        max_chunk_size: int = 800,           # Trần kích thước ký tự (~400 từ)
        min_section_size: int = 120,         # Ngưỡng từ tối thiểu dải vàng (150-450 từ)
        chunk_overlap: int = 100,            # Độ gối đầu khi phải chia nhỏ đoạn dài
        image_area_ratio_threshold: float = 0.40, # Tỷ lệ diện tích ảnh kích hoạt scan
        table_rows_per_chunk: int = 4        # Số hàng bảng biểu tối đa trong một chunk
    ):
        self.ocr_extractor = ocr_extractor or OCRExtractor()
        self.max_chunk_size = max_chunk_size
        self.min_section_size = min_section_size
        self.chunk_overlap = chunk_overlap
        self.image_area_ratio_threshold = image_area_ratio_threshold
        self.table_rows_per_chunk = table_rows_per_chunk

    def chunk_pdf(self, pdf_path: str | Path, max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Đọc và phân đoạn toàn diện một tệp PDF ngân hàng:
        Xử lý bảng biểu, trang scan mộc đỏ, loại bỏ header/footer, bảo toàn luồng văn bản liên tục qua trang.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            print(f"[PDF Chunker] Không tìm thấy tệp: {pdf_path}")
            return []

        if not PYMUPDF_AVAILABLE:
            raise ImportError("Vui lòng cài đặt thư viện 'pymupdf' để xử lý PDF nhận thức bố cục!")

        try:
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)
            pages_to_process = min(total_pages, max_pages) if max_pages else total_pages
        except Exception as e:
            print(f"[PDF Chunker] Không thể mở tệp {pdf_path.name}: {e}")
            return []

        chunks: List[Dict[str, Any]] = []
        raw_page_texts: List[str] = []
        page_info_list: List[Dict[str, Any]] = []
        scan_pages_lines: Dict[int, List[Dict[str, Any]]] = {}

        # ---------------------------------------------------------------------
        # BƯỚC 1: TRÍCH XUẤT TỪNG TRANG (BẢNG BIỂU, VĂN BẢN SỐ & OCR SCAN)
        # ---------------------------------------------------------------------
        for page_idx in range(pages_to_process):
            page_num = page_idx + 1
            page = doc[page_idx]
            page_rect = page.rect
            page_area = max(page_rect.width * page_rect.height, 1.0)

            # 1.1. Trích xuất bảng biểu bằng PyMuPDF find_tables
            table_chunks = self._extract_page_tables(page, pdf_path, page_num, start_idx=len(chunks))
            if table_chunks:
                chunks.extend(table_chunks)

            # 1.2. Nhận diện trang scan thông minh
            page_text, chunk_type, ocr_conf, layout_lines = self._extract_page_text_or_scan(
                page=page,
                pdf_path=pdf_path,
                page_idx=page_idx,
                page_area=page_area
            )

            if layout_lines:
                scan_pages_lines[page_num] = layout_lines

            raw_page_texts.append(page_text)
            page_info_list.append({
                "page_num": page_num,
                "chunk_type": chunk_type,
                "ocr_confidence": ocr_conf
            })

        # ---------------------------------------------------------------------
        # BƯỚC 2: PHỤC HỒI CẤU TRÚC SCAN NẾU LÀ TÀI LIỆU SCAN CHỦ ĐẠO
        # ---------------------------------------------------------------------
        if scan_pages_lines and (len(scan_pages_lines) >= pages_to_process * 0.5):
            reconstructor = ScanStructureReconstructor(doc_title=pdf_path.stem)
            ordered_scan_lines = [scan_pages_lines[p] for p in sorted(scan_pages_lines.keys())]
            scan_chunks = reconstructor.reconstruct_document_chunks(
                pages_lines=ordered_scan_lines,
                source_name=pdf_path.name,
                max_chunk_size=self.max_chunk_size,
                min_section_size=self.min_section_size
            )
            chunks.extend(scan_chunks)
            return chunks

        # ---------------------------------------------------------------------
        # BƯỚC 3 (DIGITAL PDF): LỌC HEADER/FOOTER & NỐI LUỒNG CROSS-PAGE LIÊN TỤC
        # ---------------------------------------------------------------------
        cleaned_pages = detect_and_remove_headers_footers(raw_page_texts)
        document_blocks = self._build_document_blocks(cleaned_pages, page_info_list)

        # ---------------------------------------------------------------------
        # BƯỚC 4: PHÂN ĐOẠN THEO THỨ BẬC PHÁP LÝ & CHỐNG PHÂN MẢNH VỤN
        # ---------------------------------------------------------------------
        digital_chunks = self._chunk_document_blocks(
            document_blocks=document_blocks,
            pdf_path=pdf_path,
            start_chunk_idx=len(chunks)
        )
        chunks.extend(digital_chunks)

        return chunks

    def _extract_page_tables(
        self,
        page: Any,
        pdf_path: Path,
        page_num: int,
        start_idx: int
    ) -> List[Dict[str, Any]]:
        """Dò và trích xuất các bảng biểu trên trang thành pdf_table."""
        table_chunks: List[Dict[str, Any]] = []
        try:
            tabs = page.find_tables()
            if tabs and tabs.tables:
                for tab in tabs.tables:
                    extracted = tab.extract()
                    if extracted and len(extracted) >= 2:
                        t_chunks = self._process_pdf_table(
                            table_data=extracted,
                            pdf_path=pdf_path,
                            page_num=page_num,
                            start_idx=start_idx + len(table_chunks)
                        )
                        table_chunks.extend(t_chunks)
        except Exception:
            pass
        return table_chunks

    def _extract_page_text_or_scan(
        self,
        page: Any,
        pdf_path: Path,
        page_idx: int,
        page_area: float
    ) -> Tuple[str, str, float, Optional[List[Dict[str, Any]]]]:
        """Nhận diện trang scan hay văn bản số và trích xuất nội dung phù hợp."""
        img_list = page.get_images()
        total_img_area = 0.0
        for img in img_list:
            try:
                for bbox in page.get_image_rects(img[0]):
                    total_img_area += (bbox.width * bbox.height)
            except Exception:
                pass
        img_area_ratio = min(total_img_area / page_area, 1.0)

        digital_text = page.get_text("text").strip()
        is_valid_vn = is_valid_vietnamese_text(digital_text)
        word_count = count_words(digital_text)

        # Tiêu chí nhận diện cần chạy OCR
        needs_ocr = (
            (img_area_ratio >= self.image_area_ratio_threshold and word_count < 30)
            or (word_count > 10 and not is_valid_vn)
            or (len(digital_text) < 20 and len(img_list) > 0)
        )

        if needs_ocr and self.ocr_extractor.is_available():
            img_np = self.ocr_extractor.render_pdf_page_300dpi(pdf_path, page_idx=page_idx)
            if img_np is not None:
                lines = self.ocr_extractor.extract_layout_lines(img_np, apply_deskew=True, apply_red_stamp_filter=True)
                if lines:
                    text_out = "\n".join(l["text"] for l in lines)
                    conf = round(sum(l["score"] for l in lines) / len(lines), 3)
                    return text_out, "pdf_ocr_scan", conf, lines

            # Fallback
            ocr_out, conf = self.ocr_extractor.extract_text_from_pdf_page(str(pdf_path), page_idx=page_idx)
            if ocr_out.strip():
                return ocr_out.strip(), "pdf_ocr_scan", conf, None

        return digital_text, "pdf_text", 1.0, None

    def _build_document_blocks(
        self,
        cleaned_pages: List[str],
        page_info_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Ghép các trang thành luồng khối đoạn văn toàn tài liệu liên tục."""
        document_blocks: List[Dict[str, Any]] = []
        for p_idx, p_text in enumerate(cleaned_pages):
            p_clean = clean_and_dehyphenate_text(p_text)
            if not p_clean:
                continue

            info = page_info_list[p_idx]
            for para in p_clean.split("\n\n"):
                para_s = para.strip()
                if para_s:
                    document_blocks.append({
                        "text": para_s,
                        "page_num": info["page_num"],
                        "chunk_type": info["chunk_type"],
                        "ocr_confidence": info["ocr_confidence"]
                    })
        return document_blocks

    def _chunk_document_blocks(
        self,
        document_blocks: List[Dict[str, Any]],
        pdf_path: Path,
        start_chunk_idx: int
    ) -> List[Dict[str, Any]]:
        """Phân đoạn các khối văn bản số liên tục qua các trang."""
        chunks: List[Dict[str, Any]] = []
        chunk_counter = start_chunk_idx
        active_headers: Dict[int, str] = {}
        buffer_blocks: List[Dict[str, Any]] = []

        def get_breadcrumb() -> str:
            sorted_levels = sorted(active_headers.keys())
            titles = [active_headers[lvl] for lvl in sorted_levels if active_headers[lvl]]
            return " > ".join(titles) if titles else pdf_path.stem

        def get_section_title() -> str:
            if not active_headers:
                return "Tổng quan"
            return active_headers[max(active_headers.keys())]

        def emit_buffer(force_flush: bool = False) -> None:
            nonlocal chunk_counter
            if not buffer_blocks:
                return

            combined_text = "\n\n".join(b["text"] for b in buffer_blocks).strip()
            total_words = count_words(combined_text)

            if not force_flush and total_words < self.min_section_size:
                return

            pages_involved = sorted(list(set(b["page_num"] for b in buffer_blocks)))
            chunk_type = "pdf_ocr_scan" if any(b["chunk_type"] == "pdf_ocr_scan" for b in buffer_blocks) else "pdf_text"
            confs = [b["ocr_confidence"] for b in buffer_blocks if b["chunk_type"] == "pdf_ocr_scan"]
            avg_ocr_conf = round(sum(confs) / len(confs), 3) if confs else 1.0

            buffer_blocks.clear()
            if not combined_text:
                return

            breadcrumb = get_breadcrumb()
            section_title = get_section_title()
            legal_ref = extract_canonical_legal_ref(breadcrumb, combined_text, source_name=pdf_path.name)

            # Cắt nhỏ nếu dài hơn max_chunk_size
            if len(combined_text) <= self.max_chunk_size:
                sub_parts = [combined_text]
            else:
                sentences = split_sentences_banking(combined_text)
                sub_parts = group_sentences_banking(sentences, self.max_chunk_size, self.chunk_overlap)

            for part in sub_parts:
                item = self._create_digital_chunk(
                    chunk_idx=chunk_counter,
                    pdf_path=pdf_path,
                    chunk_type=chunk_type,
                    breadcrumb=breadcrumb,
                    section_title=section_title,
                    legal_ref=legal_ref,
                    pages_involved=pages_involved,
                    content_text=part,
                    ocr_confidence=avg_ocr_conf
                )
                chunks.append(item)
                chunk_counter += 1

        for block in document_blocks:
            heading_level, heading_title = self._detect_legal_heading(block["text"])
            if heading_level is not None:
                if heading_level == 1:
                    emit_buffer(force_flush=True)
                else:
                    curr_words = count_words("\n".join(b["text"] for b in buffer_blocks))
                    if curr_words >= self.min_section_size:
                        emit_buffer(force_flush=True)
                    else:
                        if buffer_blocks:
                            buffer_blocks.append({
                                "text": f"### {heading_title}",
                                "page_num": block["page_num"],
                                "chunk_type": block["chunk_type"],
                                "ocr_confidence": block["ocr_confidence"]
                            })

                active_headers[heading_level] = heading_title
                for lvl in [l for l in list(active_headers.keys()) if l > heading_level]:
                    del active_headers[lvl]
                continue

            buffer_blocks.append(block)
            if sum(len(b["text"]) for b in buffer_blocks) >= self.max_chunk_size:
                emit_buffer(force_flush=True)

        emit_buffer(force_flush=True)
        return chunks

    def _create_digital_chunk(
        self,
        chunk_idx: int,
        pdf_path: Path,
        chunk_type: str,
        breadcrumb: str,
        section_title: str,
        legal_ref: str,
        pages_involved: List[int],
        content_text: str,
        ocr_confidence: float
    ) -> Dict[str, Any]:
        """Tạo đối tượng chunk digital hoàn chỉnh với metadata đồng nhất."""
        m_khoan = KHOAN_REGEX.search(content_text)
        khoan_val = int(m_khoan.group(1)) if m_khoan else None

        m_diem = DIEM_REGEX.search(content_text)
        diem_val = m_diem.group(1) if m_diem else None

        page_start = pages_involved[0] if pages_involved else 1
        page_end = pages_involved[-1] if pages_involved else 1
        b_src = "regex+layout" if ("Điều" in breadcrumb or "Chương" in breadcrumb) else "page_fallback"

        header_prefix = f"[Breadcrumbs: {breadcrumb}]"
        sub_legal = extract_canonical_legal_ref(breadcrumb, content_text, source_name=pdf_path.name) or legal_ref
        if sub_legal:
            header_prefix += f"\n[Pháp lý: {sub_legal}]"

        full_content = f"{header_prefix}\n{content_text}"

        item = {
            "chunk_id": f"{pdf_path.stem}_pdf_c{chunk_idx}",
            "source": pdf_path.name,
            "format": "pdf",
            "chunk_type": chunk_type,
            "breadcrumb": breadcrumb,
            "breadcrumb_source": b_src,
            "breadcrumb_confidence": 1.0,
            "heading_ocr_confidence": 1.0,
            "page_start": page_start,
            "page_end": page_end,
            "page_numbers": pages_involved,
            "legal_ref": sub_legal,
            "khoan": khoan_val,
            "diem": diem_val,
            "section_title": section_title,
            "content": full_content,
            "raw_content": content_text,
            "char_length": len(full_content),
            "word_count": count_words(full_content),
            "syllable_count": count_syllables_vietnamese(full_content)
        }
        if chunk_type == "pdf_ocr_scan":
            item["ocr_confidence"] = ocr_confidence

        return item

    def _detect_legal_heading(self, text: str) -> Tuple[Optional[int], str]:
        """Nhận diện tiêu đề phân cấp pháp lý trong PDF chuẩn thể thức quốc gia."""
        first_line = text.split("\n")[0].strip()
        if len(first_line) > 120:
            return None, ""

        if re.match(r"^(PHẦN|CHƯƠNG)\s+[IVXLCDM\d]+", first_line, re.IGNORECASE):
            return 1, first_line
        if re.match(r"^MỤC\s+[IVXLCDM\d]+", first_line, re.IGNORECASE):
            return 2, first_line
        if re.match(r"^Điều\s+\d+", first_line, re.IGNORECASE):
            return 3, first_line
        if re.match(r"^(BIỂU PHÍ|QUY TẮC|THỎA THUẬN|BẢNG DỮ LIỆU)", first_line, re.IGNORECASE):
            return 2, first_line
        if re.match(r"^Câu\s+\d+[:.]", first_line, re.IGNORECASE):
            return 3, first_line
        if re.match(r"^Khoản\s+\d+", first_line, re.IGNORECASE):
            return 4, first_line

        return None, ""

    def _process_pdf_table(
        self,
        table_data: List[List[Optional[str]]],
        pdf_path: Path,
        page_num: int,
        start_idx: int
    ) -> List[Dict[str, Any]]:
        """Xử lý bảng PDF thành Linearized Chunks (pdf_table) kèm header cột."""
        if not table_data or len(table_data) < 2:
            return []

        raw_header = [str(c).strip().replace("\n", " ") if c is not None else "" for c in table_data[0]]
        header_names = [name if name else f"Cột_{i+1}" for i, name in enumerate(raw_header)]

        data_rows: List[str] = []
        raw_rows: List[str] = []

        for r_idx, row in enumerate(table_data[1:], start=1):
            cleaned_cells = [str(c).strip().replace("\n", " ") if c is not None else "" for c in row]
            if not any(cleaned_cells):
                continue

            row_items = []
            for c_idx, val in enumerate(cleaned_cells):
                col_name = header_names[c_idx] if c_idx < len(header_names) else f"Cột_{c_idx+1}"
                row_items.append(f"{col_name}: {val}")

            row_str = " | ".join(row_items)
            data_rows.append(f"* Hàng {r_idx}: {row_str}")
            raw_rows.append(row_str)

        if not data_rows:
            return []

        table_breadcrumb = f"{pdf_path.stem} > Trang {page_num} > Bảng biểu"
        chunks: List[Dict[str, Any]] = []
        batch_size = self.table_rows_per_chunk

        for i in range(0, len(data_rows), batch_size):
            batch_data = data_rows[i:i + batch_size]
            batch_raw = raw_rows[i:i + batch_size]

            content = (
                f"[Breadcrumbs: {table_breadcrumb}]\n"
                f"Cột dữ liệu: {', '.join(header_names)}\n"
                f"Chi tiết từng hàng:\n" + "\n".join(batch_data)
            )
            raw_content = "\n".join(batch_raw)
            legal_ref = extract_canonical_legal_ref(table_breadcrumb, raw_content, source_name=pdf_path.name)
            if legal_ref:
                content = f"[Pháp lý: {legal_ref}]\n{content}"

            chunks.append({
                "chunk_id": f"{pdf_path.stem}_pdf_tbl_{start_idx + len(chunks)}",
                "source": pdf_path.name,
                "format": "pdf",
                "chunk_type": "pdf_table",
                "breadcrumb": table_breadcrumb,
                "legal_ref": legal_ref,
                "section_title": f"Bảng biểu Trang {page_num}",
                "page_numbers": [page_num],
                "page_start": page_num,
                "page_end": page_num,
                "breadcrumb_source": "layout_visual",
                "breadcrumb_confidence": 1.0,
                "heading_ocr_confidence": 1.0,
                "content": content,
                "raw_content": raw_content,
                "char_length": len(content),
                "word_count": count_words(content),
                "syllable_count": count_syllables_vietnamese(content)
            })

        return chunks
