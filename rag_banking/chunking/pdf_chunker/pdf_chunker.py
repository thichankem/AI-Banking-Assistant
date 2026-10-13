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
from __future__ import annotations

from pathlib import Path

import pymupdf as fitz
import pdfplumber

from .scan_structure_reconstructor import ScanStructureReconstructor
from .scan_detector import detect_is_scanned_page
from .vision_table_extractor import LocalVisionTableExtractor
from .ocr_extractor import OCRExtractor
try:
    from ..unified_chunk.canonical_resolver import (
        count_words,
        split_sentences_banking,
        group_sentences_banking,
        clean_and_dehyphenate_text,
        is_valid_vietnamese_text,
        detect_and_remove_headers_footers
    )
except (ImportError, ValueError):
    from rag_banking.chunking.unified_chunk.canonical_resolver import (
        count_words,
        split_sentences_banking,
        group_sentences_banking,
        clean_and_dehyphenate_text,
        is_valid_vietnamese_text,
        detect_and_remove_headers_footers
    )


class PDFChunker:
    """
    Bộ phân đoạn tài liệu PDF chuyên dụng cho ngân hàng theo giải thuật DLA & Advanced Table/OCR/Vision LLM.
    """

    def __init__(
        self,
        ocr_extractor: OCRExtractor | None = None,
        vision_table_extractor: LocalVisionTableExtractor | None = None,
        enable_vision_llm_tables: bool = False,
        max_chunk_size: int = 800,           # Trần kích thước ký tự (~400 từ)
        min_section_size: int = 120,         # Ngưỡng từ tối thiểu dải vàng (150-450 từ)
        chunk_overlap: int = 100,            # Độ gối đầu khi phải chia nhỏ đoạn dài
        image_area_ratio_threshold: float = 0.40, # Tỷ lệ diện tích ảnh kích hoạt scan
        table_rows_per_chunk: int = 4        # Số hàng bảng biểu tối đa trong một chunk
    ):
        self.ocr_extractor = ocr_extractor or OCRExtractor()
        self.vision_table_extractor = vision_table_extractor or LocalVisionTableExtractor()
        self.enable_vision_llm_tables = enable_vision_llm_tables
        self.max_chunk_size = max_chunk_size
        self.min_section_size = min_section_size
        self.chunk_overlap = chunk_overlap
        self.image_area_ratio_threshold = image_area_ratio_threshold
        self.table_rows_per_chunk = table_rows_per_chunk

    def chunk_pdf(
        self,
        pdf_path: str | Path,
        max_pages: int | None = None,
        enable_vision_tables: bool | None = None
    ) -> list[dict]:
        """
        Đọc và phân đoạn toàn diện một tệp PDF ngân hàng:
        Xử lý bảng biểu, trang scan mộc đỏ, loại bỏ header/footer, bảo toàn luồng văn bản liên tục qua trang.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            print(f"[PDF Chunker] Không tìm thấy tệp: {pdf_path}")
            return []


        try:
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)
            pages_to_process = min(total_pages, max_pages) if max_pages else total_pages
        except Exception as e:
            print(f"[PDF Chunker] Không thể mở tệp {pdf_path.name}: {e}")
            return []

        chunks: list[dict] = []
        raw_page_texts: list[str] = []
        page_info_list: list[dict] = []
        scan_pages_lines: dict[int, list[dict]] = {}

        use_vision = self.enable_vision_llm_tables if enable_vision_tables is None else enable_vision_tables

        # ---------------------------------------------------------------------
        # BƯỚC 1: TRÍCH XUẤT TỪNG TRANG (BẢNG BIỂU, VĂN BẢN SỐ & OCR SCAN)
        # ---------------------------------------------------------------------
        for page_idx in range(pages_to_process):
            page_num = page_idx + 1
            page = doc[page_idx]
            page_rect = page.rect
            page_area = max(page_rect.width * page_rect.height, 1.0)

            # 1.1. Trích xuất bảng biểu bằng PyMuPDF find_tables (hoặc Local Vision LLM)
            table_chunks = self._extract_page_tables(
                page=page,
                pdf_path=pdf_path,
                page_num=page_num,
                start_idx=len(chunks),
                use_vision=use_vision
            )
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
            if scan_chunks:
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
        page: object,
        pdf_path: Path,
        page_num: int,
        start_idx: int,
        use_vision: bool = False
    ) -> list[dict]:
        """Dò và trích xuất các bảng biểu trên trang thành pdf_table (PyMuPDF hoặc Local Vision LLM)."""
        table_chunks: list[dict] = []

        # 1. Trích xuất bảng số học bằng PyMuPDF find_tables
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

        # 2. Xử lý bảng phức tạp bằng Local Vision LLM (kèm cơ chế xếp hàng đợi an toàn)
        if not table_chunks and use_vision:
            try:
                page_text = page.get_text("text").strip()
                has_images = bool(page.get_images())
                if self.vision_table_extractor.is_available() and self.vision_table_extractor.is_complex_table_page(
                    page_text, len(table_chunks), has_images=has_images
                ):
                    v_res = self.vision_table_extractor.extract_table_from_page(
                        page=page,
                        doc_title=pdf_path.stem,
                        page_num=page_num
                    )
                    if v_res and v_res.get("markdown_table"):
                        md_table = v_res["markdown_table"]
                        breadcrumb = f"{pdf_path.stem} > Trang {page_num} > Bảng phức tạp Vision LLM"
                        full_content = f"[Breadcrumbs: {breadcrumb}]\n{md_table}"
                        table_chunks.append({
                            "chunk_id": f"{pdf_path.stem}_vision_tbl_{start_idx + len(table_chunks)}",
                            "source": pdf_path.name,
                            "chunk_type": "pdf_vision_table",
                            "breadcrumb": breadcrumb,
                            "breadcrumb_source": "vision_llm",
                            "breadcrumb_confidence": 0.98,
                            "page_start": page_num,
                            "page_end": page_num,
                            "page_numbers": [page_num],
                            "section_title": f"Bảng biểu Trang {page_num}",
                            "content": full_content,
                            "word_count": count_words(full_content)
                        })
            except Exception as e:
                print(f"[PDF Chunker] Lỗi bóc tách Vision LLM trang {page_num}: {e}")

        return table_chunks

    def _extract_page_text_or_scan(
        self,
        page: object,
        pdf_path: Path,
        page_idx: int,
        page_area: float
    ) -> tuple[str, str, float, list[dict] | None]:
        """Nhận diện trang scan hay văn bản số và trích xuất nội dung phù hợp."""
        # 1. Nhận diện trang scan bằng ScanDetector chuyên biệt
        needs_ocr, metrics = detect_is_scanned_page(
            page=page,
            image_area_ratio_threshold=self.image_area_ratio_threshold
        )
        digital_text = page.get_text("text").strip()

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
        cleaned_pages: list[str],
        page_info_list: list[dict]
    ) -> list[dict]:
        """Ghép các trang thành luồng khối đoạn văn toàn tài liệu liên tục."""
        document_blocks: list[dict] = []
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
        document_blocks: list[dict],
        pdf_path: Path,
        start_chunk_idx: int
    ) -> list[dict]:
        """Phân đoạn các khối văn bản số liên tục qua các trang."""
        chunks: list[dict] = []
        chunk_counter = start_chunk_idx
        active_headers: dict[int, str] = {}
        buffer_blocks: list[dict] = []

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
                    pages_involved=pages_involved,
                    content_text=part,
                    ocr_confidence=avg_ocr_conf
                )
                chunks.append(item)
                chunk_counter += 1

        for block in document_blocks:
            heading_level, heading_title = self._detect_block_heading(block["text"])
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
        if not chunks and document_blocks:
            for b in document_blocks:
                buffer_blocks.append(b)
            emit_buffer(force_flush=True)
        return chunks

    def _create_digital_chunk(
        self,
        chunk_idx: int,
        pdf_path: Path,
        chunk_type: str,
        breadcrumb: str,
        section_title: str,
        pages_involved: list[int],
        content_text: str,
        ocr_confidence: float
    ) -> dict:
        """Tạo đối tượng chunk digital hoàn chỉnh với metadata tinh gọn."""
        khoan_val = None
        diem_val = None
        for line in content_text.split("\n"):
            toks = line.strip().split()
            if toks:
                if toks[0].endswith(".") and toks[0][:-1].isdigit() and len(toks) > 1 and toks[1] and toks[1][0].isupper():
                    if khoan_val is None:
                        try:
                            khoan_val = int(toks[0][:-1])
                        except ValueError:
                            pass
                elif toks[0].endswith(")") and len(toks[0]) == 2 and toks[0][0].isalpha() and len(toks) > 1:
                    if diem_val is None:
                        diem_val = toks[0][0].lower()

        page_start = pages_involved[0] if pages_involved else 1
        page_end = pages_involved[-1] if pages_involved else 1
        has_structure_title = bool(breadcrumb and breadcrumb != pdf_path.stem and not breadcrumb.startswith("Trang "))
        b_src = "structure+layout" if has_structure_title else "page_fallback"

        header_prefix = f"[Breadcrumbs: {breadcrumb}]"
        full_content = f"{header_prefix}\n{content_text}"

        item = {
            "chunk_id": f"{pdf_path.stem}_pdf_c{chunk_idx}",
            "source": pdf_path.name,
            "chunk_type": chunk_type,
            "breadcrumb": breadcrumb,
            "breadcrumb_source": b_src,
            "breadcrumb_confidence": 1.0,
            "heading_ocr_confidence": 1.0,
            "page_start": page_start,
            "page_end": page_end,
            "page_numbers": pages_involved,
            "khoan": khoan_val,
            "diem": diem_val,
            "section_title": section_title,
            "content": full_content,
            "word_count": count_words(full_content)
        }
        if chunk_type == "pdf_ocr_scan":
            item["ocr_confidence"] = ocr_confidence

        return item

    def _detect_block_heading(self, text: str) -> tuple[int | None, str]:
        """Nhận diện tiêu đề phân cấp tài liệu tổng quát hoàn toàn dựa trên cấu trúc hình thức & outline (0% Rule-based Keywords)."""
        first_line = text.split("\n")[0].strip()
        tokens = first_line.split()
        if not tokens or len(first_line) > 120:
            return None, ""

        # 1. Ký hiệu số La Mã chuẩn quốc tế (I., II., III., IV., V., etc.)
        first_tok_clean = tokens[0].rstrip(".:)-")
        is_roman = first_tok_clean in ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII")
        if is_roman and len(tokens) >= 2:
            return 1, first_line

        # 2. Tiêu đề in hoa ngắn (standalone heading)
        if first_line.isupper() and 3 < len(first_line) < 100 and not first_line.endswith((".", ";", ":")):
            return 2, first_line

        # 3. Đánh số phân cấp đa tầng (1.1, 1.1.1, etc.)
        num_clean = tokens[0].rstrip(".")
        dot_count = num_clean.count(".")
        if dot_count >= 1 and all(part.isdigit() for part in num_clean.split(".")) and len(tokens) >= 2:
            return min(2 + dot_count, 5), first_line

        # 4. Đánh số mục đơn (1., 2.) kèm tiêu đề ngắn
        if tokens[0].endswith(".") and tokens[0][:-1].isdigit() and len(tokens) >= 2 and len(tokens) <= 15:
            if not first_line.endswith((".", ";", ":")):
                return 3, first_line

        return None, ""

    def _process_pdf_table(
        self,
        table_data: list[list[str | None]],
        pdf_path: Path,
        page_num: int,
        start_idx: int
    ) -> list[dict]:
        """Xử lý bảng PDF thành Linearized Chunks (pdf_table) kèm header cột."""
        if not table_data or len(table_data) < 2:
            return []

        raw_header = [str(c).strip().replace("\n", " ") if c is not None else "" for c in table_data[0]]
        header_names = [name if name else f"Cột_{i+1}" for i, name in enumerate(raw_header)]

        data_rows: list[str] = []

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

            if not data_rows:
                return []

        table_breadcrumb = f"{pdf_path.stem} > Trang {page_num} > Bảng biểu"
        chunks: list[dict] = []
        batch_size = self.table_rows_per_chunk

        for i in range(0, len(data_rows), batch_size):
            batch_data = data_rows[i:i + batch_size]

            content = (
                f"[Breadcrumbs: {table_breadcrumb}]\n"
                f"Cột dữ liệu: {', '.join(header_names)}\n"
                f"Chi tiết từng hàng:\n" + "\n".join(batch_data)
            )

            chunks.append({
                "chunk_id": f"{pdf_path.stem}_pdf_tbl_{start_idx + len(chunks)}",
                "source": pdf_path.name,
                "chunk_type": "pdf_table",
                "breadcrumb": table_breadcrumb,
                "section_title": f"Bảng biểu Trang {page_num}",
                "page_numbers": [page_num],
                "page_start": page_num,
                "page_end": page_num,
                "breadcrumb_source": "layout_visual",
                "breadcrumb_confidence": 1.0,
                "heading_ocr_confidence": 1.0,
                "content": content,
                "word_count": count_words(content)
            })

        return chunks
