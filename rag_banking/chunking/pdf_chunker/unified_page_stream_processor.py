"""
Module Unified Page Stream Processor (Thống nhất Luồng Xử lý PDF Lai Scan & Digital)
====================================================================================
Giải quyết bài toán: Loại bỏ ngưỡng cứng "Tỷ lệ trang scan >= 50%" gây đứt gãy
tài liệu lai (Hybrid PDF: vài trang scan có dấu mộc đỏ, các trang sau là văn bản số).

Kiến trúc thống nhất:
1. Page-Level Routing:
   - Từng trang được phân loại độc lập bằng `detect_is_scanned_page`.
   - Trang scan -> Xử lý ảnh 300 DPI, lọc mộc đỏ HSV, OCR dòng.
   - Trang digital -> Trích xuất text block không gian số học PyMuPDF (`Natural Spatial Reading Order`).
2. Chuẩn hóa Hệ quy chiếu Hình học 5 Chiều (5D Geometry Normalization):
   - Đưa cả dòng Digital lẫn dòng Scan về cùng hệ trục tọa độ chuẩn:
     `rel_h` (chiều cao tương đối), `gap_before` (giãn dòng trước), `is_centered` (căn giữa),
     `left_indent` (độ thụt lề), `rel_w` (tỷ lệ bề ngang dòng).
3. Luồng trạng thái liên tục (Single State Machine & Cross-Page Continuity):
   - Nuôi duy nhất MỘT `BreadcrumbTracker` và MỘT bộ phân loại cấp bậc xuyên suốt
     từ trang 1 đến trang cuối mà không bị chia cắt hay rẽ nhánh 50%.
   - Bảo toàn luồng văn bản liên tục qua trang (`page_start`, `page_end`, `page_numbers`).

TUÂN THỦ 100% SCHEMA CONTRACT:
- `chunk_id`, `source`, `chunk_type`, `breadcrumb`, `section_title`, `content`, `word_count`.
- Tuyệt đối không sinh trường rác ('ref', 'legal_ref', 'raw_content', 'char_length', 'format').
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf as fitz

try:
    from rag_banking.chunking.pdf_chunker.scan_detector import detect_is_scanned_page
    from rag_banking.chunking.pdf_chunker.ocr_extractor import OCRExtractor
    from rag_banking.chunking.pdf_chunker.ocr_layout_predictor import (
        OCRLayoutHierarchyPredictor,
        LayoutHierarchyLevel
    )
    from rag_banking.chunking.pdf_chunker.scan_structure_reconstructor import BreadcrumbTracker
    from rag_banking.chunking.pdf_chunker.pdf_chunker import (
        extract_document_title_dla,
        clean_filename_fallback
    )
    from rag_banking.chunking.unified_chunk.canonical_resolver import (
        count_words,
        clean_and_dehyphenate_text,
        is_valid_vietnamese_text
    )
except (ImportError, ValueError):
    from .scan_detector import detect_is_scanned_page
    from .ocr_extractor import OCRExtractor
    from .ocr_layout_predictor import (
        OCRLayoutHierarchyPredictor,
        LayoutHierarchyLevel
    )
    from .scan_structure_reconstructor import BreadcrumbTracker
    from .pdf_chunker import (
        extract_document_title_dla,
        clean_filename_fallback
    )
    from ..unified_chunk.canonical_resolver import (
        count_words,
        clean_and_dehyphenate_text,
        is_valid_vietnamese_text
    )

logger = logging.getLogger("unified_page_stream_processor")


@dataclass
class UnifiedGeometryLine:
    """Dòng văn bản chuẩn hóa về cùng hệ quy chiếu hình học 5 chiều (dùng chung Scan + Digital)."""
    text: str
    page_num: int
    page_mode: str  # 'scanned' hoặc 'digital'
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    score: float = 1.0
    page_width: float = 800.0
    page_height: float = 1100.0

    # 5D Normalized Features
    rel_h: float = 1.0
    gap_before: float = 8.0
    is_centered: bool = False
    left_indent: float = 0.0
    rel_w: float = 0.5

    # Cấp bậc dự đoán
    predicted_level: str = LayoutHierarchyLevel.BODY
    is_heading: bool = False
    indent_level: int = 0

    @property
    def width(self) -> float:
        return max(self.x_max - self.x_min, 1.0)

    @property
    def height(self) -> float:
        return max(self.y_max - self.y_min, 1.0)

    @property
    def center_x(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y_min + self.y_max) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "page_num": self.page_num,
            "page_mode": self.page_mode,
            "score": round(self.score, 3),
            "x_min": round(self.x_min, 1),
            "y_min": round(self.y_min, 1),
            "x_max": round(self.x_max, 1),
            "y_max": round(self.y_max, 1),
            "width": round(self.width, 1),
            "height": round(self.height, 1),
            "center_x": round(self.center_x, 1),
            "center_y": round(self.center_y, 1),
            "page_width": round(self.page_width, 1),
            "page_height": round(self.page_height, 1),
            "rel_h": round(self.rel_h, 2),
            "gap_before": round(self.gap_before, 1),
            "is_centered": self.is_centered,
            "left_indent": round(self.left_indent, 1),
            "rel_w": round(self.rel_w, 2),
            "predicted_level": self.predicted_level,
            "is_heading": self.is_heading,
            "indent_level": self.indent_level
        }


class UnifiedPageStreamProcessor:
    """
    Bộ xử lý luồng trang thống nhất cho tài liệu Hybrid PDF (kết hợp cả trang scan và digital).
    """

    def __init__(
        self,
        ocr_extractor: Any | None = None,
        max_chunk_size: int = 800,
        min_section_size: int = 120,
        chunk_overlap: int = 100,
        image_area_ratio_threshold: float = 0.40
    ):
        self.ocr_extractor = ocr_extractor or OCRExtractor()
        self.max_chunk_size = max_chunk_size
        self.min_section_size = min_section_size
        self.chunk_overlap = chunk_overlap
        self.image_area_ratio_threshold = image_area_ratio_threshold

        self.layout_predictor = OCRLayoutHierarchyPredictor()

    def extract_digital_lines(
        self,
        page: fitz.Page,
        page_num: int,
        table_bboxes: list[tuple[float, float, float, float]] | None = None
    ) -> list[UnifiedGeometryLine]:
        """
        Trích xuất các dòng văn bản từ trang Digital bằng PyMuPDF,
        nhóm theo dòng không gian và gắn tọa độ hình học thực tế.
        """
        page_rect = page.rect
        pw = float(page_rect.width)
        ph = float(page_rect.height)

        words = page.get_text("words")
        if not words:
            return []

        # words: [x0, y0, x1, y1, word, block_no, line_no, word_no]
        lines_map: dict[tuple[int, int], list[tuple[float, float, float, float, str]]] = {}
        for w_item in words:
            if len(w_item) < 5:
                continue
            x0, y0, x1, y1, w_text = float(w_item[0]), float(w_item[1]), float(w_item[2]), float(w_item[3]), str(w_item[4])
            block_no = int(w_item[5])
            line_no = int(w_item[6])

            # Loại trừ vùng bảng đã bóc tách
            if table_bboxes:
                in_tbl = False
                for tx0, ty0, tx1, ty1 in table_bboxes:
                    if tx0 <= x0 <= tx1 and ty0 <= y0 <= ty1:
                        in_tbl = True
                        break
                if in_tbl:
                    continue

            # Bỏ qua số trang đơn lẻ sát mép cực cận
            if (y1 > ph * 0.955 or y0 < ph * 0.045) and len(w_text) <= 4 and w_text.strip().isdigit():
                continue

            key = (block_no, line_no)
            if key not in lines_map:
                lines_map[key] = []
            lines_map[key].append((x0, y0, x1, y1, w_text))

        unified_lines: list[UnifiedGeometryLine] = []
        for (b_no, l_no), word_list in lines_map.items():
            if not word_list:
                continue
            line_text = " ".join(w[4] for w in word_list).strip()
            if not line_text:
                continue

            x_min = min(w[0] for w in word_list)
            y_min = min(w[1] for w in word_list)
            x_max = max(w[2] for w in word_list)
            y_max = max(w[3] for w in word_list)

            unified_lines.append(UnifiedGeometryLine(
                text=line_text,
                page_num=page_num,
                page_mode="digital",
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                score=1.0,
                page_width=pw,
                page_height=ph
            ))

        # Sắp xếp Natural Spatial Reading Order
        unified_lines.sort(key=lambda l: (round(l.y_min / 6.0) * 6.0, l.x_min))
        return unified_lines

    def extract_scanned_lines(
        self,
        pdf_path: str | Path,
        page_idx: int,
        page_w_pt: float,
        page_h_pt: float,
        table_bboxes: list[tuple[float, float, float, float]] | None = None
    ) -> list[UnifiedGeometryLine]:
        """
        Trích xuất các dòng văn bản từ trang Scan thông qua OCR Extractor.
        Ánh xạ tọa độ từ pixel (300 DPI) sang point (72 DPI) để đồng nhất với Digital.
        """
        page_num = page_idx + 1
        img_np = self.ocr_extractor.render_pdf_page_300dpi(pdf_path, page_idx=page_idx)
        if img_np is None:
            return []

        # OCR trích xuất layout lines
        raw_lines = self.ocr_extractor.extract_layout_lines(
            img_np,
            apply_deskew=True,
            apply_red_stamp_filter=True
        )
        if not raw_lines:
            return []

        img_h, img_w = img_np.shape[:2]
        scale_x = page_w_pt / float(img_w)
        scale_y = page_h_pt / float(img_h)

        unified_lines: list[UnifiedGeometryLine] = []
        for l in raw_lines:
            t = l.get("text", "").strip()
            if not t:
                continue

            x0 = float(l.get("x_min", 0.0)) * scale_x
            y0 = float(l.get("y_min", 0.0)) * scale_y
            x1 = float(l.get("x_max", 0.0)) * scale_x
            y1 = float(l.get("y_max", 0.0)) * scale_y
            score = float(l.get("score", 0.85))

            # Loại trừ nếu rơi vào vùng bảng scan đã bóc tách
            if table_bboxes:
                in_tbl = False
                for tx0, ty0, tx1, ty1 in table_bboxes:
                    if tx0 <= x0 <= tx1 and ty0 <= y0 <= ty1:
                        in_tbl = True
                        break
                if in_tbl:
                    continue

            unified_lines.append(UnifiedGeometryLine(
                text=t,
                page_num=page_num,
                page_mode="scanned",
                x_min=x0,
                y_min=y0,
                x_max=x1,
                y_max=y1,
                score=score,
                page_width=page_w_pt,
                page_height=page_h_pt
            ))

        return unified_lines

    def normalize_5d_geometry_for_page(
        self,
        lines: list[UnifiedGeometryLine],
        page_w: float,
        page_h: float
    ) -> list[UnifiedGeometryLine]:
        """
        Tính toán các đặc trưng hình học 5D chuẩn hóa cho toàn bộ dòng trong trang.
        """
        if not lines:
            return []

        # 1. Tính toán baseline stats (median_h, base_left, median_gap)
        line_dicts = [l.to_dict() for l in lines]
        stats = self.layout_predictor.compute_page_layout_stats(line_dicts, page_w, page_h)

        median_h = stats.get("median_h", 16.0)
        base_left = stats.get("base_left", 50.0)
        median_gap = stats.get("median_gap", 8.0)

        # 2. Gán đặc trưng 5D và phân loại bằng layout_predictor
        for i, line in enumerate(lines):
            prev_line = lines[i - 1].to_dict() if i > 0 else None
            next_line = lines[i + 1].to_dict() if i + 1 < len(lines) else None

            # Tính 5 đặc trưng hình học
            rel_h = line.height / max(median_h, 1.0)
            rel_w = line.width / max(page_w, 1.0)
            is_centered = abs(line.center_x - page_w / 2.0) < (page_w * 0.16)
            left_indent = max(line.x_min - base_left, 0.0)
            gap_before = (line.y_min - lines[i - 1].y_max) if i > 0 else (median_gap * 2.0)

            line.rel_h = rel_h
            line.rel_w = rel_w
            line.is_centered = is_centered
            line.left_indent = left_indent
            line.gap_before = gap_before

            # Dự đoán cấp bậc phân cấp qua predictor (0% Regex)
            pred = self.layout_predictor.predict_line_hierarchy(
                line=line.to_dict(),
                stats=stats,
                page_w=page_w,
                page_h=page_h,
                prev_line=prev_line,
                next_line=next_line
            )

            line.predicted_level = pred.get("predicted_level", LayoutHierarchyLevel.BODY)
            line.is_heading = pred.get("is_heading", False)
            line.indent_level = pred.get("indent_level", 0)

        return lines

    def process_pdf_document(
        self,
        pdf_path: str | Path,
        max_pages: int | None = None
    ) -> tuple[list[dict], dict[str, Any]]:
        """
        Quy trình xử lý Hybrid PDF hoàn chỉnh không cần rẽ nhánh 50%:
        1. Trích xuất từng trang theo chế độ phù hợp (Page-level routing).
        2. Đồng nhất hệ quy chiếu 5D.
        3. Duy trì duy nhất 1 State Machine BreadcrumbTracker và Cross-Page Continuity.
        4. Trả về danh sách Chunk chuẩn Schema Contract.
        """
        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        pages_to_process = min(total_pages, max_pages) if max_pages else total_pages

        doc_title = extract_document_title_dla(doc, pdf_path)
        tracker = BreadcrumbTracker(doc_title=doc_title)

        page_routing_info: list[dict] = []
        all_document_lines: list[UnifiedGeometryLine] = []

        # ---------------------------------------------------------------------
        # BƯỚC 1: PAGE-LEVEL ROUTING & 5D GEOMETRY NORMALIZATION
        # ---------------------------------------------------------------------
        for page_idx in range(pages_to_process):
            page_num = page_idx + 1
            page = doc[page_idx]
            pw = float(page.rect.width)
            ph = float(page.rect.height)

            # Phân loại trang scan hay digital
            is_scanned, metrics = detect_is_scanned_page(
                page=page,
                image_area_ratio_threshold=self.image_area_ratio_threshold
            )

            page_routing_info.append({
                "page_num": page_num,
                "mode": "scanned" if is_scanned else "digital",
                "reason": metrics.get("reason", "")
            })

            # Trích xuất dòng
            if is_scanned and self.ocr_extractor.is_available():
                page_lines = self.extract_scanned_lines(
                    pdf_path=pdf_path,
                    page_idx=page_idx,
                    page_w_pt=pw,
                    page_h_pt=ph
                )
            else:
                page_lines = self.extract_digital_lines(
                    page=page,
                    page_num=page_num
                )

            # Chuẩn hóa về hệ quy chiếu 5D
            normalized_page_lines = self.normalize_5d_geometry_for_page(page_lines, pw, ph)
            all_document_lines.extend(normalized_page_lines)

        # ---------------------------------------------------------------------
        # BƯỚC 2: SINGLE CONTINUOUS STATE MACHINE & CHUNKING PIPELINE
        # ---------------------------------------------------------------------
        chunks: list[dict] = []
        chunk_counter = 0

        buffer_lines: list[UnifiedGeometryLine] = []
        buffer_page_start = 1
        buffer_page_end = 1

        def emit_stream_chunk(buf: list[UnifiedGeometryLine]) -> dict | None:
            nonlocal chunk_counter
            if not buf:
                return None

            clean_text = "\n".join(l.text for l in buf if l.text).strip()
            if not clean_text:
                return None

            p_start = buf[0].page_num
            p_end = buf[-1].page_num
            p_modes = sorted(list(set(l.page_mode for l in buf)))
            chunk_type = "pdf_hybrid_stream" if len(p_modes) > 1 else f"pdf_{p_modes[0]}_stream"

            breadcrumb, b_src, b_conf = tracker.path(p_start)
            section_title = tracker.stack.get("h2") or tracker.stack.get("dieu") or ""

            full_content = f"[Breadcrumbs: {breadcrumb}]\n{clean_text}"

            chunk_obj = {
                "chunk_id": f"{pdf_path.stem}_stream_c{chunk_counter}",
                "source": pdf_path.name,
                "chunk_type": chunk_type,
                "breadcrumb": breadcrumb,
                "section_title": section_title,
                "content": full_content,
                "word_count": count_words(full_content),
                "page_start": p_start,
                "page_end": p_end,
                "page_numbers": list(range(p_start, p_end + 1))
            }
            chunk_counter += 1
            return chunk_obj

        for line in all_document_lines:
            # Bỏ qua các dòng mép trang (header/footer)
            if line.predicted_level == LayoutHierarchyLevel.HEADER_FOOTER:
                continue

            # Khi gặp Heading: cập nhật Breadcrumb và ngắt chunk nếu buffer đủ lớn hoặc là tiêu đề lớn
            if line.is_heading:
                tracker.on_heading(
                    kind=line.predicted_level,
                    label=line.text,
                    score=line.score,
                    source=f"{line.page_mode}_5d_geometry"
                )

                is_major_break = line.predicted_level in (LayoutHierarchyLevel.TITLE, LayoutHierarchyLevel.H1, "chuong", "dieu")
                cur_chars = sum(len(l.text) + 1 for l in buffer_lines)
                if buffer_lines and (cur_chars >= self.min_section_size or is_major_break):
                    c = emit_stream_chunk(buffer_lines)
                    if c:
                        chunks.append(c)
                    buffer_lines = []

                buffer_lines.append(line)
                continue

            # Tích lũy dòng nội dung
            buffer_lines.append(line)

            # Kiểm tra kích thước chunk
            cur_chars = sum(len(l.text) + 1 for l in buffer_lines)
            if cur_chars >= self.max_chunk_size:
                c = emit_stream_chunk(buffer_lines)
                if c:
                    chunks.append(c)
                buffer_lines = []

        # Flush buffer cuối cùng
        if buffer_lines:
            c = emit_stream_chunk(buffer_lines)
            if c:
                chunks.append(c)

        stats = {
            "total_pages": total_pages,
            "processed_pages": pages_to_process,
            "routing": page_routing_info,
            "total_lines_5d": len(all_document_lines),
            "total_chunks": len(chunks)
        }

        return chunks, stats
