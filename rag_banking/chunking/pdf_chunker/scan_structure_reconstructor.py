"""
Module Scan Structure Reconstructor
====================================
Tái tạo cấu trúc cây phân cấp văn bản scan ngân hàng THUẦN TÚY BẰNG ĐẶC TRƯNG HÌNH HỌC OCR.
TUYỆT ĐỐI KHÔNG DÙNG REGEX ĐỂ PHÂN CẤP (0% Regex-Based Hierarchy).

Các thành phần cốt lõi:
1. `BreadcrumbTracker`: State machine quản lý stack phả hệ phân cấp tài liệu xuyên trang (title -> h1 -> h2 -> h3).
2. `TOCVerifier`: Bộ phân tích và đối soát Mục lục (Table of Contents) ở 3 trang đầu.
3. `ScanStructureReconstructor`: Trình tái cấu trúc văn bản scan hoàn chỉnh, lọc Header/Footer và đóng gói chunks chuẩn DLA.
"""

from __future__ import annotations

import numpy as np

from .ocr_layout_predictor import (
    OCRLayoutHierarchyPredictor,
    LayoutHierarchyLevel
)

try:
    from ..unified_chunk.canonical_resolver import strip_vietnamese_accents
except (ImportError, ValueError):
    from rag_banking.chunking.unified_chunk.canonical_resolver import strip_vietnamese_accents

def fix_num(raw_str: str) -> int:
    """Khắc phục các lỗi OCR thường gặp trong ký số (O, o, l, I -> 0, 1)."""
    clean_str = raw_str.translate(str.maketrans("OolI", "0011"))
    digits = [c for c in clean_str if c.isdigit()]
    return int("".join(digits)) if digits else 1


# =============================================================================
# 1. STATE MACHINE QUẢN LÝ BREADCRUMB QUA CÁC TRANG DỰA TRÊN DỰ ĐOÁN OCR
# =============================================================================

class BreadcrumbTracker:
    """
    State machine duy trì stack phả hệ phân cấp tài liệu scan qua các trang.
    Phân cấp được điều khiển hoàn toàn bởi nhãn dự đoán hình học OCR:
    title -> h1 -> h2 -> h3.
    """
    ORDER = ["title", "h1", "h2", "h3"]
    KIND_ALIASES = {
        "phan": "title",
        "chuong": "h1",
        "muc": "h2",
        "dieu": "h2",
        "visual": "h2"
    }

    def __init__(self, doc_title: str):
        self.doc_title = doc_title
        self.stack: dict[str, str] = {}
        self.scores: dict[str, float] = {}
        self.last_kind: str | None = None
        self.source: str = "page_fallback"
        self.current_caption: str | None = None
        self.last_dieu: int = 0

    def _normalize_kind(self, kind: str) -> str:
        """Chuẩn hóa loại nhãn phân cấp về hệ trục hình học (title, h1, h2, h3)."""
        k = kind.lower()
        return self.KIND_ALIASES.get(k, k)

    def on_heading(
        self,
        kind: str,
        label: str,
        number: int | None = None,
        score: float = 1.0,
        source: str = "ocr_geometry_prediction"
    ) -> None:
        """Khi phát hiện heading cấp N, xóa cấp N trở xuống và cập nhật cấp mới."""
        norm_txt = strip_vietnamese_accents(label).lower()
        # Loại trừ Letterhead (quốc hiệu, tiêu ngữ, tên ngân hàng) khỏi stack phân cấp
        if any(ign in norm_txt for ign in ["cong hoa xa hoi", "doc lap - tu do", "doc lap tu do", "ngan hang thuong mai", "ngan hang tmcp", "vietcombank"]):
            return

        norm_kind = self._normalize_kind(kind)
        if norm_kind in self.ORDER:
            idx = self.ORDER.index(norm_kind)
            for k in self.ORDER[idx:]:
                self.stack.pop(k, None)
                self.scores.pop(k, None)

        self.stack[norm_kind] = label
        self.scores[norm_kind] = score
        self.last_kind = norm_kind
        self.source = source
        self.current_caption = None

        if number is not None:
            self.last_dieu = number


    def reset_for_new_doc(self, new_doc_title: str) -> None:
        """Reset toàn bộ stack khi phát hiện văn bản mới ghép trong file PDF."""
        self.doc_title = new_doc_title
        self.stack.clear()
        self.scores.clear()
        self.last_dieu = 0
        self.last_kind = None
        self.source = "page_fallback"
        self.current_caption = None

    def path(self, page_num: int) -> tuple[str, str, float]:
        """
        Trả về đường dẫn breadcrumb, nguồn trích xuất và điểm tin cậy:
        Returns:
            (breadcrumb_string, breadcrumb_source, breadcrumb_confidence)
        """
        parts = [self.doc_title]
        for k in self.ORDER:
            if k in self.stack:
                parts.append(self.stack[k])

        if self.current_caption:
            parts.append(self.current_caption)

        if len(parts) == 1:
            parts.append(f"Trang {page_num}")
            src = "page_fallback"
            conf = 0.70
        else:
            src = self.source
            sc_list = list(self.scores.values())
            conf = round(sum(sc_list) / max(len(sc_list), 1), 3) if sc_list else 0.85

        return " > ".join(parts), src, conf


# =============================================================================
# 2. BỘ KIỂM CHỨNG MỤC LỤC (TOC VERIFIER) - ĐỐI CHIẾU TIÊU ĐỀ
# =============================================================================

class TOCVerifier:
    """
    Trình phân tích và kiểm chứng Mục lục (Table of Contents) ở 1-3 trang đầu.
    Sử dụng fuzzy matching dựa trên chuỗi không dấu để sửa lỗi chính tả OCR.
    """

    def __init__(self):
        self.toc_found: bool = False
        self.expected_dieus: dict[int, str] = {}
        self.toc_pages: set[int] = set()

    def parse_toc_page(self, lines: list[dict], page_idx: int = 1) -> bool:
        """Nhận diện trang mục lục và thu thập outline tiêu đề (page_idx 1-indexed)."""
        if not lines:
            return False

        full_text = " ".join(l.get("text", "") for l in lines)
        norm_full = strip_vietnamese_accents(full_text).lower()

        if "muc luc" in norm_full or "table of contents" in norm_full or "content" in norm_full:
            self.toc_found = True
            self.toc_pages.add(page_idx)

        for line in lines:
            txt = line.get("text", "").strip()
            # Bắt cặp tiêu đề và số trang trong dòng mục lục (dạng: ... 12 hoặc . . . 12)
            if "..." in txt or ". ." in txt:
                clean_title = txt.split("..")[0].strip()
                if "." in clean_title:
                    clean_title = clean_title.split(".", 1)[-1].strip()
                if len(clean_title) >= 3:
                    num_cand = len(self.expected_dieus) + 1
                    self.expected_dieus[num_cand] = clean_title

        return self.toc_found

    def verify_dieu(self, dieu_num: int, ocr_title: str) -> tuple[bool, str, float]:
        """Đối chiếu tên tiêu đề với mục lục."""
        if not self.toc_found or dieu_num not in self.expected_dieus:
            return False, ocr_title, 0.0

        expected = self.expected_dieus[dieu_num]
        norm_exp = strip_vietnamese_accents(expected).lower()
        norm_ocr = strip_vietnamese_accents(ocr_title).lower()

        if norm_exp in norm_ocr or norm_ocr in norm_exp:
            return True, expected, 0.05

        return False, ocr_title, 0.0


# =============================================================================
# 3. TRÌNH TÁI TẠO CẤU TRÚC VĂN BẢN SCAN CHÍNH (OCR GEOMETRIC DLA)
# =============================================================================

class ScanStructureReconstructor:
    """
    Bộ phục hồi cấu trúc và phân loại dòng cho tài liệu scan ngân hàng
    dựa 100% trên đặc trưng hình học OCR (OCR Geometric Predictor).
    """

    def __init__(self, doc_title: str):
        self.doc_title = doc_title
        self.tracker = BreadcrumbTracker(doc_title)
        self.toc_verifier = TOCVerifier()
        self.layout_predictor = OCRLayoutHierarchyPredictor()

    def filter_header_footer_lines(
        self,
        pages_lines: list[list[dict]]
    ) -> list[list[dict]]:
        """
        Loại bỏ Header/Footer bằng vị trí tọa độ vùng biên 8% trên/dưới
        và các dòng lặp lại tần suất cao qua các trang.
        """
        total_pages = len(pages_lines)
        text_page_counts: dict[str, int] = {}

        for page_lines in pages_lines:
            seen_on_this_page = set()
            for l in page_lines:
                norm = "".join(c for c in l.get("text", "").strip().lower() if not c.isdigit())
                if len(norm) > 5 and norm not in seen_on_this_page:
                    text_page_counts[norm] = text_page_counts.get(norm, 0) + 1
                    seen_on_this_page.add(norm)

        cleaned_pages: list[list[dict]] = []
        for page_lines in pages_lines:
            page_clean: list[dict] = []
            for l in page_lines:
                page_h = l.get("page_height", 1000.0)
                y_min = l.get("y_min", 0.0)
                y_max = l.get("y_max", 0.0)
                norm = "".join(c for c in l.get("text", "").strip().lower() if not c.isdigit())

                is_edge_y = (y_min < 0.08 * page_h) or (y_max > 0.92 * page_h)
                is_repetitive = total_pages >= 3 and (text_page_counts.get(norm, 0) >= total_pages * 0.5)

                if is_edge_y or is_repetitive:
                    # Giữ lại nếu là tiêu đề cực lớn ở đầu trang
                    if l.get("height", 0) > 30 and not is_repetitive:
                        page_clean.append(l)
                    continue

                page_clean.append(l)
            cleaned_pages.append(page_clean)

        return cleaned_pages

    def detect_heading(
        self,
        line: dict,
        median_h: float,
        page_w: float,
        next_line: dict | None = None
    ) -> dict | None:
        """
        Dự đoán Heading THUẦN TÚY BẰNG BỐ CỤC HÌNH HỌC OCR (0% Regex):
        Sử dụng OCRLayoutHierarchyPredictor tính toán font size, căn giữa, khoảng cách đứng.
        """
        text = line.get("text", "").strip()
        score = line.get("score", 0.85)
        page_h = line.get("page_height", 1000.0)

        stats = {
            "median_h": median_h,
            "base_left": float(line.get("x_min", 50.0)),
            "median_gap": max(median_h * 0.5, 6.0)
        }

        # Gọi bộ phân loại hình học OCR
        pred = self.layout_predictor.predict_line_hierarchy(
            line=line,
            stats=stats,
            page_w=page_w,
            page_h=page_h,
            next_line=next_line
        )

        level = pred["predicted_level"]
        is_heading = pred["is_heading"]

        if not is_heading:
            return None

        # Trích xuất số nếu có trong tiêu đề (dùng cho tính liên tục)
        num = None
        for token in text.split():
            clean_tok = "".join(c for c in token if c.isdigit())
            if clean_tok:
                num = int(clean_tok)
                break

        # Ghép 2 dòng nếu dòng tiếp theo cũng là tiêu đề viết hoa đi liền kề
        label = text
        if next_line:
            next_t = next_line.get("text", "").strip()
            next_h = next_line.get("height", 15.0)
            if next_t.isupper() and next_h >= median_h * 0.95 and len(next_t) < 80:
                label = f"{label}. {next_t}"

        # Ánh xạ kết quả dự đoán OCR
        kind = "dieu" if level in [LayoutHierarchyLevel.H2, LayoutHierarchyLevel.H3] else "chuong"
        if level == LayoutHierarchyLevel.TITLE:
            kind = "title"

        return {
            "kind": kind,
            "level": level,
            "number": num,
            "label": label,
            "score": score,
            "source": "ocr_geometry_prediction",
            "features": pred["features"]
        }

    def _build_scan_chunk(
        self,
        buffer_lines: list[dict],
        source_name: str,
        chunk_idx: int,
        page_start: int,
        page_end: int
    ) -> dict | None:
        """Đóng gói các dòng văn bản đệm thành đối tượng Chunk hoàn chỉnh kèm metadata hình học."""
        if not buffer_lines:
            return None

        text_content = "\n".join(l.get("text", "") for l in buffer_lines).strip()
        if not text_content:
            return None

        breadcrumb, b_source, b_conf = self.tracker.path(page_start)
        ocr_scores = [l.get("score", 0.85) for l in buffer_lines]
        avg_ocr_conf = round(sum(ocr_scores) / max(len(ocr_scores), 1), 3)

        # Tiêu đề mục đang hoạt động
        section_label = self.tracker.stack.get("h2") or self.tracker.stack.get("dieu") or ""

        # Xác định độ thụt lề lớn nhất trong chunk từ OCR (0: Body, 1: Khoản, 2: Điểm)
        indent_levels = [l.get("indent_level", 0) for l in buffer_lines]
        max_indent = max(indent_levels) if indent_levels else 0

        header_prefix = f"[Breadcrumbs: {breadcrumb}]"
        full_chunk_text = f"{header_prefix}\n{text_content}"
        word_count = len(text_content.split())

        return {
            "chunk_id": f"{source_name}_scan_c{chunk_idx}",
            "source": source_name,
            "chunk_type": "pdf_ocr_scan",
            "breadcrumb": breadcrumb,
            "breadcrumb_source": b_source,
            "breadcrumb_confidence": b_conf,
            "heading_ocr_confidence": self.tracker.scores.get(self.tracker.last_kind or "h2", 0.88),
            "ocr_confidence": avg_ocr_conf,
            "page_start": page_start,
            "page_end": page_end,
            "page_numbers": list(range(page_start, page_end + 1)),
            "section_title": section_label,
            "indent_level": max_indent,
            "content": full_chunk_text,
            "word_count": word_count
        }

    def reconstruct_document_chunks(
        self,
        pages_lines: list[list[dict]],
        source_name: str,
        max_chunk_size: int = 800,
        min_section_size: int = 120
    ) -> list[dict]:
        """
        Quy trình chuẩn hóa và phân đoạn toàn bộ tài liệu scan bằng OCR Layout Analysis:
        1. Parse Mục lục TOC ở 3 trang đầu (1-indexed).
        2. Lọc Header/Footer lặp lại.
        3. Phân loại bố cục từng dòng bằng OCRLayoutHierarchyPredictor (0% Regex).
        4. Cập nhật State Machine BreadcrumbTracker.
        5. Đóng gói chunk đạt chuẩn Section-Length-Aware.
        """
        if not pages_lines:
            return []

        # 1. Parse Mục lục TOC ở 3 trang đầu (chú ý page_idx 1-indexed)
        for page_idx in range(min(3, len(pages_lines))):
            self.toc_verifier.parse_toc_page(pages_lines[page_idx], page_idx=page_idx + 1)

        # 2. Lọc Header/Footer
        cleaned_pages = self.filter_header_footer_lines(pages_lines)

        chunks: list[dict] = []
        chunk_counter = 0

        buffer_lines: list[dict] = []
        buffer_page_start = 1
        buffer_page_end = 1

        for page_idx, page_lines in enumerate(cleaned_pages, start=1):
            if not page_lines:
                continue

            # Bỏ qua trang mục lục thuần túy
            if page_idx in self.toc_verifier.toc_pages:
                continue

            page_w = page_lines[0].get("page_width", 800.0)
            page_h = page_lines[0].get("page_height", 1100.0)

            # Dự đoán phân cấp bố cục cho toàn bộ dòng của trang
            annotated_lines = self.layout_predictor.classify_page_lines(
                lines=page_lines,
                page_w=page_w,
                page_h=page_h
            )

            for line in annotated_lines:
                is_heading = line.get("is_heading", False)
                pred_level = line.get("predicted_level", LayoutHierarchyLevel.BODY)

                # Khi OCR dự đoán đây là một Heading (title, h1, h2, h3)
                if is_heading:
                    # Flush buffer văn bản trước đó nếu có
                    if buffer_lines:
                        chunk_obj = self._build_scan_chunk(
                            buffer_lines=buffer_lines,
                            source_name=source_name,
                            chunk_idx=chunk_counter,
                            page_start=buffer_page_start,
                            page_end=buffer_page_end
                        )
                        if chunk_obj:
                            chunks.append(chunk_obj)
                            chunk_counter += 1
                        buffer_lines = []

                    # Cập nhật State Machine
                    self.tracker.on_heading(
                        kind=pred_level,
                        label=line.get("text", ""),
                        score=line.get("score", 0.90),
                        source="ocr_geometry_prediction"
                    )

                    buffer_page_start = page_idx
                    buffer_page_end = page_idx
                    continue

                # Dòng nội dung (body, sub_item, deep_item)
                buffer_lines.append(line)
                buffer_page_end = page_idx

                # Kiểm tra dung lượng buffer
                current_chars = sum(len(l.get("text", "")) + 1 for l in buffer_lines)
                if current_chars >= max_chunk_size:
                    chunk_obj = self._build_scan_chunk(
                        buffer_lines=buffer_lines,
                        source_name=source_name,
                        chunk_idx=chunk_counter,
                        page_start=buffer_page_start,
                        page_end=buffer_page_end
                    )
                    if chunk_obj:
                        chunks.append(chunk_obj)
                        chunk_counter += 1
                    buffer_lines = []
                    buffer_page_start = page_idx

        # Flush buffer còn lại cuối cùng
        if buffer_lines:
            chunk_obj = self._build_scan_chunk(
                buffer_lines=buffer_lines,
                source_name=source_name,
                chunk_idx=chunk_counter,
                page_start=buffer_page_start,
                page_end=buffer_page_end
            )
            if chunk_obj:
                chunks.append(chunk_obj)

        # Fallback an toàn: Nếu tài liệu toàn tiêu đề / slide bìa không có dòng body
        if not chunks:
            all_lines = [l for page in cleaned_pages for l in page if l.get("text", "").strip()]
            if all_lines:
                chunk_obj = self._build_scan_chunk(
                    buffer_lines=all_lines,
                    source_name=source_name,
                    chunk_idx=0,
                    page_start=1,
                    page_end=max(len(cleaned_pages), 1)
                )
                if chunk_obj:
                    chunks.append(chunk_obj)

        return chunks
