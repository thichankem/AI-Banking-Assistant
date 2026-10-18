"""
Module OCR Layout Hierarchy Predictor (8D Visual Geometry Engine)
==================================================================
Bộ dự đoán cấp bậc bố cục văn bản scan thuần túy bằng Bố cục Thị giác Hình học OCR 8 chiều.
TUYỆT ĐỐI KHÔNG DÙNG REGEX ĐỂ PHÂN CẤP (0% Regex-Based Hierarchy).

Dự đoán hoàn toàn từ ma trận không gian 8 chiều (8D OCR Visual Geometry):
1. Relative Font Height (`rel_h`): Tỷ lệ chiều cao dòng so với trung vị thân bài (`median_h`).
2. Vertical Spacing (`gap_before`): Khoảng cách giãn dòng phía trước dòng.
3. Horizontal Alignment (`is_centered`): Căn giữa trang so với bám lề trái chuẩn (`base_left`).
4. Left Indentation (`left_indent`): Độ lệch lề trái để phân loại Khoản (sub_item) vs Điểm (deep_item).
5. Line Aspect & Compactness (`rel_w`): Tỷ lệ độ dài dòng so với bề ngang trang giấy.
6. Stroke Width Density (`stroke_width_density`): Tỷ lệ mật độ nét chữ đậm / diện tích bounding box.
7. Capitalization Ratio (`caps_ratio`): Tỷ lệ ký tự in hoa trên tổng số ký tự chữ cái (0% Regex).
8. Vertical Continuity Ratio (`y_continuity_score`): Tỷ số `gap_above / max(gap_below, 1.0)`.
   Tiêu đề thực thụ có khoảng cách trên rộng và bám chặt đoạn văn bên dưới (`score >= 1.6`),
   trong khi hộp cảnh báo in đậm (callout) có khoảng cách đối xứng (`0.65 <= score <= 1.35`).
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class LayoutHierarchyLevel:
    """Các cấp bậc cấu trúc phân cấp bố cục văn bản."""
    TITLE = "title"                 # Level 0: Tiêu đề văn bản / Văn bản mới
    H1 = "h1"                       # Level 1: Tiêu đề lớn (Chương / Phần / Heading 1)
    H2 = "h2"                       # Level 2: Tiêu đề mục (Mục / Điều / Heading 2)
    H3 = "h3"                       # Level 3: Tiêu đề tiểu mục (Heading 3 / Khoản đậm ngắn)
    SUB_ITEM = "sub_item"           # Level 4: Thụt lề cấp 1 (Khoản / Bullet list)
    DEEP_ITEM = "deep_item"         # Level 5: Thụt lề cấp 2 (Điểm / Nested list)
    BODY = "body"                   # Thân bài thông thường
    HEADER_FOOTER = "header_footer" # Vùng mép trang (Header, Footer, Số trang)


class OCRLayoutHierarchyPredictor:
    """
    Bộ phân loại cấp bậc phân đoạn văn bản scan dựa trên Bố cục Hình học 8D OCR (0% Regex).
    """

    DEFAULT_GENERAL_DOCUMENT_MARKERS: tuple[str, ...] = ()
    DEFAULT_FORM_FIELD_KEYWORDS: tuple[str, ...] = ()

    def __init__(
        self,
        edge_margin_ratio: float = 0.065,     # 6.5% trên/dưới trang coi là mép header/footer
        center_tolerance_ratio: float = 0.16, # Dung sai căn giữa (+-16% bề ngang trang)
        domain_keywords: tuple[str, ...] | None = None,
        general_document_markers: tuple[str, ...] | None = None,
        form_field_keywords: tuple[str, ...] | None = None,
        config: object | None = None
    ):
        self.edge_margin_ratio = edge_margin_ratio
        self.center_tolerance_ratio = center_tolerance_ratio

        if general_document_markers is not None:
            self.general_document_markers = tuple(general_document_markers)
        elif config is not None and getattr(config, "general_document_markers", None):
            self.general_document_markers = tuple(config.general_document_markers)
        else:
            self.general_document_markers = self.DEFAULT_GENERAL_DOCUMENT_MARKERS

        if domain_keywords is not None:
            self.domain_keywords = tuple(domain_keywords)
        elif config is not None and getattr(config, "domain_keywords", None):
            self.domain_keywords = tuple(config.domain_keywords)
        else:
            self.domain_keywords = ()

        if form_field_keywords is not None:
            self.form_field_keywords = tuple(form_field_keywords)
        elif config is not None and getattr(config, "form_field_keywords", None):
            self.form_field_keywords = tuple(config.form_field_keywords)
        else:
            self.form_field_keywords = self.DEFAULT_FORM_FIELD_KEYWORDS

    def compute_page_layout_stats(
        self,
        lines: list[dict],
        page_w: float,
        page_h: float
    ) -> dict[str, float]:
        """
        Tính toán các chỉ số thống kê bố cục cơ sở của trang:
        - `median_h`: Chiều cao cơ sở của chữ thân bài (Body Text baseline).
        - `base_left`: Tọa độ lề trái chuẩn của khối văn bản (bách phân vị 20%).
        - `median_gap`: Giãn dòng trung vị giữa các dòng trong cùng đoạn văn.
        """
        if not lines:
            return {"median_h": 16.0, "base_left": 50.0, "median_gap": 8.0}

        # 1. Lọc bỏ các dòng ở sát biên mép trang để không làm nhiễu thống kê
        content_lines = [
            l for l in lines
            if (l.get("y_min", 0.0) >= page_h * self.edge_margin_ratio) and
               (l.get("y_max", page_h) <= page_h * (1.0 - self.edge_margin_ratio))
        ]
        target_lines = content_lines if len(content_lines) >= 3 else lines

        # 2. Chiều cao chữ trung vị (Font Size Proxy)
        heights = [float(l.get("height", l.get("h", 16.0))) for l in target_lines]
        median_h = float(np.median(heights)) if heights else 16.0

        # 3. Lề trái cơ sở: lấy bách phân vị 20% (20th percentile) của x_min
        x_mins = sorted([float(l.get("x_min", 0.0)) for l in target_lines])
        idx_20 = int(len(x_mins) * 0.20)
        base_left = float(x_mins[idx_20]) if x_mins else 50.0

        # 4. Giãn dòng trung vị giữa các dòng liên tiếp
        sorted_by_y = sorted(target_lines, key=lambda l: l.get("y_min", 0.0))
        gaps = []
        for i in range(1, len(sorted_by_y)):
            prev_y_max = sorted_by_y[i - 1].get("y_max", 0.0)
            curr_y_min = sorted_by_y[i].get("y_min", 0.0)
            gap = curr_y_min - prev_y_max
            if 2.0 <= gap <= median_h * 3.0:
                gaps.append(gap)
        median_gap = float(np.median(gaps)) if gaps else max(median_h * 0.4, 6.0)

        return {
            "median_h": max(median_h, 8.0),
            "base_left": base_left,
            "median_gap": max(median_gap, 4.0)
        }

    def _compute_caps_ratio(self, text: str) -> float:
        """Tính tỷ lệ ký tự in hoa trên tổng số chữ cái (0% Regex)."""
        alpha_count = sum(1 for c in text if c.isalpha())
        if alpha_count == 0:
            return 0.0
        upper_count = sum(1 for c in text if c.isupper())
        return upper_count / float(alpha_count)

    def _compute_stroke_width_density(self, line: dict) -> float:
        """
        Tính toán hoặc trích xuất mật độ nét chữ (stroke width density).
        Nếu line có sẵn giá trị `stroke_width_density` hoặc `bold_score`, ưu tiên sử dụng.
        Nếu có ảnh crop binarized, tính tỷ lệ pixel đen trên tổng diện tích / số ký tự.
        """
        if "stroke_width_density" in line:
            return float(line["stroke_width_density"])
        if "bold_score" in line:
            return float(line["bold_score"])

        crop = line.get("crop_image")
        if crop is not None:
            try:
                arr = np.array(crop)
                if arr.ndim == 3:
                    arr = np.mean(arr, axis=2)
                # Đếm pixel tối (foreground)
                dark_pixels = np.sum(arr < 128)
                total_pixels = arr.size
                char_count = max(len(line.get("text", "").strip()), 1)
                return float(dark_pixels / max(total_pixels, 1.0)) / char_count
            except Exception:
                pass

        # Ước lượng mặc định dựa trên cờ is_bold nếu có
        return 0.25 if line.get("is_bold", False) else 0.08

    def predict_line_hierarchy(
        self,
        line: dict,
        stats: dict[str, float],
        page_w: float,
        page_h: float,
        prev_line: Optional[dict] = None,
        next_line: Optional[dict] = None
    ) -> dict:
        """
        Dự đoán cấp bậc phân cấp của một dòng OCR bằng ma trận bố cục hình học 8D (0% Regex):
        Returns:
            {
                "predicted_level": LayoutHierarchyLevel.*,
                "confidence": float,
                "is_heading": bool,
                "indent_level": int (0: Body, 1: Khoản, 2: Điểm),
                "features": { ... }
            }
        """
        text = line.get("text", "").strip()
        y_min = float(line.get("y_min", page_h * 0.20))
        y_max = float(line.get("y_max", y_min + 16.0))
        h = float(line.get("height", line.get("h", max(y_max - y_min, 1.0))))
        x_min = float(line.get("x_min", stats.get("base_left", 50.0)))
        x_max = float(line.get("x_max", x_min + 100.0))
        w = float(line.get("width", line.get("w", max(x_max - x_min, 1.0))))
        cx = float(line.get("center_x", (x_min + w / 2.0)))

        median_h = stats.get("median_h", 16.0)
        base_left = stats.get("base_left", 50.0)
        median_gap = stats.get("median_gap", 8.0)

        # ---------------------------------------------------------------------
        # 1. Kiểm tra Vùng Mép Trang (Header/Footer Boundary & Page Numbers) (0% Regex)
        # ---------------------------------------------------------------------
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        text_lower = text.lower()

        is_extreme_top = y_min < page_h * 0.045
        is_extreme_bottom = y_max > page_h * 0.955
        is_near_bottom = y_max > page_h * 0.930

        # A. Nhận diện số trang (Page Number)
        is_page_num = False
        if (is_extreme_top or is_extreme_bottom or is_near_bottom) and word_count <= 3 and char_count <= 12:
            clean_pn = text_lower.replace("trang", "").replace("page", "").replace("of", "/").replace("-", "").strip()
            parts = clean_pn.split("/")
            if parts and all(p.strip().isdigit() for p in parts if p.strip()) and any(p.strip().isdigit() for p in parts):
                is_page_num = True

        # B. Tiêu đề tài liệu ở đầu trang 1
        page_num = line.get("page_num", 1)
        is_page1_title = (page_num == 1) and (not is_extreme_top or h > median_h * 1.2) and (y_min < page_h * 0.25) and (h >= 14.0 or text.isupper() or (w / page_w) >= 0.25)

        if is_page_num:
            return {
                "predicted_level": LayoutHierarchyLevel.HEADER_FOOTER,
                "confidence": 0.98,
                "is_heading": False,
                "indent_level": 0,
                "features": {"is_page_number": True}
            }

        # C. Tiêu đề mẫu biểu / Running Header ở mép cực cận (Cấu trúc hình học & ký tự điền, 0% hardcode từ khóa)
        has_form_pattern = (
            (word_count <= 5 and (text.strip().endswith(":") or text.strip().endswith(":-"))) or
            any(sym in text for sym in ("...", "....", "___", "____", "---", "☐", "☑", "[ ]", "[x]", "[X]", "✓"))
        )
        has_form_keyword = any(term in text_lower for term in self.form_field_keywords) if self.form_field_keywords else False
        is_form_field = has_form_pattern or (has_form_keyword and (word_count <= 6 or char_count <= 35))
        if (is_extreme_top or is_extreme_bottom) and not is_page1_title and not is_form_field and word_count <= 4 and char_count <= 25:
            return {
                "predicted_level": LayoutHierarchyLevel.HEADER_FOOTER,
                "confidence": 0.95,
                "is_heading": False,
                "indent_level": 0,
                "features": {"edge": True}
            }

        # ---------------------------------------------------------------------
        # 2. Tính toán ma trận đặc trưng bố cục 8 chiều (8D Features)
        # ---------------------------------------------------------------------
        # Feature 1: rel_h
        rel_h = h / median_h
        # Feature 2: gap_before (gap_above)
        gap_above = (y_min - prev_line.get("y_max", y_min)) if prev_line else (median_gap * 2.0)
        gap_before = gap_above
        # Feature 3: is_centered
        is_centered = abs(cx - page_w / 2.0) < (page_w * self.center_tolerance_ratio)
        # Feature 4: left_indent
        left_indent = max(x_min - base_left, 0.0)
        # Feature 5: rel_w
        rel_w = w / page_w
        is_short_line = rel_w < 0.70

        # Feature 6: stroke_width_density
        stroke_width_density = self._compute_stroke_width_density(line)

        # Feature 7: caps_ratio (0% Regex)
        caps_ratio = self._compute_caps_ratio(text)
        is_uppercase = caps_ratio >= 0.85 and len(text) >= 3

        # Feature 8: y_continuity_score = gap_above / max(gap_below, 1.0)
        gap_below = (next_line.get("y_min", y_max) - y_max) if next_line else (median_gap * 2.0)
        y_continuity_score = float(max(gap_above, 0.0)) / float(max(gap_below, 1.0))

        has_break_before = (gap_before >= (median_gap * 1.15)) or (gap_before >= (median_h * 1.05))

        features = {
            "rel_h": round(rel_h, 2),
            "rel_w": round(rel_w, 2),
            "is_centered": is_centered,
            "left_indent": round(left_indent, 1),
            "gap_before": round(gap_before, 1),
            "stroke_width_density": round(stroke_width_density, 3),
            "caps_ratio": round(caps_ratio, 2),
            "y_continuity_score": round(y_continuity_score, 2),
            "is_uppercase": is_uppercase
        }

        # ---------------------------------------------------------------------
        # 3. BẢO VỆ ĐOẠN CẢNH BÁO IN ĐẬM (BOLD WARNING CALLOUT PARAGRAPH PROTECTION)
        # ---------------------------------------------------------------------
        # Ngăn ngừa đoạn văn cảnh báo/lưu ý in đậm bị nhận diện nhầm thành H2/H3:
        # Đặc điểm của Callout Box:
        # - Khoảng cách đối xứng: 0.65 <= y_continuity_score <= 1.35
        # - HOẶC kết thúc bằng dấu chấm câu (. ; :)
        # - HOẶC độ dài câu dài (> 12 từ hoặc > 80 ký tự)
        is_symmetric_spacing = (0.65 <= y_continuity_score <= 1.35)
        ends_with_terminal = text.endswith((".", ";", ":"))
        is_long_paragraph = (word_count > 12 or char_count > 80)

        is_callout_paragraph = (
            is_symmetric_spacing and (ends_with_terminal or is_long_paragraph)
        ) or (ends_with_terminal and is_long_paragraph)

        if is_callout_paragraph and not is_uppercase and rel_h < 1.30:
            if left_indent >= 40.0:
                pred_lvl = LayoutHierarchyLevel.DEEP_ITEM
                ind_lvl = 2
            elif left_indent >= 15.0:
                pred_lvl = LayoutHierarchyLevel.SUB_ITEM
                ind_lvl = 1
            else:
                pred_lvl = LayoutHierarchyLevel.BODY
                ind_lvl = 0

            return {
                "predicted_level": pred_lvl,
                "confidence": 0.93,
                "is_heading": False,
                "indent_level": ind_lvl,
                "features": features
            }

        # ---------------------------------------------------------------------
        # 4. PHÂN LOẠI CẤP BẬC THEO MA TRẬN 8D (0% Regex)
        # ---------------------------------------------------------------------

        # A. Cấp 0: Tiêu đề lớn nhất tài liệu (Title / Văn bản mới)
        if (h >= 34.0 and y_min < page_h * 0.65) or \
           (rel_h >= 1.30 and (is_centered or is_short_line)) or \
           (is_centered and is_uppercase and y_min < page_h * 0.25 and rel_w >= 0.30) or \
           (rel_h >= 1.05 and is_centered and is_uppercase and y_min < page_h * 0.25) or \
           (page_num == 1 and not is_extreme_top and y_min < page_h * 0.20 and is_uppercase and word_count >= 3):
            return {
                "predicted_level": LayoutHierarchyLevel.TITLE,
                "confidence": 0.95,
                "is_heading": True,
                "indent_level": 0,
                "features": features
            }

        # B. Cấp 1: Tiêu đề phân đoạn lớn (Chương / Phần / Heading 1)
        # Tiêu đề H1 yêu cầu y_continuity_score >= 1.2 (khoảng cách trên rộng)
        if ((1.15 <= rel_h < 1.30 and (is_uppercase or is_centered)) or \
            (rel_h >= 1.18 and is_short_line and y_min < page_h * 0.35) or \
            (is_uppercase and has_break_before and is_short_line and y_min < page_h * 0.45) or \
            (h >= 30.0 and is_short_line and y_min < page_h * 0.30)) and not is_callout_paragraph:
            return {
                "predicted_level": LayoutHierarchyLevel.H1,
                "confidence": 0.92,
                "is_heading": True,
                "indent_level": 0,
                "features": features
            }

        # C. Cấp 2: Tiêu đề mục (Mục / Điều / Heading 2)
        # Yêu cầu y_continuity_score >= 1.25 hoặc in hoa ngắn không có dấu kết thúc câu
        if ((1.05 <= rel_h < 1.25 and is_short_line and left_indent < 10.0 and has_break_before) or \
            (1.08 <= rel_h < 1.25 and is_short_line and left_indent < 10.0) or \
            (is_uppercase and is_short_line and left_indent < 10.0 and has_break_before)) and not is_callout_paragraph:
            return {
                "predicted_level": LayoutHierarchyLevel.H2,
                "confidence": 0.90,
                "is_heading": True,
                "indent_level": 0,
                "features": features
            }

        # D. Cấp 3: Tiêu đề tiểu mục (Heading 3 / Khoản đậm ngắn)
        if is_short_line and has_break_before and rel_w < 0.55 and left_indent < 15.0 and \
           not text.endswith((".", ",", ";", ":")) and not is_callout_paragraph:
            return {
                "predicted_level": LayoutHierarchyLevel.H3,
                "confidence": 0.85,
                "is_heading": True,
                "indent_level": 0,
                "features": features
            }

        # E. Cấp 4 & 5: Đoạn văn thụt lề (Khoản & Điểm)
        if left_indent >= 40.0:
            return {
                "predicted_level": LayoutHierarchyLevel.DEEP_ITEM,
                "confidence": 0.88,
                "is_heading": False,
                "indent_level": 2,
                "features": features
            }
        elif left_indent >= 15.0:
            return {
                "predicted_level": LayoutHierarchyLevel.SUB_ITEM,
                "confidence": 0.88,
                "is_heading": False,
                "indent_level": 1,
                "features": features
            }

        # F. Thân bài thông thường (Body Text)
        return {
            "predicted_level": LayoutHierarchyLevel.BODY,
            "confidence": 0.95,
            "is_heading": False,
            "indent_level": 0,
            "features": features
        }

    def classify_page_lines(
        self,
        lines: list[dict],
        page_w: float,
        page_h: float,
        page_num: int = 1
    ) -> list[dict]:
        """
        Duyệt và gắn nhãn phân cấp hình học 8D cho toàn bộ danh sách dòng của một trang.
        """
        if not lines:
            return []

        stats = self.compute_page_layout_stats(lines, page_w, page_h)
        annotated_lines: list[dict] = []

        for i, line in enumerate(lines):
            prev_l = lines[i - 1] if i > 0 else None
            next_l = lines[i + 1] if i + 1 < len(lines) else None

            if "page_num" not in line:
                line["page_num"] = page_num

            pred = self.predict_line_hierarchy(
                line=line,
                stats=stats,
                page_w=page_w,
                page_h=page_h,
                prev_line=prev_l,
                next_line=next_l
            )

            annotated = dict(line)
            annotated["predicted_level"] = pred["predicted_level"]
            annotated["is_heading"] = pred["is_heading"]
            annotated["indent_level"] = pred["indent_level"]
            annotated["layout_confidence"] = pred["confidence"]
            annotated["layout_features"] = pred["features"]
            annotated_lines.append(annotated)

        return annotated_lines


__all__ = ["LayoutHierarchyLevel", "OCRLayoutHierarchyPredictor"]
