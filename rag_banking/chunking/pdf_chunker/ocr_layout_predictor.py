"""
Module OCR Layout Hierarchy Predictor
======================================
Bộ dự đoán cấp bậc bố cục văn bản scan thuần túy bằng Bố cục Thị giác Hình học OCR.
TUYỆT ĐỐI KHÔNG DÙNG REGEX ĐỂ PHÂN CẤP (0% Regex-Based Hierarchy).

Dự đoán hoàn toàn từ ma trận không gian 5 chiều (5D OCR Visual Geometry):
1. Relative Font Height (`rel_h`): Tỷ lệ chiều cao dòng so với trung vị thân bài (`median_h`).
2. Vertical Spacing (`gap_before`): Khoảng cách giãn dòng phía trước dòng.
3. Horizontal Alignment (`is_centered`): Căn giữa trang so với bám lề trái chuẩn (`base_left`).
4. Left Indentation (`left_indent`): Độ lệch lề trái để phân loại Khoản (sub_item) vs Điểm (deep_item).
5. Line Aspect & Compactness (`rel_w`): Tỷ lệ độ dài dòng so với bề ngang trang giấy.
"""

from __future__ import annotations

import numpy as np


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
    Bộ phân loại cấp bậc phân đoạn văn bản scan dựa trên Bố cục Hình học OCR (0% Regex).
    """

    def __init__(
        self,
        edge_margin_ratio: float = 0.065,     # 6.5% trên/dưới trang coi là mép header/footer
        center_tolerance_ratio: float = 0.16  # Dung sai căn giữa (+-16% bề ngang trang)
    ):
        self.edge_margin_ratio = edge_margin_ratio
        self.center_tolerance_ratio = center_tolerance_ratio

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
        heights = [float(l.get("height", 16.0)) for l in target_lines]
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

    def predict_line_hierarchy(
        self,
        line: dict,
        stats: dict[str, float],
        page_w: float,
        page_h: float,
        prev_line: dict | None = None,
        next_line: dict | None = None
    ) -> dict:
        """
        Dự đoán cấp bậc phân cấp của một dòng OCR hoàn toàn bằng đặc trưng hình học:
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
        has_explicit_y = "y_min" in line
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

        # B. Bảo vệ Footnote & Chú thích nghiệp vụ: nếu dòng có câu hoàn chỉnh hoặc từ khóa nghiệp vụ
        is_footnote_or_content = False
        if word_count >= 5 or char_count >= 25:
            banking_terms = (
                "bổ sung", "ghi chú", "lưu ý", "áp dụng", "hiệu lực", "vnđ", "đồng", 
                "phí", "thẻ", "khách hàng", "chủ tk", "tài khoản", "nghĩa vụ", "quyền",
                "cam kết", "điều", "khoản", "hợp đồng", "thỏa thuận", "quy định",
                "hạn mức", "lãi suất", "thanh toán", "chi nhánh", "phòng giao dịch",
                "không trái", "pháp luật", "giấy tờ", "căn cước", "sử dụng"
            )
            if any(term in text_lower for term in banking_terms):
                is_footnote_or_content = True
            elif text.endswith((".", ",", ";", ":", ")")):
                is_footnote_or_content = True

        # C. Tiêu đề tài liệu ở đầu trang 1
        page_num = line.get("page_num", 1)
        is_page1_title = (page_num == 1) and (y_min < page_h * 0.25) and (h >= 14.0 or text.isupper() or (w / page_w) >= 0.25)

        if is_page_num:
            return {
                "predicted_level": LayoutHierarchyLevel.HEADER_FOOTER,
                "confidence": 0.98,
                "is_heading": False,
                "indent_level": 0,
                "features": {"is_page_number": True}
            }

        # D. Tiêu đề mẫu biểu / Running Header ở mép cực cận (Mẫu số 02, PUBLIC, VIETCOMBANK)
        is_form_field = any(term in text_lower for term in (
            "loại", "thẻ", "card", "tài khoản", "account", "dịch vụ", "service", 
            "họ và tên", "ngày sinh", "chữ ký", "stt", "mức phí", "đơn vị", "nội dung"
        ))
        if (is_extreme_top or is_extreme_bottom) and not is_page1_title and not is_form_field and word_count <= 4 and char_count <= 25:
            return {
                "predicted_level": LayoutHierarchyLevel.HEADER_FOOTER,
                "confidence": 0.95,
                "is_heading": False,
                "indent_level": 0,
                "features": {"edge": True}
            }

        # Nếu là văn bản nội dung / footnote / điều khoản pháp lý / tiêu đề / nhãn biểu mẫu: KHÔNG BAO GIỜ là header_footer
        if is_footnote_or_content or is_page1_title or is_form_field:
            pass
        elif (is_extreme_top or is_extreme_bottom) and not is_form_field and char_count <= 40 and word_count <= 6:
            return {
                "predicted_level": LayoutHierarchyLevel.HEADER_FOOTER,
                "confidence": 0.90,
                "is_heading": False,
                "indent_level": 0,
                "features": {"edge": True}
            }

        # ---------------------------------------------------------------------
        # 2. Tính toán các đặc trưng bố cục hình học chuẩn hóa
        # ---------------------------------------------------------------------
        rel_h = h / median_h
        rel_w = w / page_w
        is_centered = abs(cx - page_w / 2.0) < (page_w * self.center_tolerance_ratio)
        left_indent = max(x_min - base_left, 0.0)
        is_short_line = rel_w < 0.70

        # Khoảng cách dọc trước dòng
        gap_before = (y_min - prev_line.get("y_max", y_min)) if prev_line else (median_gap * 2.0)
        has_break_before = (gap_before >= (median_gap * 1.15)) or (gap_before >= (median_h * 1.05))

        # Tín hiệu bề mặt: in hoa toàn bộ
        is_uppercase = text.isupper() and len(text) >= 3

        features = {
            "rel_h": round(rel_h, 2),
            "rel_w": round(rel_w, 2),
            "is_centered": is_centered,
            "left_indent": round(left_indent, 1),
            "gap_before": round(gap_before, 1),
            "is_uppercase": is_uppercase
        }

        # ---------------------------------------------------------------------
        # 3. Phân loại theo Ma trận Bố cục Hình học (0% Regex)
        # ---------------------------------------------------------------------

        # A. Cấp 0: Tiêu đề lớn nhất tài liệu (Title / Văn bản mới)
        if (h >= 34.0 and y_min < page_h * 0.65) or \
           (rel_h >= 1.30 and (is_centered or is_short_line)) or \
           (is_centered and is_uppercase and y_min < page_h * 0.25 and rel_w >= 0.30) or \
           (rel_h >= 1.05 and is_centered and is_uppercase and y_min < page_h * 0.25) or \
           (page_num == 1 and y_min < page_h * 0.20 and is_uppercase and word_count >= 3):
            return {
                "predicted_level": LayoutHierarchyLevel.TITLE,
                "confidence": 0.95,
                "is_heading": True,
                "indent_level": 0,
                "features": features
            }

        # B. Cấp 1: Tiêu đề phân đoạn lớn (Chương / Phần / Slide Title / Mục lục)
        if (1.15 <= rel_h < 1.30 and (is_uppercase or is_centered)) or \
           (rel_h >= 1.18 and is_short_line and y_min < page_h * 0.35) or \
           (is_uppercase and has_break_before and is_short_line and y_min < page_h * 0.45) or \
           (h >= 30.0 and is_short_line and y_min < page_h * 0.30):
            return {
                "predicted_level": LayoutHierarchyLevel.H1,
                "confidence": 0.92,
                "is_heading": True,
                "indent_level": 0,
                "features": features
            }

        # C. Cấp 2: Tiêu đề mục (Mục / Điều / Heading 2)
        if (1.05 <= rel_h < 1.25 and is_short_line and left_indent < 10.0 and has_break_before) or \
           (1.08 <= rel_h < 1.25 and is_short_line and left_indent < 10.0) or \
           (is_uppercase and is_short_line and left_indent < 10.0 and has_break_before):
            return {
                "predicted_level": LayoutHierarchyLevel.H2,
                "confidence": 0.90,
                "is_heading": True,
                "indent_level": 0,
                "features": features
            }

        # D. Cấp 3: Tiêu đề tiểu mục (Heading 3 / Dòng ngắn ngắt đoạn không có dấu chấm cuối)
        if is_short_line and has_break_before and rel_w < 0.55 and left_indent < 15.0 and not text.endswith((".", ",", ";", ":")):
            return {
                "predicted_level": LayoutHierarchyLevel.H3,
                "confidence": 0.85,
                "is_heading": True,
                "indent_level": 0,
                "features": features
            }

        # E. Cấp 4 & 5: Đoạn văn thụt lề (Khoản & Điểm)
        # Ngưỡng thụt lề >= 40px tương đương thụt lề cấp 2 (Điểm)
        if left_indent >= 40.0:
            return {
                "predicted_level": LayoutHierarchyLevel.DEEP_ITEM,
                "confidence": 0.88,
                "is_heading": False,
                "indent_level": 2,  # Tương đương cấp Điểm
                "features": features
            }
        # Ngưỡng thụt lề >= 15px tương đương thụt lề cấp 1 (Khoản)
        elif left_indent >= 15.0:
            return {
                "predicted_level": LayoutHierarchyLevel.SUB_ITEM,
                "confidence": 0.88,
                "is_heading": False,
                "indent_level": 1,  # Tương đương cấp Khoản
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
        Duyệt và gắn nhãn phân cấp hình học cho toàn bộ danh sách dòng của một trang.
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
