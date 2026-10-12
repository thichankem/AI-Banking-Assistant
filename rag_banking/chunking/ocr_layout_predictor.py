"""
OCR Layout Hierarchy Predictor (Bộ dự đoán Cấp bậc Bố cục bằng Hình học Thị giác OCR)
Tuyệt đối KHÔNG dùng Regex để phân cấp (0% Regex-based hierarchy).

Dự đoán cấp bậc dựa trên 5 chiều không gian hình học từ Bounding Box của OCR:
1. Relative Font Height: Tỷ lệ chiều cao dòng so với chiều cao trung vị thân bài (median_height).
2. Vertical Whitespace: Khoảng cách dọc trước và sau dòng (gap_before, gap_after).
3. Horizontal Alignment: Căn giữa trang (is_centered) vs Bám lề trái (base_left).
4. Left Indentation: Độ thụt lề ngang (x_min - base_left) để nhận diện Khoản và Điểm.
5. Line Aspect & Compactness: Tỷ lệ độ dài dòng so với bề ngang trang (rel_width).
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class LayoutHierarchyLevel:
    TITLE = "title"           # Level 0: Tiêu đề văn bản / Văn bản mới
    H1 = "h1"                 # Level 1: Tiêu đề lớn (Chương / Phần / Heading 1)
    H2 = "h2"                 # Level 2: Tiêu đề mục (Mục / Điều / Heading 2)
    H3 = "h3"                 # Level 3: Tiêu đề tiểu mục (Heading 3)
    SUB_ITEM = "sub_item"     # Level 4: Thụt lề cấp 1 (Khoản / Danh sách)
    DEEP_ITEM = "deep_item"   # Level 5: Thụt lề cấp 2 (Điểm / Danh sách con)
    BODY = "body"             # Thân bài thông thường
    HEADER_FOOTER = "header_footer"  # Vùng lề trang (Header/Footer/Số trang)


class OCRLayoutHierarchyPredictor:
    """
    Bộ phân loại cấp bậc phân đoạn thuần túy bằng Bố cục Hình học OCR (0% Regex).
    """

    def __init__(
        self,
        edge_margin_ratio: float = 0.08,    # 8% trên/dưới coi là header/footer
        center_tolerance_ratio: float = 0.12 # Dung sai căn giữa (+-12% trang)
    ):
        self.edge_margin_ratio = edge_margin_ratio
        self.center_tolerance_ratio = center_tolerance_ratio

    def compute_page_layout_stats(
        self,
        lines: List[Dict[str, Any]],
        page_w: float,
        page_h: float
    ) -> Dict[str, float]:
        """
        Tính toán các chỉ số thống kê bố cục cơ sở của toàn bộ trang:
        - median_height: Chiều cao cơ sở của chữ thân bài (Body Text baseline).
        - base_left: Tọa độ lề trái chuẩn của khối văn bản.
        - median_line_gap: Giãn dòng trung vị trong cùng đoạn văn.
        """
        if not lines:
            return {"median_h": 16.0, "base_left": 50.0, "median_gap": 8.0}

        # 1. Lọc bỏ các dòng ở sát biên trang để không làm nhiễu thống kê
        content_lines = [
            l for l in lines
            if (l.get("y_min", 0) >= page_h * self.edge_margin_ratio) and
               (l.get("y_max", page_h) <= page_h * (1.0 - self.edge_margin_ratio))
        ]
        target_lines = content_lines if len(content_lines) >= 3 else lines

        # 2. Chiều cao trung vị
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
        line: Dict[str, Any],
        stats: Dict[str, float],
        page_w: float,
        page_h: float,
        prev_line: Optional[Dict[str, Any]] = None,
        next_line: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
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
        h = float(line.get("height", 16.0))
        w = float(line.get("width", 100.0))
        x_min = float(line.get("x_min", stats.get("base_left", 50.0)))
        has_explicit_y = "y_min" in line
        y_min = float(line.get("y_min", page_h * 0.20))
        y_max = float(line.get("y_max", y_min + h))
        cx = float(line.get("center_x", (x_min + w / 2.0)))

        median_h = stats.get("median_h", 16.0)
        base_left = stats.get("base_left", 50.0)
        median_gap = stats.get("median_gap", 8.0)

        # -------------------------------------------------------------
        # 1. Kiểm tra Vùng Mép Trang (Header/Footer Boundary)
        # -------------------------------------------------------------
        if has_explicit_y and (y_min < page_h * self.edge_margin_ratio or y_max > page_h * (1.0 - self.edge_margin_ratio)):
            # Nếu chữ rất to ở đầu trang 1 thì có thể là Title, còn lại là header/footer
            if y_min < page_h * self.edge_margin_ratio and h >= 1.35 * median_h:
                pass  # Cho phép xét Title
            else:
                return {
                    "predicted_level": LayoutHierarchyLevel.HEADER_FOOTER,
                    "confidence": 0.95,
                    "is_heading": False,
                    "indent_level": 0,
                    "features": {"edge": True}
                }

        # -------------------------------------------------------------
        # 2. Tính toán các đặc trưng bố cục chuẩn hóa
        # -------------------------------------------------------------
        rel_h = h / median_h
        rel_w = w / page_w
        is_centered = abs(cx - page_w / 2.0) < (page_w * self.center_tolerance_ratio)
        left_indent = max(x_min - base_left, 0.0)
        is_short_line = rel_w < 0.72

        # Khoảng cách dọc trước dòng
        gap_before = (y_min - prev_line.get("y_max", y_min)) if prev_line else (median_gap * 2.0)
        has_break_before = gap_before >= (median_gap * 1.35)

        # Tín hiệu bề mặt (in hoa toàn bộ)
        is_uppercase = text.isupper() and len(text) >= 3

        features = {
            "rel_h": round(rel_h, 2),
            "rel_w": round(rel_w, 2),
            "is_centered": is_centered,
            "left_indent": round(left_indent, 1),
            "gap_before": round(gap_before, 1),
            "is_uppercase": is_uppercase
        }

        # -------------------------------------------------------------
        # 3. Phân loại theo Ma trận Bố cục Hình học (0% Regex)
        # -------------------------------------------------------------

        # A. Cấp 0: Tiêu đề lớn nhất tài liệu (Title / Document Cover)
        if rel_h >= 1.35 and (is_centered or is_short_line):
            return {
                "predicted_level": LayoutHierarchyLevel.TITLE,
                "confidence": 0.95,
                "is_heading": True,
                "indent_level": 0,
                "features": features
            }

        # B. Cấp 1: Tiêu đề phân đoạn lớn (Major Heading / Chương / Phần)
        if rel_h >= 1.18 or (is_centered and is_uppercase and has_break_before):
            return {
                "predicted_level": LayoutHierarchyLevel.H1,
                "confidence": 0.92,
                "is_heading": True,
                "indent_level": 0,
                "features": features
            }

        # C. Cấp 2: Tiêu đề mục (Section Heading / Mục / Điều)
        if (1.08 <= rel_h < 1.18 and is_short_line) or (is_uppercase and is_short_line and has_break_before):
            return {
                "predicted_level": LayoutHierarchyLevel.H2,
                "confidence": 0.90,
                "is_heading": True,
                "indent_level": 0,
                "features": features
            }

        # D. Cấp 3: Tiêu đề tiểu mục (Subsection Heading / Dòng ngắn ngắt đoạn rõ rệt)
        if is_short_line and has_break_before and rel_w < 0.60 and not text.endswith((".", ",", ";", ":")):
            return {
                "predicted_level": LayoutHierarchyLevel.H3,
                "confidence": 0.85,
                "is_heading": True,
                "indent_level": 0,
                "features": features
            }

        # E. Cấp 4 & 5: Đoạn văn thụt lề (Indented Clauses / Khoản và Điểm)
        if left_indent >= 45.0:
            return {
                "predicted_level": LayoutHierarchyLevel.DEEP_ITEM,
                "confidence": 0.88,
                "is_heading": False,
                "indent_level": 2,  # Tương đương cấp Điểm
                "features": features
            }
        elif left_indent >= 18.0:
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
        lines: List[Dict[str, Any]],
        page_w: float,
        page_h: float
    ) -> List[Dict[str, Any]]:
        """
        Duyệt và gán nhãn phân cấp cho toàn bộ danh sách dòng của một trang.
        """
        if not lines:
            return []

        stats = self.compute_page_layout_stats(lines, page_w, page_h)
        annotated_lines = []

        for i, line in enumerate(lines):
            prev_l = lines[i - 1] if i > 0 else None
            next_l = lines[i + 1] if i + 1 < len(lines) else None

            pred = self.predict_line_hierarchy(
                line=line,
                stats=stats,
                page_w=page_w,
                page_h=page_h,
                prev_line=prev_l,
                next_line=next_l
            )
            # Gộp nhãn dự đoán vào đối tượng dòng
            annotated = dict(line)
            annotated["predicted_level"] = pred["predicted_level"]
            annotated["is_heading"] = pred["is_heading"]
            annotated["indent_level"] = pred["indent_level"]
            annotated["layout_confidence"] = pred["confidence"]
            annotated["layout_features"] = pred["features"]
            annotated_lines.append(annotated)

        return annotated_lines
