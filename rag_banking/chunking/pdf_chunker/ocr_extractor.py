"""
Bộ trích xuất ký tự quang học (OCR Extractor) tối ưu cho tiếng Việt
Sử dụng Tesseract OCR v5 kết hợp xử lý ảnh OpenCV:
1. Render PDF 300 DPI bằng PyMuPDF.
2. Lọc con dấu mộc đỏ ngân hàng bằng không gian màu HSV.
3. Cân thẳng góc nghiêng (deskew) trang scan.
4. Trích xuất văn bản tiếng Việt/Anh (vie+eng), tính điểm tin cậy OCR và cấu trúc dòng.
"""

from __future__ import annotations

import os
from pathlib import Path
import cv2
import numpy as np
import pymupdf as fitz
import pytesseract


def _find_default_tesseract_cmd() -> str:
    """Tự động xác định đường dẫn thực thi Tesseract OCR v5."""
    candidates = [
        os.environ.get("TESSERACT_CMD", ""),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "tesseract",
    ]
    for cand in candidates:
        if cand and (Path(cand).is_file() or cand == "tesseract"):
            return cand
    return "tesseract"


class OCRExtractor:
    """Trình trích xuất OCR cục bộ cho tài liệu tiếng Việt với Tesseract OCR v5 và OpenCV."""

    def __init__(self, tesseract_cmd: str | None = None, lang: str = "vie+eng"):
        self.tesseract_cmd = tesseract_cmd or _find_default_tesseract_cmd()
        self.lang = lang
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

    def is_available(self) -> bool:
        """Kiểm tra Tesseract OCR v5 đã sẵn sàng hay chưa."""
        try:
            _ = pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def get_active_engine_name(self) -> str:
        """Tên engine OCR đang hoạt động."""
        return "Tesseract OCR v5"

    def filter_red_stamp(self, image_np: np.ndarray) -> np.ndarray:
        """Lọc bớt màu đỏ của con dấu mộc để làm rõ nét chữ đen bên dưới."""
        if image_np is None or len(image_np.shape) == 2:
            return image_np

        try:
            hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
            red_mask = cv2.bitwise_or(mask1, mask2)

            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            cleaned_gray = gray.copy()
            cleaned_gray[red_mask > 0] = np.clip(cleaned_gray[red_mask > 0].astype(np.int32) + 60, 0, 255).astype(np.uint8)
            return cv2.cvtColor(cleaned_gray, cv2.COLOR_GRAY2BGR)
        except Exception:
            return image_np

    def deskew_image(self, image_np: np.ndarray) -> np.ndarray:
        """Căn chỉnh góc nghiêng (deskew) cho ảnh scan văn bản."""
        if image_np is None:
            return image_np

        try:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) if len(image_np.shape) == 3 else image_np
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            coords = np.column_stack(np.where(thresh > 0))
            if len(coords) < 150:
                return image_np

            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            elif angle > 45:
                angle = 90 - angle
            else:
                angle = -angle

            if 0.5 < abs(angle) < 15.0:
                h, w = image_np.shape[:2]
                m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                return cv2.warpAffine(
                    image_np, m, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255) if len(image_np.shape) == 3 else 255
                )
            return image_np
        except Exception:
            return image_np

    def render_pdf_page_300dpi(self, pdf_path: str | Path, page_idx: int = 0) -> np.ndarray | None:
        """Render trang PDF ở độ phân giải 300 DPI thành ảnh OpenCV BGR."""
        try:
            doc = fitz.open(str(pdf_path))
            if page_idx < len(doc):
                pix = doc[page_idx].get_pixmap(dpi=300)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
                if pix.n == 4:
                    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                elif pix.n == 3:
                    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif pix.n == 1:
                    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            print(f"[OCR] Lỗi render trang PDF {page_idx}: {e}")
        return None

    def extract_text_and_confidence(self, image_input: str | Path | np.ndarray) -> tuple[str, float]:
        """Trích xuất văn bản và tính điểm tin cậy trung bình của trang."""
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists():
                return "", 0.0
            img_np = cv2.imread(str(p))
        else:
            img_np = image_input

        if img_np is None:
            return "", 0.0

        img_np = self.filter_red_stamp(img_np)
        try:
            data = pytesseract.image_to_data(img_np, lang=self.lang, output_type=pytesseract.Output.DICT)
            words = []
            scores = []
            for i in range(len(data["text"])):
                w = str(data["text"][i]).strip()
                conf = float(data["conf"][i])
                if conf >= 0 and w:
                    words.append(w)
                    scores.append(conf / 100.0)

            avg_score = round(sum(scores) / max(len(scores), 1), 3) if scores else 0.85
            text_out = " ".join(words)
            return text_out, avg_score
        except Exception as e:
            print(f"[OCR] Lỗi Tesseract OCR: {e}")
            return "", 0.0

    def extract_layout_lines(
        self,
        image_input: str | Path | np.ndarray,
        apply_deskew: bool = True,
        apply_red_stamp_filter: bool = True
    ) -> list[dict]:
        """Trích xuất từng dòng kèm tọa độ bounding box và sắp xếp theo thứ tự đọc."""
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists():
                return []
            img_np = cv2.imread(str(p))
        else:
            img_np = image_input

        if img_np is None:
            return []

        if apply_deskew:
            img_np = self.deskew_image(img_np)
        if apply_red_stamp_filter:
            img_np = self.filter_red_stamp(img_np)

        page_h, page_w = img_np.shape[:2]

        try:
            data = pytesseract.image_to_data(img_np, lang=self.lang, output_type=pytesseract.Output.DICT)
        except Exception as e:
            print(f"[OCR] Lỗi trích xuất layout Tesseract OCR: {e}")
            return []

        lines_map: dict[tuple[int, int, int], list[dict]] = {}
        for i in range(len(data["text"])):
            w = str(data["text"][i]).strip()
            conf = float(data["conf"][i])
            if conf >= 0 and w:
                key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                if key not in lines_map:
                    lines_map[key] = []
                lines_map[key].append({
                    "text": w,
                    "conf": conf,
                    "left": float(data["left"][i]),
                    "top": float(data["top"][i]),
                    "width": float(data["width"][i]),
                    "height": float(data["height"][i])
                })

        lines_data: list[dict] = []
        for words in lines_map.values():
            if not words:
                continue
            line_text = " ".join(w["text"] for w in words)
            x_min = float(min(w["left"] for w in words))
            y_min = float(min(w["top"] for w in words))
            x_max = float(max(w["left"] + w["width"] for w in words))
            y_max = float(max(w["top"] + w["height"] for w in words))
            scores = [w["conf"] / 100.0 for w in words]
            score = round(sum(scores) / len(scores), 3)

            box = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]

            lines_data.append({
                "text": line_text,
                "score": score,
                "box": box,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "width": max(x_max - x_min, 1.0),
                "height": max(y_max - y_min, 1.0),
                "center_x": (x_min + x_max) / 2.0,
                "center_y": (y_min + y_max) / 2.0,
                "page_width": float(page_w),
                "page_height": float(page_h)
            })

        if not lines_data:
            return []

        return self._sort_reading_order(lines_data, page_w, page_h)

    def _sort_reading_order(self, lines: list[dict], page_w: float, page_h: float) -> list[dict]:
        """Sắp xếp các dòng theo thứ tự đọc tự nhiên (hỗ trợ bố cục 1 cột và 2 cột)."""
        if len(lines) <= 3:
            return sorted(lines, key=lambda l: (l["y_min"], l["x_min"]))

        mid_x = page_w * 0.50
        left_boxes = [b for b in lines if b["x_max"] < mid_x * 1.08]
        right_boxes = [b for b in lines if b["x_min"] > mid_x * 0.92]
        crossing_boxes = [b for b in lines if b["x_min"] < mid_x < b["x_max"] and b["width"] > page_w * 0.4]

        is_two_columns = (
            len(left_boxes) >= len(lines) * 0.25 and
            len(right_boxes) >= len(lines) * 0.25 and
            len(crossing_boxes) <= len(lines) * 0.15
        )

        if is_two_columns:
            top_headers = sorted([b for b in crossing_boxes if b["y_min"] < page_h * 0.25], key=lambda b: b["y_min"])
            other_crossing = [b for b in crossing_boxes if b not in top_headers]
            left_sorted = sorted(left_boxes, key=lambda b: b["y_min"])
            right_sorted = sorted(right_boxes, key=lambda b: b["y_min"])
            bottom_items = sorted(other_crossing, key=lambda b: b["y_min"])
            return top_headers + left_sorted + right_sorted + bottom_items

        median_h = float(np.median([l["height"] for l in lines])) if lines else 15.0
        tolerance = max(median_h * 0.6, 6.0)

        rows: list[list[dict]] = []
        for line in sorted(lines, key=lambda l: l["y_min"]):
            if rows and abs(line["y_min"] - rows[-1][0]["y_min"]) < tolerance:
                rows[-1].append(line)
            else:
                rows.append([line])

        return [line for row in rows for line in sorted(row, key=lambda l: l["x_min"])]

    def extract_text_from_pdf_page(self, pdf_path: str | Path, page_idx: int = 0) -> tuple[str, float]:
        """Render và OCR một trang PDF, trả về văn bản và độ tin cậy."""
        img_np = self.render_pdf_page_300dpi(pdf_path, page_idx=page_idx)
        if img_np is not None:
            return self.extract_text_and_confidence(img_np)
        return "", 0.0
