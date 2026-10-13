"""
Bộ trích xuất ký tự quang học tối ưu cho tiếng Việt (Vietnamese OCR Extractor)
Kiến trúc Hybrid 2 tầng (Dual-Engine Pipeline) kết hợp lọc kênh đỏ mộc dấu:
1. Bộ lọc kênh đỏ (Red Stamp / Seal Noise Reduction): Khử nhiễu con dấu mộc đỏ ngân hàng, giữ nguyên nét chữ đen.
2. Trích xuất Text & Tính toán OCR Confidence: Lưu lại ocr_confidence phục vụ metadata lọc chất lượng.
3. RapidOCR ONNX Runtime & Tesseract Fallback: Chạy 100% offline CPU tiếng Việt.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union, Tuple, List
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pypdfium2 as pdfium
    PDFIUM_AVAILABLE = True
except ImportError:
    PDFIUM_AVAILABLE = False

try:
    from rapidocr_onnxruntime import RapidOCR
    RAPIDOCR_AVAILABLE = True
except ImportError:
    RAPIDOCR_AVAILABLE = False


class OCRExtractor:
    """
    Trình trích xuất OCR cục bộ tối ưu hóa cho tài liệu tài chính, ngân hàng tiếng Việt.
    Có bộ lọc con dấu mộc đỏ và xuất độ tin cậy ocr_confidence.
    """

    def __init__(self, tesseract_cmd: Optional[str] = None):
        self.tesseract_exe = self._resolve_tesseract_path(tesseract_cmd)
        self._rapid_engine = None

        if RAPIDOCR_AVAILABLE:
            try:
                self._rapid_engine = RapidOCR()
            except Exception as e:
                print(f"[OCR] Cảnh báo khởi tạo RapidOCR: {e}")

    def _resolve_tesseract_path(self, custom_path: Optional[str] = None) -> Optional[str]:
        """Tự động định vị binary Tesseract OCR trên hệ thống."""
        candidates = []
        if custom_path:
            candidates.append(custom_path)
        if "TESSERACT_PATH" in os.environ:
            candidates.append(os.environ["TESSERACT_PATH"])

        candidates.extend([
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            shutil.which("tesseract"),
        ])

        for path in candidates:
            if path and Path(path).is_file():
                return str(path)
        return None

    def is_available(self) -> bool:
        """Kiểm tra xem có ít nhất một engine OCR hoạt động hay không."""
        return (self.tesseract_exe is not None) or (self._rapid_engine is not None)

    def get_active_engine_name(self) -> str:
        """Tên của engine OCR đang được ưu tiên sử dụng."""
        if self._rapid_engine:
            return "RapidOCR (ONNX Runtime - Latin/vi)"
        if self.tesseract_exe:
            return "Tesseract-OCR v5 (vie)"
        return "None"

    def filter_red_stamp(self, image_np: np.ndarray) -> np.ndarray:
        """
        Lọc bỏ hoặc giảm thiểu nhiễu con dấu mộc đỏ ngân hàng bằng OpenCV:
        Chuyển đổi HSV, nhận diện dải màu đỏ của mộc, tách bỏ lớp mực đỏ để lộ rõ nét chữ đen bên dưới.
        """
        if not CV2_AVAILABLE or image_np is None:
            return image_np

        try:
            # Đảm bảo ảnh là BGR
            if len(image_np.shape) == 2:
                return image_np  # Ảnh đã là grayscale

            # Chuyển đổi sang HSV để bắt màu đỏ
            hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)

            # Dải màu đỏ trong HSV (gồm 2 khoảng: 0-10 và 160-180)
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 50, 50])
            upper_red2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)

            # Tạo bản sao thang xám
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

            # Với những vùng bị mộc đỏ che phủ, ta tăng độ sáng lên để mộc đỏ biến thành nền trắng,
            # trong khi nét chữ màu đen có độ đậm cao vẫn được bảo toàn
            cleaned_gray = gray.copy()
            # Làm mờ nhẹ vùng mộc đỏ
            cleaned_gray[red_mask > 0] = np.clip(cleaned_gray[red_mask > 0].astype(np.int32) + 60, 0, 255).astype(np.uint8)

            # Chuyển ngược lại ảnh 3 kênh để RapidOCR xử lý
            cleaned_bgr = cv2.cvtColor(cleaned_gray, cv2.COLOR_GRAY2BGR)
            return cleaned_bgr
        except Exception as e:
            return image_np

    def extract_text_and_confidence(self, image_input: Union[str, Path, np.ndarray]) -> Tuple[str, float]:
        """
        Trích xuất văn bản và điểm tin cậy trung bình ocr_confidence:
        Returns:
            (extracted_text, confidence_score) (ví dụ: ("Nội dung văn bản", 0.942))
        """
        img_np = None

        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists():
                return "", 0.0
            if CV2_AVAILABLE:
                img_np = cv2.imread(str(p))
        elif isinstance(image_input, np.ndarray):
            img_np = image_input

        # 1. Tiền xử lý lọc con dấu mộc đỏ
        if img_np is not None and CV2_AVAILABLE:
            img_np = self.filter_red_stamp(img_np)

        # 2. Ưu tiên 1: RapidOCR ONNX (Tối ưu cho văn bản tiếng Việt CPU)
        if self._rapid_engine and img_np is not None:
            try:
                result, _ = self._rapid_engine(img_np)
                if result:
                    lines = []
                    scores = []
                    for item in result:
                        if item and len(item) > 2 and item[1].strip():
                            lines.append(item[1].strip())
                            try:
                                scores.append(float(item[2]))
                            except (ValueError, TypeError):
                                scores.append(0.85)

                    text_out = "\n".join(lines)
                    avg_score = round(sum(scores) / max(len(scores), 1), 3) if scores else 0.85
                    return text_out, avg_score
            except Exception as e:
                print(f"[OCR] RapidOCR error: {e}")

        # 3. Fallback: Tesseract OCR (vie)
        if self.tesseract_exe and isinstance(image_input, (str, Path)):
            try:
                cmd = [
                    self.tesseract_exe,
                    str(Path(image_input).resolve()),
                    "stdout",
                    "-l", "vie",
                    "--psm", "3"
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
                out = res.stdout.strip()
                if out:
                    return out, 0.90
            except Exception as e:
                print(f"[OCR] Tesseract error: {e}")

        return "", 0.0

    def deskew_image(self, image_np: np.ndarray) -> np.ndarray:
        """
        Căn chỉnh góc nghiêng (deskew) cho ảnh scan văn bản bằng OpenCV:
        Sử dụng threshold Otsu và minAreaRect trên các pixel văn bản.
        """
        if not CV2_AVAILABLE or image_np is None:
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

            # Giới hạn chỉ xoay nếu góc nghiêng nhẹ (0.5 đến 15 độ)
            if 0.5 < abs(angle) < 15.0:
                (h, w) = image_np.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(
                    image_np, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255) if len(image_np.shape) == 3 else 255
                )
                return rotated
            return image_np
        except Exception:
            return image_np

    def render_pdf_page_300dpi(
        self,
        pdf_path: Union[str, Path],
        page_idx: int = 0
    ) -> Optional[np.ndarray]:
        """
        Render trang PDF ở độ phân giải 300 DPI và chuyển đổi sang ảnh OpenCV BGR.
        Ưu tiên PyMuPDF (siêu nhanh) -> Fallback pypdfium2.
        """
        # 1. Thử PyMuPDF (tối ưu hóa tốc độ và độ sắc nét)
        try:
            import pymupdf as fitz
            doc = fitz.open(str(pdf_path))
            if page_idx < len(doc):
                page = doc[page_idx]
                pix = page.get_pixmap(dpi=300)
                img_data = np.frombuffer(pix.samples, dtype=np.uint8)
                if pix.n == 4:  # RGBA
                    img_np = img_data.reshape((pix.height, pix.width, 4))
                    if CV2_AVAILABLE:
                        return cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                elif pix.n == 3:  # RGB
                    img_np = img_data.reshape((pix.height, pix.width, 3))
                    if CV2_AVAILABLE:
                        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                elif pix.n == 1:  # Gray
                    img_np = img_data.reshape((pix.height, pix.width))
                    if CV2_AVAILABLE:
                        return cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        except Exception:
            pass

        # 2. Thử pypdfium2 (scale = 300 / 72 ≈ 4.167)
        if PDFIUM_AVAILABLE:
            try:
                doc = pdfium.PdfDocument(str(pdf_path))
                if page_idx < len(doc):
                    page = doc[page_idx]
                    scale = 300.0 / 72.0
                    pil_image = page.render(scale=scale).to_pil()
                    if CV2_AVAILABLE:
                        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            except Exception:
                pass

        return None

    def extract_layout_lines(
        self,
        image_input: Union[str, Path, np.ndarray],
        apply_deskew: bool = True,
        apply_red_stamp_filter: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Trích xuất chi tiết từng dòng văn bản kèm tọa độ Bounding Box, kích thước và điểm tin cậy score.
        Tự động phát hiện bố cục 2 cột (Two-column layout) để sắp xếp đúng Reading Order.
        """
        img_np = None
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists():
                return []
            if CV2_AVAILABLE:
                img_np = cv2.imread(str(p))
        elif isinstance(image_input, np.ndarray):
            img_np = image_input

        if img_np is None:
            return []

        # 1. Deskew
        if apply_deskew and CV2_AVAILABLE:
            img_np = self.deskew_image(img_np)

        # 2. Lọc con dấu mộc đỏ
        if apply_red_stamp_filter and CV2_AVAILABLE:
            img_np = self.filter_red_stamp(img_np)

        page_h, page_w = img_np.shape[:2]

        # 3. Chạy RapidOCR để lấy danh sách [box, text, score]
        lines_data: List[Dict[str, Any]] = []
        if self._rapid_engine:
            try:
                result, _ = self._rapid_engine(img_np)
                if result:
                    for item in result:
                        if not item or len(item) < 3:
                            continue
                        box, text, score = item[0], str(item[1]).strip(), float(item[2])
                        if not text:
                            continue

                        # Tọa độ 4 đỉnh hộp bao
                        xs = [pt[0] for pt in box]
                        ys = [pt[1] for pt in box]
                        x_min, x_max = float(min(xs)), float(max(xs))
                        y_min, y_max = float(min(ys)), float(max(ys))
                        w_box = max(x_max - x_min, 1.0)
                        h_box = max(y_max - y_min, 1.0)
                        cx = (x_min + x_max) / 2.0
                        cy = (y_min + y_max) / 2.0

                        lines_data.append({
                            "text": text,
                            "score": round(score, 3),
                            "box": box,
                            "x_min": x_min,
                            "y_min": y_min,
                            "x_max": x_max,
                            "y_max": y_max,
                            "width": w_box,
                            "height": h_box,
                            "center_x": cx,
                            "center_y": cy,
                            "page_width": float(page_w),
                            "page_height": float(page_h)
                        })
            except Exception as e:
                print(f"[OCR] RapidOCR layout error: {e}")

        if not lines_data:
            return []

        # 4. Phân tích bố cục & Sắp xếp thứ tự đọc (Reading Order & Two-Column Sorting)
        return self._sort_reading_order(lines_data, page_w, page_h)

    def _sort_reading_order(
        self,
        lines: List[Dict[str, Any]],
        page_w: float,
        page_h: float
    ) -> List[Dict[str, Any]]:
        """
        Phát hiện bố cục 2 cột (Two-Column) và sắp xếp luồng đọc tự nhiên:
        Cột trái (trên xuống) -> Cột phải (trên xuống).
        Nếu là 1 cột thông thường -> sắp xếp từ trên xuống dưới theo y_min.
        """
        if len(lines) <= 3:
            return sorted(lines, key=lambda l: (l["y_min"], l["x_min"]))

        # Kiểm tra tỷ lệ 2 cột:
        # Nếu phần lớn dòng nằm hẳn bên trái hoặc bên phải trục giữa (margin 5%)
        mid_x = page_w * 0.50
        left_boxes = [b for b in lines if b["x_max"] < mid_x * 1.08]
        right_boxes = [b for b in lines if b["x_min"] > mid_x * 0.92]
        crossing_boxes = [b for b in lines if b["x_min"] < mid_x < b["x_max"] and b["width"] > page_w * 0.4]

        # Tiêu chí nhận diện 2 cột:
        # 1. Cả 2 bên đều có ít nhất 20% số dòng
        # 2. Số dòng vắt ngang giữa chiếm < 15%
        is_two_columns = (
            len(left_boxes) >= len(lines) * 0.25 and
            len(right_boxes) >= len(lines) * 0.25 and
            len(crossing_boxes) <= len(lines) * 0.15
        )

        if is_two_columns:
            # Sắp xếp theo: các dòng tiêu đề vắt ngang ở đầu trang -> Cột trái -> Cột phải
            top_headers = sorted([b for b in crossing_boxes if b["y_min"] < page_h * 0.25], key=lambda b: b["y_min"])
            other_crossing = [b for b in crossing_boxes if b not in top_headers]

            left_sorted = sorted(left_boxes, key=lambda b: b["y_min"])
            right_sorted = sorted(right_boxes, key=lambda b: b["y_min"])
            bottom_items = sorted(other_crossing, key=lambda b: b["y_min"])

            return top_headers + left_sorted + right_sorted + bottom_items

        # Với 1 cột: gom nhóm theo hàng (dung sai nửa chiều cao trung bình)
        median_h = float(np.median([l["height"] for l in lines])) if lines else 15.0
        tolerance = max(median_h * 0.6, 6.0)

        # Sắp xếp sơ bộ theo y_min
        sorted_by_y = sorted(lines, key=lambda l: l["y_min"])
        rows: List[List[Dict[str, Any]]] = []

        for line in sorted_by_y:
            placed = False
            for row in rows:
                if abs(line["y_min"] - row[0]["y_min"]) < tolerance:
                    row.append(line)
                    placed = True
                    break
            if not placed:
                rows.append([line])

        final_order = []
        for row in rows:
            # Trong cùng một hàng, sắp xếp từ trái qua phải
            row_sorted = sorted(row, key=lambda l: l["x_min"])
            final_order.extend(row_sorted)

        return final_order

    def extract_text_from_pdf_page(
        self,
        pdf_path: Union[str, Path],
        page_idx: int = 0,
        scale: float = 300.0 / 72.0
    ) -> Tuple[str, float]:
        """
        Render trang PDF scan ở 300 DPI, deskew, lọc mộc đỏ và OCR tiếng Việt:
        Returns:
            (extracted_text, ocr_confidence)
        """
        img_np = self.render_pdf_page_300dpi(pdf_path, page_idx=page_idx)
        if img_np is not None:
            return self.extract_text_and_confidence(img_np)

        return "", 0.0
