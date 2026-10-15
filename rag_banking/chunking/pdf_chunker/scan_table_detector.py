"""
Module Scan Table Detector (Phát hiện và Trích xuất Vùng Bảng trên Trang Scan)
=============================================================================
Giải quyết triệt để vấn đề: PyMuPDF find_tables và pdfplumber bị mù trên PDF Scan
do không có vector graphics hay text stream số học.

Cơ chế hoạt động:
1. OpenCV Mathematical Morphology (Geometric Table Detection):
   - Phân tích hình thái học nhị phân (Binary Morphology) để trích xuất hệ thống đường kẻ
     ngang (Horizontal Kernels) và đường kẻ dọc (Vertical Kernels).
   - Tổng hợp mặt nạ lưới bảng (Table Grid Lattice Mask) và xác định giao điểm ô (Intersections).
   - Lọc bỏ đường gạch chân văn bản đơn lẻ, chỉ giữ lại các cấu trúc có cấu trúc ô/bảng thực tế.
2. Interface cắm mô hình nhẹ (YOLO DocLayNet / Table Transformer) dự phòng khi có sẵn weights.
3. Cắt (Crop) chính xác vùng ảnh bảng kèm padding bảo vệ để chuyển sang LocalVisionTableExtractor.
4. Xóa che (Masking) vùng bảng trên trang scan để bộ OCR dòng không bị nhiễu bởi lưới ô.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pymupdf as fitz

logger = logging.getLogger("scan_table_detector")


@dataclass
class DetectedTableBox:
    """Đại diện cho một vùng bảng biểu được phát hiện trên ảnh scan."""
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float = 0.90
    detection_method: str = "opencv_morphology"
    cell_count_est: int = 0
    area_ratio: float = 0.0
    cropped_image: np.ndarray | None = field(default=None, repr=False)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Tọa độ (x0, y0, x1, y1)."""
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def width(self) -> float:
        return max(self.x1 - self.x0, 1.0)

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": [round(self.x0, 1), round(self.y0, 1), round(self.x1, 1), round(self.y1, 1)],
            "width": round(self.width, 1),
            "height": round(self.height, 1),
            "confidence": round(self.confidence, 3),
            "detection_method": self.detection_method,
            "cell_count_est": self.cell_count_est,
            "area_ratio": round(self.area_ratio, 4)
        }


class ScanTableDetector:
    """
    Bộ dò tìm và cắt vùng bảng trên trang PDF scan bằng OpenCV Contour & Mathematical Morphology.
    Hoàn toàn cục bộ, hiệu năng cao (~10-20ms/trang), không phụ thuộc mạng.
    """

    def __init__(
        self,
        min_area_ratio: float = 0.03,         # Bảng tối thiểu chiếm 3% diện tích trang
        min_width_ratio: float = 0.25,        # Bảng tối thiểu chiếm 25% chiều rộng trang
        min_height_ratio: float = 0.04,       # Bảng tối thiểu chiếm 4% chiều cao trang
        horizontal_scale: int = 35,           # Hệ số độ dài kernel ngang (w / scale)
        vertical_scale: int = 35,             # Hệ số độ dài kernel dọc (h / scale)
        line_thickness_max: int = 8,          # Độ dày tối đa của nét kẻ ô bảng
        padding: int = 12,                    # Khoảng đệm px khi crop ảnh bảng
        external_model_fn: Callable[[np.ndarray], list[DetectedTableBox]] | None = None
    ):
        self.min_area_ratio = min_area_ratio
        self.min_width_ratio = min_width_ratio
        self.min_height_ratio = min_height_ratio
        self.horizontal_scale = horizontal_scale
        self.vertical_scale = vertical_scale
        self.line_thickness_max = line_thickness_max
        self.padding = padding
        self.external_model_fn = external_model_fn

    def detect_tables(self, image_np: np.ndarray) -> list[DetectedTableBox]:
        """
        Dò tìm toàn bộ các bảng biểu trên ảnh scan:
        Ưu tiên sử dụng mô hình học sâu nếu được cấu hình, mặc định chạy OpenCV Contour Morphology.
        """
        if image_np is None or image_np.size == 0:
            return []

        # 1. Gọi external model nếu người dùng cung cấp
        if self.external_model_fn is not None:
            try:
                ext_results = self.external_model_fn(image_np)
                if ext_results:
                    return ext_results
            except Exception as e:
                logger.warning(f"[ScanTableDetector] Lỗi khi chạy mô hình ngoài: {e}. Fallback OpenCV.")

        # 2. Thuật toán hình thái học OpenCV thuần túy
        return self._detect_tables_opencv(image_np)

    def _detect_tables_opencv(self, image_np: np.ndarray) -> list[DetectedTableBox]:
        """Thuật toán phân tích hình học đường kẻ ô bảng qua Morphology."""
        h, w = image_np.shape[:2]
        page_area = float(h * w)

        # Chuyển sang ảnh xám
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) if len(image_np.shape) == 3 else image_np.copy()

        # Nhị phân hóa Otsu đảo ngược (nền đen, chữ/nét kẻ trắng)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # 1. Trích xuất đường kẻ ngang
        horiz_kernel_len = max(w // self.horizontal_scale, 20)
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_kernel_len, 1))
        horiz_lines = cv2.erode(thresh, horiz_kernel, iterations=1)
        horiz_lines = cv2.dilate(horiz_lines, horiz_kernel, iterations=1)

        # 2. Trích xuất đường kẻ dọc
        vert_kernel_len = max(h // self.vertical_scale, 20)
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_kernel_len))
        vert_lines = cv2.erode(thresh, vert_kernel, iterations=1)
        vert_lines = cv2.dilate(vert_lines, vert_kernel, iterations=1)

        # 3. Kết hợp khung lưới bảng
        table_grid = cv2.addWeighted(horiz_lines, 0.5, vert_lines, 0.5, 0.0)
        table_grid = cv2.threshold(table_grid, 50, 255, cv2.THRESH_BINARY)[1]

        # Giãn nhẹ để nối các mối nối góc ô
        joint_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        table_grid = cv2.dilate(table_grid, joint_kernel, iterations=2)

        # 4. Tìm các đường bao (Contours) ứng viên
        contours, _ = cv2.findContours(table_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_tables: list[DetectedTableBox] = []

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            box_area = float(bw * bh)
            area_ratio = box_area / page_area
            w_ratio = bw / float(w)
            h_ratio = bh / float(h)

            # Lọc theo diện tích, chiều ngang và chiều dọc tối thiểu
            if area_ratio < self.min_area_ratio:
                continue
            if w_ratio < self.min_width_ratio:
                continue
            if h_ratio < self.min_height_ratio:
                continue

            # Kiểm tra mật độ ô / giao điểm bên trong ROI để tránh nhầm với khung tranh / đường phân cách
            roi_grid = table_grid[y:y + bh, x:x + bw]
            roi_horiz = horiz_lines[y:y + bh, x:x + bw]
            roi_vert = vert_lines[y:y + bh, x:x + bw]

            # Đếm số lượng đường kẻ ngang và dọc bên trong
            intersections = cv2.bitwise_and(roi_horiz, roi_vert)
            num_intersections = int(np.sum(intersections > 0))

            # Tìm số ô con bên trong
            inner_cnts, _ = cv2.findContours(roi_grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            valid_cells = 0
            for ic in inner_cnts:
                ix, iy, iw, ih = cv2.boundingRect(ic)
                if iw > 15 and ih > 10 and (iw * ih) < (bw * bh * 0.90):
                    valid_cells += 1

            # Một bảng hợp lệ trên tài liệu ngân hàng thường có ít nhất 2 đường kẻ ngang và một số ô
            if valid_cells < 2 and num_intersections < 4:
                # Kiểm tra phụ: Nếu có ít nhất 3 đường kẻ ngang song song rõ rệt
                horiz_proj = np.sum(roi_horiz > 0, axis=1)
                line_peaks = int(np.sum(horiz_proj > bw * 0.35))
                if line_peaks < 2:
                    continue

            # Tính điểm tin cậy
            conf = min(0.85 + (valid_cells * 0.01) + (num_intersections * 0.002), 0.99)

            x0 = float(max(0, x))
            y0 = float(max(0, y))
            x1 = float(min(w, x + bw))
            y1 = float(min(h, y + bh))

            # Crop ảnh bảng với padding
            pad_x0 = max(0, int(x0 - self.padding))
            pad_y0 = max(0, int(y0 - self.padding))
            pad_x1 = min(w, int(x1 + self.padding))
            pad_y1 = min(h, int(y1 + self.padding))
            cropped = image_np[pad_y0:pad_y1, pad_x0:pad_x1].copy()

            detected_tables.append(DetectedTableBox(
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                confidence=round(conf, 3),
                detection_method="opencv_morphology",
                cell_count_est=valid_cells,
                area_ratio=round(area_ratio, 4),
                cropped_image=cropped
            ))

        # Sắp xếp các bảng từ trên xuống dưới
        detected_tables.sort(key=lambda tbl: tbl.y0)
        return detected_tables

    def mask_tables_on_image(
        self,
        image_np: np.ndarray,
        tables: list[DetectedTableBox],
        fill_color: tuple[int, int, int] = (255, 255, 255),
        enabled: bool = False
    ) -> np.ndarray:
        """
        Xóa trắng các vùng bảng trên ảnh trang scan để text OCR không bị nhiễu nét ô.
        Mặc định enabled=False để tránh xóa nhầm các khung văn bản/khung chữ ký hoặc tiêu đề có kẻ viền.
        Nếu enabled=True, áp dụng điều kiện chặt: chỉ mask vùng bảng nếu phát hiện lưới giao cắt
        thực sự có >= 3 hàng và >= 2 cột (tương đương cell_count_est >= 6).
        """
        if not enabled or not tables or image_np is None:
            return image_np

        masked_img = image_np.copy()
        for tbl in tables:
            # Điều kiện chặt: Chỉ che nếu có cấu trúc lưới ô rõ ràng (>= 3 hàng x 2 cột)
            if tbl.cell_count_est < 6:
                continue
            x0 = int(max(tbl.x0 - 2, 0))
            y0 = int(max(tbl.y0 - 2, 0))
            x1 = int(min(tbl.x1 + 2, masked_img.shape[1]))
            y1 = int(min(tbl.y1 + 2, masked_img.shape[0]))
            cv2.rectangle(masked_img, (x0, y0), (x1, y1), fill_color, thickness=-1)

        return masked_img

    def extract_tables_from_pdf_page(
        self,
        pdf_path: str | Path,
        page_idx: int = 0,
        dpi: int = 300
    ) -> tuple[list[DetectedTableBox], np.ndarray | None]:
        """
        Render trang PDF scan 300 DPI và tự động dò bảng biểu.
        """
        doc = fitz.open(str(pdf_path))
        if page_idx >= len(doc):
            return [], None

        page = doc[page_idx]
        pix = page.get_pixmap(dpi=dpi)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
        if pix.n == 4:
            bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

        tables = self.detect_tables(bgr)
        return tables, bgr
