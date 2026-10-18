"""
Module Scan Detector (Bộ Nhận Diện Trang PDF Scan / Digital)
=============================================================
Hiện thực hóa khối điều kiện rẽ nhánh cốt lõi trong Flowchart:
"Trang Scan? (img_area > 40% hoặc lỗi font)"

Nguyên lý hoạt động 3 điều kiện:
1. Tỷ lệ diện tích ảnh (Image Area Ratio):
   - Quét toàn bộ hình ảnh XObject trên trang qua PyMuPDF (`page.get_images()`).
   - Tính tổng diện tích Bounding Box thực tế: `total_img_area / page_area`.
   - Điều kiện A: `img_area_ratio >= threshold` (mặc định 40%) và chữ số nghèo nàn (< 30 từ).
2. Kiểm tra chất lượng font tiếng Việt (Font Corruption / Garbage OCR Text):
   - Điều kiện B: `w_count > 10` nhưng `is_valid_vietnamese_text() == False`
     (phát hiện lỗi mã hóa font VNI/TCVN3 hoặc text scan bị OCR hỏng trước đó).
3. Trang thuần ảnh (Pure Scanned Image):
   - Điều kiện C: Số ký tự số < 20 nhưng trang có chứa ảnh XObject (ảnh chụp công văn, scan giấy phẳng).
"""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz

try:
    from ..unified_chunk.canonical_resolver import is_valid_vietnamese_text, is_valid_readable_text
except (ImportError, ValueError):
    from rag_banking.chunking.unified_chunk.canonical_resolver import is_valid_vietnamese_text, is_valid_readable_text


def detect_is_scanned_page(
    page: object,
    image_area_ratio_threshold: float = 0.40,
    min_word_threshold: int = 30
) -> tuple[bool, dict]:
    """
    Hàm phân loại độc lập: Kiểm tra một trang PDF là Trang Scan (Scanned) hay Trang Số (Digital).

    Args:
        page: Đối tượng fitz.Page (PyMuPDF) của trang cần kiểm tra.
        image_area_ratio_threshold: Ngưỡng tỷ lệ diện tích ảnh kích hoạt scan (mặc định 0.40 hay 40%).
        min_word_threshold: Ngưỡng số từ số tối thiểu (mặc định 30 từ).

    Returns:
        (is_scanned, metrics):
            is_scanned (bool): True nếu là trang scan, False nếu là trang số.
            metrics (dict): Chi tiết các chỉ số đo lường phục vụ audit/debug.
    """
    page_rect = page.rect
    page_area = max(page_rect.width * page_rect.height, 1.0)

    # 1. Đo lường diện tích ảnh XObject trên trang
    img_list = page.get_images()
    total_img_area = 0.0
    for img in img_list:
        try:
            for bbox in page.get_image_rects(img[0]):
                total_img_area += (bbox.width * bbox.height)
        except Exception:
            pass
    img_area_ratio = min(total_img_area / page_area, 1.0)

    # 2. Đo lường văn bản số (digital text)
    digital_text = page.get_text("text").strip()
    char_count = len(digital_text)
    w_count = len(digital_text.split())
    is_valid_readable = is_valid_readable_text(digital_text)
    is_valid_vn = is_valid_vietnamese_text(digital_text)

    # 3. Đánh giá 3 điều kiện quyết định theo Flowchart
    # - Điều kiện A: Ảnh chiếm diện tích lớn (>= 40%) và chữ số nghèo nàn (< 30 từ)
    cond_a = (img_area_ratio >= image_area_ratio_threshold and w_count < min_word_threshold)
    # - Điều kiện B: Có chữ số nhưng lỗi font / Mojibake / text OCR rác (chấp nhận cả tiếng Việt, tiếng Anh, code)
    cond_b = (w_count > 10 and not is_valid_readable)
    # - Điều kiện C: Rất ít ký tự (< 20) nhưng trang có chứa ảnh XObject
    cond_c = (char_count < 20 and len(img_list) > 0)

    is_scanned = cond_a or cond_b or cond_c

    # Xác định lý do phân loại
    reason = "digital_text"
    if cond_a:
        reason = f"image_area_ratio ({img_area_ratio:.2%}) >= {image_area_ratio_threshold:.0%} và word_count ({w_count}) < {min_word_threshold}"
    elif cond_b:
        reason = "corrupted_font_or_mojibake (tỷ lệ ký tự lỗi hoặc bảng mã hỏng)"
    elif cond_c:
        reason = f"low_char_count ({char_count} < 20) có chứa {len(img_list)} ảnh XObject"

    metrics = {
        "is_scanned": is_scanned,
        "classification": "scanned" if is_scanned else "digital",
        "reason": reason,
        "img_count": len(img_list),
        "img_area_ratio": round(img_area_ratio, 4),
        "char_count": char_count,
        "word_count": w_count,
        "is_valid_readable": is_valid_readable,
        "is_valid_vietnamese": is_valid_vn
    }

    return is_scanned, metrics


def is_scanned_page(page: object, image_area_ratio_threshold: float = 0.40) -> bool:
    """Hàm tiện ích nhanh trả về True nếu là bản scan, False nếu là văn bản số."""
    is_scanned, _ = detect_is_scanned_page(page, image_area_ratio_threshold=image_area_ratio_threshold)
    return is_scanned


def detect_pdf_file_scan_status(
    pdf_path: str | Path,
    max_pages: int | None = None,
    image_area_ratio_threshold: float = 0.40
) -> dict:
    """
    Kiểm tra toàn diện trạng thái scan của một tệp PDF (hỗ trợ file đa trang):
    Trả về tỷ lệ trang scan, danh sách từng trang và kết luận tài liệu scan chủ đạo hay không.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp: {pdf_path}")


    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    pages_to_check = min(total_pages, max_pages) if max_pages else total_pages

    pages_report = []
    scanned_count = 0

    for idx in range(pages_to_check):
        page = doc[idx]
        is_scanned, metrics = detect_is_scanned_page(page, image_area_ratio_threshold=image_area_ratio_threshold)
        metrics["page_number"] = idx + 1
        pages_report.append(metrics)
        if is_scanned:
            scanned_count += 1

    doc.close()

    scan_ratio = scanned_count / max(pages_to_check, 1)
    is_mostly_scanned = scan_ratio >= 0.50

    return {
        "file_name": pdf_path.name,
        "total_pages": total_pages,
        "checked_pages": pages_to_check,
        "scanned_pages_count": scanned_count,
        "scan_ratio": round(scan_ratio, 4),
        "is_mostly_scanned": is_mostly_scanned,
        "pages_report": pages_report
    }


class ScanDetector:
    """
    Lớp đóng gói hướng đối tượng (OOP Wrapper) cho bộ nhận diện PDF scan/digital.
    """

    def __init__(
        self,
        image_area_ratio_threshold: float = 0.40,
        min_word_threshold: int = 30
    ):
        self.image_area_ratio_threshold = image_area_ratio_threshold
        self.min_word_threshold = min_word_threshold

    def is_scanned_page(self, page: object) -> bool:
        """Kiểm tra nhanh trang scan."""
        return is_scanned_page(page, image_area_ratio_threshold=self.image_area_ratio_threshold)

    def detect_page(self, page: object) -> tuple[bool, dict]:
        """Phân tích chi tiết trang scan."""
        return detect_is_scanned_page(
            page,
            image_area_ratio_threshold=self.image_area_ratio_threshold,
            min_word_threshold=self.min_word_threshold
        )

    def detect_file(
        self,
        pdf_path: str | Path,
        max_pages: int | None = None
    ) -> dict:
        """Phân tích toàn bộ tệp PDF đa trang."""
        return detect_pdf_file_scan_status(
            pdf_path,
            max_pages=max_pages,
            image_area_ratio_threshold=self.image_area_ratio_threshold
        )
