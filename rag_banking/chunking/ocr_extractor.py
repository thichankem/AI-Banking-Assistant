"""
Bộ trích xuất ký tự quang học (OCR Extractor)
Sử dụng RapidOCR trên nền ONNX Runtime kết hợp pypdfium2:
- Tự động nhận diện chữ tiếng Việt từ các trang PDF scan, ảnh báo cáo tài chính, nghị quyết có dấu mộc.
- Hoạt động 100% offline cục bộ, tốc độ xử lý nhanh trên CPU.
"""

import time
from pathlib import Path

try:
    import pypdfium2 as pdfium
    from rapidocr_onnxruntime import RapidOCR
    OCR_SUPPORTED = True
except ImportError:
    OCR_SUPPORTED = False

class OCRExtractor:
    def __init__(self):
        self._engine = None
        if OCR_SUPPORTED:
            try:
                self._engine = RapidOCR()
            except Exception as e:
                print(f"[OCR] Lỗi khởi tạo RapidOCR engine: {e}")

    def is_available(self):
        """Kiểm tra xem thư viện OCR có khả dụng không"""
        return OCR_SUPPORTED and (self._engine is not None)

    def extract_text_from_image(self, image):
        """Trích xuất văn bản từ đối tượng PIL Image hoặc numpy array"""
        if not self.is_available():
            return ""

        try:
            result, _ = self._engine(image)
            if not result:
                return ""
            # result có cấu trúc: [[box, text, score], ...]
            lines = [item[1] for item in result if item and len(item) > 1 and item[1].strip()]
            return "\n".join(lines)
        except Exception as e:
            print(f"[OCR] Lỗi khi nhận diện ảnh: {e}")
            return ""

    def extract_text_from_pdf_page(self, pdf_path, page_idx=0, scale=2):
        """
        Render trang PDF thành hình ảnh và nhận diện OCR tiếng Việt.
        """
        if not self.is_available():
            return ""

        try:
            doc = pdfium.PdfDocument(pdf_path)
            if page_idx >= len(doc):
                return ""
            page = doc[page_idx]
            pil_image = page.render(scale=scale).to_pil()
            return self.extract_text_from_image(pil_image)
        except Exception as e:
            print(f"[OCR] Lỗi khi render/OCR trang {page_idx} của {Path(pdf_path).name}: {e}")
            return ""
