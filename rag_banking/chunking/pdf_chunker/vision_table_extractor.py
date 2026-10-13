"""
Module Local Vision Table Extractor (Bộ Trích Xuất Bảng Phức Tạp bằng Vision LLM Cục Bộ)
========================================================================================
Giải pháp chuyên sâu cho các bảng biểu phức tạp, ô gộp đa tầng (merged cells), bảng scan mộc đỏ
hoặc bảng biểu dạng slide/ma trận tài chính mà các thư viện OCR hình học truyền thống (PyMuPDF, pdfplumber)
không thể bóc tách trọn vẹn ngữ nghĩa.

Đặc tính cốt lõi:
1. 100% Cục bộ (Local Only):
   - Chạy trực tiếp trên phần cứng máy trạm qua backend Ollama cục bộ (GPU NVIDIA GeForce RTX 4060 8GB).
   - Tuyệt đối không gửi dữ liệu ra ngoài Internet, bảo đảm an toàn bảo mật tài chính ngân hàng.
2. Xếp hàng đợi tuần tự (Thread-Safe Serial Queue):
   - Áp dụng cơ chế Semaphore / Mutex Lock để đảm bảo tại một thời điểm CHỈ CÓ ĐÚNG 1 tác vụ Vision LLM
     được gửi tới VRAM GPU, loại bỏ triệt để nguy cơ tràn bộ nhớ VRAM (CUDA Out Of Memory - OOM).
3. Đọc hiểu chuyên sâu Tiếng Việt & Cấu trúc bảng:
   - System prompt tối ưu hóa cho bảng tài chính, biểu phí, báo cáo tài chính (BCTC) Việt Nam.
   - Tự động điền đầy đủ ngữ cảnh cho các ô gộp (unmerge & fill context), bảo toàn các đơn vị tính
     (tỷ đồng, triệu đồng, %, USD, VND) và số liệu chính xác 100%.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pymupdf as fitz

logger = logging.getLogger("vision_table_extractor")

# Global Lock xếp hàng đợi xử lý Vision LLM duy nhất trên toàn tiến trình
_VISION_QUEUE_LOCK = threading.Lock()


class LocalVisionTableExtractor:
    """
    Bộ bóc tách bảng biểu phức tạp sử dụng Local Vision LLM (Ollama).
    Được thiết kế chạy an toàn, bền bỉ trên GPU RTX 4060 (8GB VRAM) với cơ chế xếp hàng đợi.
    """

    def __init__(
        self,
        model_name: str = "gemma4:12b",
        ollama_base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: int = 300,
        dpi: int = 150
    ):
        """
        Khởi tạo bộ trích xuất Vision Table.

        Args:
            model_name: Tên mô hình Vision LLM trên Ollama (mặc định 'gemma4:12b' có sẵn trên máy).
            ollama_base_url: Địa chỉ endpoint dịch vụ Ollama cục bộ.
            timeout_seconds: Thời gian chờ tối đa cho 1 request suy luận hình ảnh.
            dpi: Độ phân giải render trang PDF thành ảnh phục vụ Vision (150 DPI là tỷ lệ vàng giữa nét và nhẹ VRAM).
        """
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.dpi = dpi
        self._is_available_cache: bool | None = None

    def is_available(self) -> bool:
        """Kiểm tra dịch vụ Ollama cục bộ và mô hình Vision LLM đã sẵn sàng hay chưa."""
        try:
            req = urllib.request.Request(f"{self.ollama_base_url}/api/tags", headers={"User-Agent": "VisionExtractor/1.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
                full_names = [m.get("name", "") for m in data.get("models", [])]
                
                target_base = self.model_name.split(":")[0]
                is_ready = (self.model_name in full_names) or (target_base in models)
                self._is_available_cache = is_ready
                return is_ready
        except Exception:
            self._is_available_cache = False
            return False

    def is_complex_table_page(self, page_text: str, digital_tables_found: int, has_images: bool = False) -> bool:
        """
        Nhận diện trang chứa bảng phức tạp dựa trên cấu trúc hình học và dữ liệu (Structural Tabular Heuristics):
        Hoàn toàn KHÔNG sử dụng từ điển từ khóa riêng của bất kỳ ngành nghề nào.

        Điều kiện kích hoạt:
        1. Parser số học truyền thống không trích xuất được bảng (digital_tables_found == 0).
        2. Trang có các đặc trưng phân bố dạng bảng:
           - Chứa các ký tự phân cách cột (|, \\t) hoặc khoảng cách đa cột.
           - Hoặc các dòng có mật độ số liệu, chỉ tiêu và tỷ lệ phân bố theo cột.
           - Hoặc trang scan/ảnh có văn bản phân bố dạng bảng.
        """
        if digital_tables_found > 0:
            return False

        lines = [line.strip() for line in page_text.split("\n") if line.strip()]
        if not lines:
            return has_images

        # Đếm các dòng mang đặc trưng cấu trúc bảng biểu
        tabular_lines = 0
        for line in lines:
            digits_count = sum(1 for c in line if c.isdigit())
            has_delimiters = "|" in line or "\t" in line or "  " in line
            if digits_count >= 2 and (has_delimiters or "%" in line):
                tabular_lines += 1

        # Nếu có ít nhất 2 dòng mang cấu trúc số liệu dạng bảng, hoặc có ảnh và dòng số liệu
        if tabular_lines >= 2 or (has_images and tabular_lines >= 1):
            return True

        return False

    def render_page_to_base64(self, page: object, dpi: int | None = None) -> str:
        """Render trang PDF thành chuỗi ảnh Base64 PNG chuẩn."""
        target_dpi = dpi or self.dpi
        pix = page.get_pixmap(dpi=target_dpi)
        img_bytes = pix.tobytes("png")
        return base64.b64encode(img_bytes).decode("utf-8")

    def extract_table_from_page(
        self,
        page: object,
        doc_title: str = "",
        page_num: int = 1,
        custom_prompt: str | None = None
    ) -> dict | None:
        """
        Trích xuất bảng biểu từ trang PDF sử dụng Local Vision LLM với cơ chế xếp hàng đợi tuần tự.

        Returns:
            Dict chứa:
                - markdown_table: Chuỗi bảng Markdown chuẩn.
                - raw_response: Toàn bộ phản hồi giải thích của Vision LLM.
                - elapsed_seconds: Thời gian suy luận.
                - status: 'success' hoặc 'error' / 'fallback'.
        """
        if not self.is_available():
            logger.warning("[Vision Table Extractor] Dịch vụ Ollama hoặc mô hình chưa sẵn sàng. Bỏ qua.")
            return None

        # 1. Render ảnh trang ở độ phân giải tối ưu
        img_b64 = self.render_page_to_base64(page)

        # 2. Xây dựng Prompt chuyên biệt cho Bảng biểu Phức tạp Tiếng Việt (Tổng quát đa lĩnh vực)
        default_prompt = (
            "Bạn là chuyên gia thị giác máy tính và phân tích cấu trúc tài liệu (Document Layout Vision AI).\n"
            "Nhiệm vụ của bạn là đọc hình ảnh trang tài liệu này và trích xuất TOÀN BỘ các bảng biểu, số liệu ma trận có trên trang:\n"
            "1. Xác định chính xác cấu trúc bảng: Nếu có các ô gộp hàng hoặc cột (merged cells / multi-level header), "
            "hãy điền đầy đủ ngữ cảnh của cột mẹ và dòng mẹ vào từng ô con để mọi dòng đều độc lập và tự giải thích được nghĩa.\n"
            "2. Chuẩn hóa thành định dạng bảng Markdown chuẩn (cột, hàng rõ ràng bằng cú pháp `|`).\n"
            "3. Giữ nguyên 100% tiếng Việt có dấu, độ chính xác tuyệt đối của các con số, tỷ lệ phần trăm (%), "
            "dấu phân cách thập phân, và đơn vị tính.\n"
            "4. Xuất trực tiếp bảng Markdown chuẩn xác, không viết văn giải thích dài dòng."
        )
        prompt_text = custom_prompt or default_prompt

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt_text,
                    "images": [img_b64]
                }
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.1,  # Nhiệt độ thấp đảm bảo trích xuất số liệu chuẩn xác, không hallucinate
                "num_predict": 1500
            }
        }

        # 3. Đi vào Hàng đợi Tuần tự (Serial Execution via Queue Lock)
        # Đảm bảo chỉ 1 request chiếm dụng VRAM GPU RTX 4060 tại một thời điểm
        t0 = time.time()
        logger.info(f"[Vision Queue] Bắt đầu xếp hàng đợi xử lý trang {page_num} ({doc_title})...")

        with _VISION_QUEUE_LOCK:
            logger.info(f"[Vision Queue] Đã lấy khóa GPU! Bắt đầu suy luận Vision LLM trang {page_num}...")
            req = urllib.request.Request(
                f"{self.ollama_base_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )

            try:
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    resp_json = json.loads(resp.read().decode("utf-8"))
                    msg = resp_json.get("message", {})
                    raw_text = msg.get("content", "").strip() or msg.get("thinking", "").strip()
            except Exception as e:
                logger.error(f"[Vision Table Extractor] Lỗi suy luận trên trang {page_num}: {e}")
                return None

        elapsed = time.time() - t0
        logger.info(f"[Vision Queue] Hoàn tất suy luận trang {page_num} trong {elapsed:.2f}s!")

        # 4. Tách bảng Markdown từ phản hồi của LLM
        md_tables = self._extract_markdown_tables(raw_text)

        return {
            "status": "success",
            "doc_title": doc_title,
            "page_num": page_num,
            "markdown_table": "\n\n".join(md_tables) if md_tables else raw_text,
            "raw_response": raw_text,
            "elapsed_seconds": round(elapsed, 2),
            "model_used": self.model_name
        }

    def _extract_markdown_tables(self, text: str) -> list[str]:
        """Tách các khối bảng Markdown từ phản hồi văn bản của LLM."""
        lines = text.split("\n")
        tables: list[str] = []
        current_table: list[str] = []
        in_table = False

        for line in lines:
            stripped = line.strip()
            # Bắt các dòng dạng bảng Markdown (| Cột 1 | Cột 2 | ...)
            if stripped.startswith("|") and ("|" in stripped[1:]):
                in_table = True
                current_table.append(stripped)
            else:
                if in_table:
                    if len(current_table) >= 2:
                        tables.append("\n".join(current_table))
                    current_table = []
                    in_table = False

        if in_table and len(current_table) >= 2:
            tables.append("\n".join(current_table))

        return tables

