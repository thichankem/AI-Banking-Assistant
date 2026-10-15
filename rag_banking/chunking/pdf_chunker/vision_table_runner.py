"""
Module Vision Table Runner (Tối ưu hóa VLM cho GPU RTX 4060 8GB VRAM)
=====================================================================
Giải quyết bài toán: Chạy model lớn 12B trên Ollama chiếm dụng gần cạn 8GB VRAM,
dễ gây hiện tượng nghẽn cổ chai (bottleneck), tràn bộ nhớ CUDA OOM khi tài liệu có nhiều bảng.

Đặc tính nâng cấp:
1. Hỗ trợ chuyển đổi linh hoạt đa mô hình Vision LLM nhẹ hơn (Multi-Model Switching):
   - Mặc định đề xuất: `qwen2.5-vl:7b` (~4.5GB VRAM), `qwen2.5-vl:3b` (~2.2GB VRAM), `minicpm-v` (~5.5GB VRAM).
   - Danh sách Fallback tự động khi mô hình chính quá tải hoặc lỗi.
2. Kiểm soát đồng thời nghiêm ngặt (Strict Concurrency Limiting via Semaphore):
   - Mặc định max_concurrency = 1 để bảo vệ 8GB VRAM của RTX 4060.
3. Giám sát vi mô VRAM & Latency (Real-time Telemetry):
   - Đo VRAM trước và sau suy luận (Allocated, Reserved, Peak MB) qua PyTorch CUDA hoặc nvidia-smi.
   - Bấm giờ chính xác từng mili-giây (Inference Latency).
4. Cơ chế Timeout an toàn và tự động dọn dẹp bộ nhớ đệm VRAM (torch.cuda.empty_cache()).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("vision_table_runner")


@dataclass
class VisionExecutionStats:
    """Bản ghi thống kê hiệu năng suy luận bảng thị giác."""
    model_name: str
    status: str  # 'success', 'error', 'timeout', 'fallback'
    latency_seconds: float
    vram_allocated_mb: float = 0.0
    vram_peak_mb: float = 0.0
    vram_system_used_mb: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "status": self.status,
            "latency_seconds": round(self.latency_seconds, 2),
            "vram_allocated_mb": round(self.vram_allocated_mb, 1),
            "vram_peak_mb": round(self.vram_peak_mb, 1),
            "vram_system_used_mb": round(self.system_vram_mb if hasattr(self, 'system_vram_mb') else self.vram_system_used_mb, 1),
            "error_message": self.error_message
        }


class VisionTableRunner:
    """
    Bộ điều phối và tối ưu hóa suy luận Vision LLM cho bảng biểu trên GPU RTX 4060.
    """

    SUPPORTED_LIGHT_MODELS = [
        "qwen2.5-vl:7b",
        "qwen2.5-vl:3b",
        "minicpm-v",
        "llama3.2-vision:11b",
        "gemma4:12b"
    ]

    def __init__(
        self,
        default_model: str = "qwen2.5-vl:7b",
        fallback_models: list[str] | None = None,
        ollama_base_url: str = "http://127.0.0.1:11434",
        max_concurrency: int = 1,
        timeout_seconds: int = 120,
        dpi: int = 150,
        temperature: float = 0.1
    ):
        self.model_name = default_model
        self.fallback_models = fallback_models or ["qwen2.5-vl:3b", "minicpm-v"]
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.dpi = dpi
        self.temperature = temperature

        # Semaphore bảo vệ VRAM RTX 4060
        self._semaphore = threading.BoundedSemaphore(max(1, max_concurrency))
        self.history_stats: list[VisionExecutionStats] = []

    def switch_model(self, new_model_name: str) -> None:
        """Chuyển đổi linh hoạt mô hình Vision LLM đang hoạt động."""
        logger.info(f"[VisionTableRunner] Chuyển đổi mô hình từ '{self.model_name}' sang '{new_model_name}'.")
        self.model_name = new_model_name

    def list_available_ollama_models(self) -> list[str]:
        """Truy vấn danh sách các mô hình hiện có sẵn trên daemon Ollama cục bộ."""
        try:
            req = urllib.request.Request(f"{self.ollama_base_url}/api/tags", headers={"User-Agent": "VisionTableRunner/1.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                return models
        except Exception as e:
            logger.warning(f"[VisionTableRunner] Không thể kết nối Ollama: {e}")
            return []

    def get_vram_telemetry(self) -> dict[str, float]:
        """
        Đo lường VRAM GPU qua PyTorch CUDA hoặc nvidia-smi.
        """
        allocated_mb = 0.0
        peak_mb = 0.0
        system_used_mb = 0.0

        # 1. Thử qua PyTorch CUDA
        try:
            import torch
            if torch.cuda.is_available():
                allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        except Exception:
            pass

        # 2. Thử truy vấn nvidia-smi cho toàn hệ thống
        try:
            cmd = ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"]
            out = subprocess.check_output(cmd, encoding="utf-8", timeout=2).strip()
            parts = out.split("\n")[0].split(",")
            system_used_mb = float(parts[0].strip())
        except Exception:
            pass

        return {
            "vram_allocated_mb": round(allocated_mb, 1),
            "vram_peak_mb": round(peak_mb, 1),
            "vram_system_used_mb": round(system_used_mb, 1)
        }

    def _render_image_to_base64(self, image_input: Any) -> str:
        """Chuẩn hóa ảnh đầu vào (Path, numpy ndarray, PIL Image hoặc fitz.Page) về Base64 PNG."""
        # 1. fitz.Page
        if hasattr(image_input, "get_pixmap"):
            pix = image_input.get_pixmap(dpi=self.dpi)
            return base64.b64encode(pix.tobytes("png")).decode("utf-8")

        # 2. numpy.ndarray (OpenCV BGR/RGB)
        if isinstance(image_input, np.ndarray):
            rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB) if len(image_input.shape) == 3 else image_input
            pil_img = Image.fromarray(rgb)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        # 3. PIL.Image
        if isinstance(image_input, Image.Image):
            buf = io.BytesIO()
            image_input.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        # 4. Path / str
        p = Path(str(image_input))
        if p.is_file():
            with open(p, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        raise ValueError(f"Định dạng ảnh không được hỗ trợ: {type(image_input)}")

    def extract_table(
        self,
        image_input: Any,
        doc_title: str = "",
        page_num: int = 1,
        custom_prompt: str | None = None,
        allow_fallback: bool = True
    ) -> dict[str, Any]:
        """
        Trích xuất cấu trúc bảng từ ảnh trang/crop bằng mô hình VLM được cấu hình.
        Có cơ chế Semaphore giới hạn concurrency, đo VRAM/Latency và fallback linh hoạt.
        """
        img_b64 = self._render_image_to_base64(image_input)

        default_prompt = (
            "Bạn là chuyên gia thị giác máy tính và phân tích cấu trúc tài liệu ngân hàng (Document Layout Vision AI).\n"
            "Nhiệm vụ: Trích xuất TOÀN BỘ các bảng biểu, số liệu ma trận từ ảnh này.\n"
            "Yêu cầu:\n"
            "1. Cấu trúc ô gộp: Điền đầy đủ ngữ cảnh của cột mẹ và dòng mẹ vào từng ô con (unmerge & context fill).\n"
            "2. Chuẩn hóa thành định dạng bảng Markdown (cột và hàng phân cách bằng `|`).\n"
            "3. Giữ nguyên 100% tiếng Việt có dấu, độ chính xác tuyệt đối của các con số, đơn vị tính (VND, USD, %, tỷ đồng).\n"
            "4. Xuất trực tiếp bảng Markdown chuẩn, không thêm lời chào hay giải thích ngoài lề."
        )
        prompt_text = custom_prompt or default_prompt

        candidate_models = [self.model_name]
        if allow_fallback:
            for fb in self.fallback_models:
                if fb not in candidate_models:
                    candidate_models.append(fb)

        last_error = None

        for current_model in candidate_models:
            # Thu thập VRAM trước suy luận
            vram_before = self.get_vram_telemetry()
            t0 = time.time()

            logger.info(f"[VisionTableRunner] Xếp hàng xử lý (Model: {current_model}, Trang: {page_num})...")

            acquired = self._semaphore.acquire(timeout=self.timeout_seconds)
            if not acquired:
                msg = f"Timeout chờ khóa GPU sau {self.timeout_seconds}s"
                stats = VisionExecutionStats(
                    model_name=current_model,
                    status="timeout",
                    latency_seconds=round(time.time() - t0, 2),
                    error_message=msg
                )
                self.history_stats.append(stats)
                return {"status": "timeout", "error": msg, "stats": stats.to_dict()}

            try:
                payload = {
                    "model": current_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt_text,
                            "images": [img_b64]
                        }
                    ],
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": 1800
                    }
                }

                req = urllib.request.Request(
                    f"{self.ollama_base_url}/api/chat",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )

                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    resp_json = json.loads(resp.read().decode("utf-8"))
                    msg_obj = resp_json.get("message", {})
                    raw_text = msg_obj.get("content", "").strip() or msg_obj.get("thinking", "").strip()

                elapsed = time.time() - t0
                vram_after = self.get_vram_telemetry()

                stats = VisionExecutionStats(
                    model_name=current_model,
                    status="success" if current_model == self.model_name else "fallback",
                    latency_seconds=round(elapsed, 2),
                    vram_allocated_mb=vram_after["vram_allocated_mb"],
                    vram_peak_mb=max(vram_after["vram_peak_mb"], vram_before["vram_peak_mb"]),
                    vram_system_used_mb=vram_after["vram_system_used_mb"]
                )
                self.history_stats.append(stats)

                # Dọn dẹp cache VRAM nhẹ nhàng
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

                return {
                    "status": "success",
                    "model_used": current_model,
                    "raw_response": raw_text,
                    "doc_title": doc_title,
                    "page_num": page_num,
                    "stats": stats.to_dict()
                }

            except Exception as e:
                last_error = str(e)
                logger.warning(f"[VisionTableRunner] Thử mô hình '{current_model}' thất bại: {e}")
                stats = VisionExecutionStats(
                    model_name=current_model,
                    status="error",
                    latency_seconds=round(time.time() - t0, 2),
                    error_message=str(e)
                )
                self.history_stats.append(stats)
            finally:
                self._semaphore.release()

        return {
            "status": "error",
            "error": f"Tất cả mô hình {candidate_models} đều thất bại. Chi tiết: {last_error}",
            "stats": self.history_stats[-1].to_dict() if self.history_stats else {}
        }
