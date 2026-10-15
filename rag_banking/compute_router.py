"""
Module Bộ Điều Phối Tính Toán Thông Minh (Smart Compute Router)
Tự động ưu tiên phân bổ tài nguyên:
1. Ưu tiên 1: Google Colab GPU (Tesla T4 / A100 qua domain cố định)
2. Ưu tiên 2: Local GPU (NVIDIA GeForce RTX 4060 qua CUDA)
3. Dự phòng 3: Local CPU
"""

import os
import sys
import requests
import torch
from typing import Dict, Any, List, Optional

# Đảm bảo in tiếng Việt không bị lỗi encoding trên Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Tên miền cố định của bạn trên Google Colab qua Ngrok
COLAB_ENDPOINT = os.getenv("COLAB_ENDPOINT", "https://emperor-blizzard-resume.ngrok-free.dev")

class ComputeTarget:
    COLAB = "colab_gpu"
    LOCAL_GPU = "local_rtx_4060"
    LOCAL_CPU = "local_cpu"

class SmartComputeRouter:
    _instance = None
    _active_target: Optional[str] = None
    _colab_info: Optional[Dict[str, Any]] = None

    def __init__(self, endpoint: str = COLAB_ENDPOINT):
        self.endpoint = endpoint.rstrip("/")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = SmartComputeRouter()
        return cls._instance

    def check_colab_status(self, timeout_sec: float = 1.2) -> Optional[Dict[str, Any]]:
        """
        Kiểm tra trạng thái máy ảo Google Colab.
        """
        try:
            url = f"{self.endpoint}/health"
            # Thêm header ngrok-skip-browser-warning để bypass trang interstitial của ngrok free
            headers = {"ngrok-skip-browser-warning": "true"}
            response = requests.get(url, headers=headers, timeout=timeout_sec)
            if response.status_code == 200:
                data = response.json()
                return data
        except Exception:
            pass
        return None

    def resolve_target(self, force_refresh: bool = False) -> str:
        """
        Xác định thiết bị tính toán tối ưu theo thứ tự ưu tiên:
        Colab GPU -> RTX 4060 Local -> Local CPU
        """
        if self._active_target and not force_refresh:
            return self._active_target

        # 1. Thử kết nối Colab GPU
        colab_data = self.check_colab_status()
        if colab_data and colab_data.get("status") == "ok":
            gpu_name = colab_data.get("gpu", "Cloud GPU")
            print(f"🚀 [Compute Router] Ưu tiên 1: Đã kết nối Google Colab GPU ({gpu_name})!")
            self._active_target = ComputeTarget.COLAB
            self._colab_info = colab_data
            return self._active_target

        # 2. Dự phòng: Kiểm tra card RTX 4060 trên máy
        if torch.cuda.is_available():
            local_gpu = torch.cuda.get_device_name(0)
            print(f"⚡ [Compute Router] Colab offline -> Chuyển sang Local GPU ({local_gpu})")
            self._active_target = ComputeTarget.LOCAL_GPU
            return self._active_target

        # 3. Dự phòng cuối cùng: CPU
        print("💻 [Compute Router] Colab & CUDA offline -> Chuyển sang CPU Local")
        self._active_target = ComputeTarget.LOCAL_CPU
        return self._active_target

    def get_device(self) -> torch.device:
        """Trả về torch.device phù hợp nếu thực thi ở local"""
        target = self.resolve_target()
        if target == ComputeTarget.LOCAL_GPU:
            return torch.device("cuda")
        return torch.device("cpu")

# Khởi tạo singleton tiện dụng
router = SmartComputeRouter.get_instance()

def get_optimal_device() -> str:
    """Hàm helper gọi nhanh từ các module khác"""
    return router.resolve_target()

if __name__ == "__main__":
    print("--- Kiểm tra trạng thái Compute Router ---")
    active = router.resolve_target(force_refresh=True)
    print(f"Thiết bị được kích hoạt: {active}")
