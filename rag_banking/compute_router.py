"""
Module Bộ Điều Phối Tính Toán Thông Minh (Smart Compute Router)
======================================================================
Tự động ưu tiên phân bổ tài nguyên tính toán:
1. Ưu tiên 1: Google Colab GPU (Tesla T4 / A100 qua domain cố định)
2. Ưu tiên 2: Local GPU (NVIDIA GeForce RTX 4060 qua CUDA)
3. Dự phòng 3: Local CPU

NGUYÊN TẮC THIẾT KẾ:
1. 0% REGEX LAW: Tuyệt đối không dùng regex.
2. ZERO TERNARY: Ưu tiên if/else tường minh, dễ đọc.
3. NO __INIT__.PY: Import trực tiếp file module.
"""

import os
import sys
import json
import urllib.request
from pathlib import Path

# Đảm bảo in tiếng Việt không bị lỗi encoding trên Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Tự động nạp biến môi trường từ tệp .env (được .gitignore bảo vệ, không đẩy lên repo)
ROOT_DIR = Path(__file__).resolve().parent.parent
env_file = ROOT_DIR / ".env"
if env_file.exists():
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    except Exception:
        pass

# Nạp địa chỉ Colab từ biến môi trường (mặc định rỗng nếu chưa cấu hình)
COLAB_ENDPOINT = os.getenv("COLAB_ENDPOINT", "")

class ComputeTarget:
    COLAB = "colab_gpu"
    LOCAL_GPU = "local_rtx_4060"
    LOCAL_CPU = "local_cpu"

class SmartComputeRouter:
    _instance = None
    _active_target = None
    _colab_info = None

    def __init__(self, endpoint=None):
        if endpoint is not None:
            self.endpoint = endpoint.rstrip("/")
        elif COLAB_ENDPOINT:
            self.endpoint = COLAB_ENDPOINT.rstrip("/")
        else:
            self.endpoint = ""

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = SmartComputeRouter()
        return cls._instance

    def check_colab_status(self, timeout_sec=1.2):
        """
        Kiểm tra trạng thái máy ảo Google Colab bằng thư viện chuẩn (0% external HTTP).
        """
        if not self.endpoint:
            return None

        try:
            url = f"{self.endpoint}/health"
            req = urllib.request.Request(
                url,
                headers={"ngrok-skip-browser-warning": "true"}
            )
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
        except Exception:
            pass
        return None

    def resolve_target(self, force_refresh=False):
        """
        Xác định thiết bị tính toán tối ưu theo thứ tự ưu tiên:
        Colab GPU -> RTX 4060 Local -> Local CPU
        """
        if self._active_target and not force_refresh:
            return self._active_target

        # 1. Thử kết nối Colab GPU
        colab_data = self.check_colab_status()
        if not colab_data or colab_data.get("status") != "ok":
            # Tự động kích hoạt ngầm Colab qua Autonomous Manager nếu đang offline
            try:
                from gg_colab.auto_colab_manager import ensure_colab_online
                if ensure_colab_online(wait_timeout=15):
                    colab_data = self.check_colab_status()
            except Exception:
                pass

        if colab_data and colab_data.get("status") == "ok":
            gpu_name = colab_data.get("gpu", "Cloud GPU")
            print(f"🚀 [Compute Router] Ưu tiên 1: Đã kết nối Google Colab GPU ({gpu_name})!")
            self._active_target = ComputeTarget.COLAB
            self._colab_info = colab_data
            return self._active_target

        # 2. Dự phòng: Kiểm tra card RTX 4060 trên máy
        try:
            import torch
            if torch.cuda.is_available():
                local_gpu = torch.cuda.get_device_name(0)
                print(f"⚡ [Compute Router] Colab offline -> Chuyển sang Local GPU ({local_gpu})")
                self._active_target = ComputeTarget.LOCAL_GPU
                return self._active_target
        except Exception:
            pass

        # 3. Dự phòng cuối cùng: CPU
        print("💻 [Compute Router] Colab & CUDA offline -> Chuyển sang CPU Local")
        self._active_target = ComputeTarget.LOCAL_CPU
        return self._active_target

    def get_device(self):
        """Trả về device phù hợp nếu thực thi ở local"""
        target = self.resolve_target()
        try:
            import torch
            if target == ComputeTarget.LOCAL_GPU:
                return torch.device("cuda")
            return torch.device("cpu")
        except Exception:
            return "cpu"

# Khởi tạo singleton tiện dụng
router = SmartComputeRouter.get_instance()

def get_optimal_device():
    """Hàm helper gọi nhanh từ các module khác"""
    return router.resolve_target()

if __name__ == "__main__":
    print("--- Kiểm tra trạng thái Compute Router ---")
    active = router.resolve_target(force_refresh=True)
    print(f"Thiết bị được kích hoạt: {active}")
