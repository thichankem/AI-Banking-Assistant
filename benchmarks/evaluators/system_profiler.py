"""
Module System Profiler
======================
Đo lường chi tiết tài nguyên và tốc độ thực thi của Pipeline Chunking:
1. Latency (ms/trang): Độ trễ trung bình để xử lý 1 trang tài liệu.
2. Throughput (trang/giây): Số lượng trang tài liệu xử lý được trong 1 giây.
3. Peak System RAM (MB): Bộ nhớ RAM hệ thống đỉnh điểm (qua psutil).
4. Peak GPU VRAM (MB): Dung lượng VRAM card đồ họa đỉnh điểm (qua nvidia-smi / torch.cuda / pynvml cho RTX 4060).

Cung cấp cả Context Manager và Decorator tiện dụng.
"""

from __future__ import annotations

import functools
import os
import subprocess
import threading
import time
from typing import Any, Callable

import psutil

try:
    import torch
    HAS_TORCH_CUDA = torch.cuda.is_available()
except Exception:
    HAS_TORCH_CUDA = False


def _get_gpu_vram_mb() -> float:
    """Truy vấn dung lượng VRAM đang dùng (MB) bằng nvidia-smi hoặc torch.cuda."""
    if HAS_TORCH_CUDA:
        try:
            return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        except Exception:
            pass

    # Fallback trực tiếp qua nvidia-smi CLI (hoạt động trên Windows RTX 4060)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            encoding="utf-8",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
        return float(out.strip().split("\n")[0].strip())
    except Exception:
        return 0.0


def _get_system_ram_mb() -> float:
    """Truy vấn RSS RAM của tiến trình hiện tại và các tiến trình con (MB)."""
    try:
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                mem += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return round(mem / (1024 * 1024), 2)
    except Exception:
        return 0.0


class SystemProfiler:
    """Context Manager và bộ đo lường hiệu năng chuyên dụng cho Pipeline Benchmarking."""

    def __init__(self, page_count: int = 1, sample_interval: float = 0.05):
        self.page_count = max(page_count, 1)
        self.sample_interval = sample_interval

        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed_sec: float = 0.0

        self.start_ram: float = 0.0
        self.peak_ram: float = 0.0
        self.start_vram: float = 0.0
        self.peak_vram: float = 0.0

        self._stop_monitor = threading.Event()
        self._monitor_thread: threading.Thread | None = None

    def _monitor_loop(self) -> None:
        """Vòng lặp ngầm lấy mẫu RAM và VRAM đỉnh điểm trong suốt quá trình chạy."""
        while not self._stop_monitor.is_set():
            curr_ram = _get_system_ram_mb()
            curr_vram = _get_gpu_vram_mb()
            if curr_ram > self.peak_ram:
                self.peak_ram = curr_ram
            if curr_vram > self.peak_vram:
                self.peak_vram = curr_vram
            time.sleep(self.sample_interval)

    def start(self) -> None:
        """Bắt đầu đo lường."""
        self.start_ram = _get_system_ram_mb()
        self.peak_ram = self.start_ram
        self.start_vram = _get_gpu_vram_mb()
        self.peak_vram = self.start_vram

        self._stop_monitor.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        self.start_time = time.perf_counter()

    def stop(self) -> dict[str, float]:
        """Dừng đo lường và trả về kết quả tổng hợp."""
        self.end_time = time.perf_counter()
        self.elapsed_sec = self.end_time - self.start_time

        self._stop_monitor.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=0.5)

        total_ms = self.elapsed_sec * 1000.0
        latency_per_page_ms = total_ms / self.page_count
        throughput_pages_per_sec = self.page_count / max(self.elapsed_sec, 1e-6)

        ram_delta = round(max(self.peak_ram - self.start_ram, 0.0), 2)
        vram_delta = round(max(self.peak_vram - self.start_vram, 0.0), 2)

        return {
            "elapsed_seconds": round(self.elapsed_sec, 3),
            "total_latency_ms": round(total_ms, 2),
            "latency_ms_per_page": round(latency_per_page_ms, 2),
            "throughput_pages_per_sec": round(throughput_pages_per_sec, 2),
            "start_ram_mb": self.start_ram,
            "peak_ram_mb": self.peak_ram,
            "ram_delta_mb": ram_delta,
            "start_vram_mb": self.start_vram,
            "peak_vram_mb": self.peak_vram,
            "vram_delta_mb": vram_delta,
            "page_count": self.page_count
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def profile_task(page_count: int = 1):
    """Decorator bọc hàm để đo lường tài nguyên."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = SystemProfiler(page_count=page_count)
            with profiler:
                result = func(*args, **kwargs)
            metrics = profiler.stop()
            return result, metrics
        return wrapper
    return decorator
