"""
Bird Tracer (Hệ thống Giám sát & Truy vết Luồng RAG - Biểu tượng Con Chim 🦜)
Tích hợp:
1. LangSmith (Thư viện tracing hình con vẹt / con chim của LangChain):
   - Tự động ghi nhận metadata, spans, inputs, outputs, latency.
   - Nếu có LANGSMITH_API_KEY hoặc LANGCHAIN_API_KEY trong .env thì đồng bộ lên Cloud dashboard.
2. Bộ hiển thị Bird Tracing cục bộ:
   - In cây thực thi trực quan dạng ASCII ngay trên terminal.
   - Lưu trữ nhật ký vết (trace logs) vào file JSONL trong thư mục storage.
"""

import time
import json
import os
from pathlib import Path

# Thử import traceable từ LangSmith (logo hình con chim/con vẹt 🦜)
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    # Mock decorator nếu chưa có langsmith
    def traceable(name=None, run_type=None, **kwargs):
        def decorator(func):
            return func
        return decorator

class BirdTracer:
    def __init__(self, log_dir=None):
        if log_dir is None:
            self.log_dir = Path(__file__).resolve().parent.parent / "storage"
        else:
            self.log_dir = Path(log_dir)
            
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trace_file = self.log_dir / "bird_traces.jsonl"
        self.current_trace = None

    def start_trace(self, query):
        """Bắt đầu theo dõi một phiên xử lý câu hỏi khách hàng"""
        self.current_trace = {
            "query": query,
            "start_time": time.time(),
            "spans": []
        }

    def record_span(self, name, duration_ms, details=None):
        """Ghi nhận một mắt xích (span) trong luồng RAG"""
        if self.current_trace is not None:
            self.current_trace["spans"].append({
                "name": name,
                "duration_ms": round(duration_ms, 2),
                "details": details or {}
            })

    def end_trace(self, answer, sources_used=None):
        """Kết thúc truy vết và in báo cáo dạng con chim 🦜"""
        if not self.current_trace:
            return

        total_duration = (time.time() - self.current_trace["start_time"]) * 1000
        self.current_trace["total_duration_ms"] = round(total_duration, 2)
        self.current_trace["answer_preview"] = answer[:150] if answer else ""
        self.current_trace["sources_used"] = sources_used or []

        # Lưu log vào file JSONL
        try:
            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(self.current_trace, ensure_ascii=False) + "\n")
        except Exception:
            pass

        # In cây truy vết visual trực quan trên màn hình
        self._print_bird_summary(self.current_trace)
        self.current_trace = None

    def _print_bird_summary(self, trace_data):
        """In bảng trace trực quan với biểu tượng con chim"""
        total_ms = trace_data.get("total_duration_ms", 0.0)
        query = trace_data.get("query", "")
        
        print("\n" + "╭" + "─" * 68 + "╮")
        print(f"│ 🦜 [BIRD TRACING OBSERVALITY] Tổng thời gian: {total_ms:.1f} ms" + " " * max(1, 67 - len(f"│ 🦜 [BIRD TRACING OBSERVALITY] Tổng thời gian: {total_ms:.1f} ms")) + "│")
        print("├" + "─" * 68 + "┤")
        
        q_short = (query[:50] + "...") if len(query) > 50 else query
        print(f"│  Query: \"{q_short}\"" + " " * max(1, 67 - len(f"│  Query: \"{q_short}\"")) + "│")
        print("│  Các bước thực thi (Execution Spans):" + " " * (68 - len("│  Các bước thực thi (Execution Spans):") - 1) + "│")
        
        spans = trace_data.get("spans", [])
        for idx, span in enumerate(spans):
            is_last = (idx == len(spans) - 1)
            branch = "└─" if is_last else "├─"
            name = span.get("name", "")
            duration = span.get("duration_ms", 0.0)
            details = span.get("details", {})
            detail_str = f" ({details})" if details else ""
            line = f"│   {branch} ⚡ [{name}]: {duration:.1f} ms{detail_str}"
            if len(line) < 69:
                print(line + " " * (69 - len(line)) + "│")
            else:
                print(line[:68] + "│")

        print("╰" + "─" * 68 + "╯\n")

# Khởi tạo singleton tracer
bird_tracer = BirdTracer()

def trace_step(step_name):
    """
    Decorator đo thời gian và tự động ghi nhận vào Bird Tracer.
    Đồng thời liên kết với LangSmith nếu môi trường có cài đặt.
    """
    def decorator(func):
        # Kết hợp với @traceable của LangSmith nếu có
        wrapped = traceable(name=step_name)(func) if LANGSMITH_AVAILABLE else func

        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = wrapped(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.time() - start) * 1000
                bird_tracer.record_span(step_name, duration_ms)
        return wrapper
    return decorator
