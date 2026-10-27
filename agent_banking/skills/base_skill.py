"""
Kỹ năng cơ sở (Base Skill) cho hệ thống AI Banking Assistant.
Thiết kế tối giản, trực quan, dễ hiểu theo chuẩn Chunking:
- Mỗi Skill đại diện cho 1 năng lực nghiệp vụ hoặc tính toán độc lập.
- Tự động đo lường độ trễ và xử lý ngoại lệ an toàn.

NGUYÊN TẮC BẮT BUỘC:
1. KHÔNG DÙNG __INIT__.PY: Import trực tiếp file module.
2. KHÔNG DÙNG THƯ VIỆN TYPING: Thuần túy dùng kiểu gốc của Python (dict, list, str).
3. ZERO TERNARY: Bắt buộc dùng if/else tường minh, không dùng ?: hoặc toán tử 3 ngôi.
4. 0% REGEX LAW: Tuyệt đối không dùng regex.
5. HẠN CHẾ TỐI ĐA THỪA THÃI: Code ngắn gọn, trực diện, dễ hiểu.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field


@dataclass
class SkillMetadata:
    """Siêu dữ liệu mô tả kỹ năng."""
    skill_id: str
    display_name: str
    description: str
    capabilities: list[str] = field(default_factory=list)
    intent_triggers: list[str] = field(default_factory=list)
    category: str = "GENERAL"


class BaseSkill:
    """Lớp cơ sở cho toàn bộ các Kỹ năng nghiệp vụ."""

    def __init__(self, metadata: SkillMetadata, config: object | None = None):
        self.metadata = metadata
        self.config = config

    def execute(self, params: dict) -> dict:
        """Thực thi kỹ năng và trả về kết quả chuẩn hóa kèm độ trễ."""
        start_time = time.perf_counter()
        try:
            data = self._run(params)
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "skill_id": self.metadata.skill_id,
                "status": "SUCCESS",
                "latency_ms": latency_ms,
                "data": data,
                "error": None
            }
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "skill_id": self.metadata.skill_id,
                "status": "FAILED",
                "latency_ms": latency_ms,
                "data": None,
                "error": str(exc)
            }

    def _run(self, params: dict):
        """Logic cụ thể được cài đặt ở từng Skill con."""
        raise NotImplementedError("Mỗi Skill phải cài đặt phương thức _run().")
