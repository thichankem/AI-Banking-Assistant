"""
Bộ Điều Phối Quản Lý Tác Vụ Ngân Hàng (Banking Task Manager).
Thiết kế tối giản, trực quan, dễ hiểu theo chuẩn Chunking:
- Node Đầu Tiên (Intent Node): Gọi TaskRouter phân loại ý định dựa trên Skills.
- Điều phối Thực thi: Chuyển giao ngữ cảnh cho Task chuyên trách (chuỗi Nodes con).
- Đóng gói Kết quả Thống nhất: Trả về câu trả lời, trích dẫn, các bước thực thi và độ trễ.

NGUYÊN TẮC BẮT BUỘC:
1. KHÔNG DÙNG __INIT__.PY: Import trực tiếp file module.
2. KHÔNG DÙNG THƯ VIỆN TYPING: Thuần túy dùng kiểu gốc của Python.
3. ZERO TERNARY: Bắt buộc dùng if/else tường minh, không dùng ?: hoặc toán tử 3 ngôi.
4. 0% REGEX LAW: Tuyệt đối không dùng regex.
5. HẠN CHẾ TỐI ĐA THỪA THÃI: Code ngắn gọn, trực diện, dễ hiểu.
"""

from __future__ import annotations
import time

from agent_banking.skills.skill_registry import SkillRegistry
from agent_banking.config import Config
from .base_task import BaseBankingTask
from .task_router import TaskRouter
from .policy_inquiry_task.task import PolicyInquiryTask
from .loan_advisory_task.task import LoanAdvisoryTask
from .biometric_compliance_task.task import BiometricComplianceTask
from .card_service_task.task import CardServiceTask
from .incident_support_task.task import IncidentSupportTask
from .general_chat_task.task import GeneralChatTask
from .react_agent_task.task import ReActAgentTask


class BankingTaskManager:
    """Bộ điều phối toàn trình quản lý các Tasks và Skills nghiệp vụ."""

    def __init__(self, pipeline_instance=None, config: Config | None = None):
        if config is not None:
            self.config = config
        else:
            self.config = Config()

        # 1. Danh bạ Kỹ năng (Skill Registry)
        self.skill_registry = SkillRegistry(pipeline_instance=pipeline_instance, config=self.config)

        # 2. Node Đầu Tiên: Intent Router
        self.router = TaskRouter(skill_registry=self.skill_registry, config=self.config)

        # 3. Đăng ký các Tasks nghiệp vụ
        self.tasks: dict[str, BaseBankingTask] = {}
        self._register(PolicyInquiryTask(skill_registry=self.skill_registry, config=self.config))
        self._register(LoanAdvisoryTask(skill_registry=self.skill_registry, config=self.config))
        self._register(BiometricComplianceTask(skill_registry=self.skill_registry, config=self.config))
        self._register(CardServiceTask(skill_registry=self.skill_registry, config=self.config))
        self._register(IncidentSupportTask(skill_registry=self.skill_registry, config=self.config))
        self._register(GeneralChatTask(skill_registry=self.skill_registry, config=self.config))
        self._register(ReActAgentTask(skill_registry=self.skill_registry, config=self.config))

    def _register(self, task: BaseBankingTask) -> None:
        self.tasks[task.task_id] = task

    def process_query(self, user_query: str, context: dict | None = None) -> dict:
        """
        Xử lý toàn trình:
        1. Node 0 (Intent Router): Chọn Task dựa trên Skills.
        2. Task Execution: Chạy chuỗi Nodes con bên trong Task.
        3. Tổng hợp kết quả thống nhất.
        """
        start_time = time.perf_counter()
        clean_q = user_query.strip()
        if not clean_q:
            return {
                "answer": "Kính chào Quý khách! Vui lòng nhập nội dung yêu cầu cần hỗ trợ.",
                "task_id": "NONE",
                "task_name": "Không xác định",
                "execution_steps": [],
                "latency_ms": 0.0
            }

        # Node Đầu Tiên: Intent Router
        decision = self.router.route(clean_q)

        # Kích hoạt Task được chọn
        task = self.tasks.get(decision.task_id, self.tasks["TASK_POLICY_INQUIRY"])
        task_res = task.execute(clean_q, context=context)

        total_latency = round((time.perf_counter() - start_time) * 1000, 2)

        # Tổng hợp các bước thực thi
        combined_steps = [
            f"Node 0 (Intent Router): Điều hướng vào [{decision.task_id}] dựa trên skills: {decision.target_skills} ({decision.reason})"
        ]
        combined_steps.extend(task_res.execution_steps)

        return {
            "answer": task_res.answer,
            "task_id": task_res.task_id,
            "task_name": task_res.task_name,
            "status": task_res.status,
            "routing_decision": {
                "task_id": decision.task_id,
                "task_name": decision.task_name,
                "confidence": decision.confidence,
                "target_skills": decision.target_skills,
                "reason": decision.reason
            },
            "citations": task_res.citations,
            "sources": task_res.sources,
            "structured_data": task_res.structured_data,
            "execution_steps": combined_steps,
            "latency_ms": total_latency
        }
