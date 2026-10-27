"""
Nhiệm vụ AI Agent Suy luận & Hành động ReAct Đa bước (ReAct Agent Task).
Thiết kế theo chuẩn Chunking: 0% Regex, Zero Ternary, No Typing Library, No __init__.py.
- Tự động đọc danh mục Skills từ SkillRegistry.
- Vận hành vòng lặp ReAct: Thought -> Action -> Observation -> Final Answer.
- Giải quyết các câu hỏi phức tạp kết hợp nhiều nghiệp vụ ngân hàng.
"""

from __future__ import annotations
from agent_banking.tasks.base_task import BaseBankingTask, TaskResult
from agent_banking.skills.skill_registry import SkillRegistry
from agent_banking.config import Config


class ReActAgentTask(BaseBankingTask):
    """Task điều phối AI Agent suy luận linh hoạt theo phương pháp ReAct."""

    def __init__(self, skill_registry: SkillRegistry | None = None, config: Config | None = None):
        super().__init__(
            task_id="TASK_REACT_AGENT",
            task_name="AI Agent Suy Luận & Hành Động Đa Bước (ReAct)",
            description="Tự động đọc danh bạ kỹ năng nghiệp vụ (Skills) và lặp vòng Thought -> Action -> Observation để giải quyết yêu cầu đa tầng của khách hàng.",
            skill_registry=skill_registry,
            config=config
        )

    def _run_flow(self, user_query: str, context: dict, steps: list[str]) -> TaskResult:
        # 1. Kế thừa Node Input Guard
        steps.append("Node 1: Input Guardrail (Kiểm duyệt an toàn đầu vào)")
        guard = self.node_input_guard(user_query)
        if not guard["is_safe"]:
            return self.node_format_output(answer=guard["refusal"])

        # 2. Vận hành vòng lặp ReAct
        steps.append("Node 2: ReAct Engine (Vòng lặp Suy luận & Hành động ReAct)")
        react_res = self.node_react_engine(user_query=user_query, context=context, max_turns=4)

        # Ghi nhận các bước suy luận và hành động vào trace
        for react_step in react_res.get("steps", []):
            steps.append(react_step)

        # 3. Kế thừa Node Format Output
        steps.append("Node 3: Format Output (Đóng gói kết quả đầu ra)")
        return self.node_format_output(
            answer=react_res["answer"],
            citations=react_res.get("citations", []),
            sources=react_res.get("sources", []),
            structured_data={"react_turns": len(react_res.get("steps", []))}
        )
