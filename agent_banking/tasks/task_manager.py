"""
Banking Task Manager.
Minimalist, readable architecture:
- First Node (Intent Node): Calls TaskRouter to classify intent based on skills.
- Execution Coordinator: Hands off context to specialized task (sub-node chain).
- Unified Result Packaging: Returns response, citations, execution steps, and latency.

RULES:
1. NO __INIT__.PY: Direct module imports.
2. NO TYPING LIBRARY: Native Python types only.
3. ZERO TERNARY: Explicit if/else only.
4. 0% REGEX LAW: Absolutely no regular expressions.
5. KEEP IT SIMPLE: Clear, direct, readable code.
"""

from __future__ import annotations
import time

from agent_banking.skills.skill_registry import SkillRegistry
from agent_banking.core.settings import Config
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
    """End-to-end coordinator managing business tasks and skills."""

    def __init__(self, pipeline_instance=None, config: Config | None = None):
        if config is not None:
            self.config = config
        else:
            self.config = Config()

        # 1. Skill Registry
        self.skill_registry = SkillRegistry(pipeline_instance=pipeline_instance, config=self.config)

        # 2. First Node: Intent Router
        self.router = TaskRouter(skill_registry=self.skill_registry, config=self.config)

        # 3. Register business tasks
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
        Full-cycle processing:
        1. Node 0 (Intent Router): Select Task based on skills.
        2. Task Execution: Run sub-nodes inside Task.
        3. Consolidate unified result.
        """
        start_time = time.perf_counter()
        clean_q = user_query.strip()
        if not clean_q:
            return {
                "answer": "Kính chào Quý khách! Vui lòng nhập nội dung yêu cầu cần hỗ trợ.",
                "task_id": "NONE",
                "task_name": "Unknown",
                "execution_steps": [],
                "latency_ms": 0.0
            }

        # First Node: Intent Router
        decision = self.router.route(clean_q)

        # Execute chosen task
        task = self.tasks.get(decision.task_id, self.tasks["TASK_POLICY_INQUIRY"])
        task_res = task.execute(clean_q, context=context)

        total_latency = round((time.perf_counter() - start_time) * 1000, 2)

        # Consolidate execution steps
        combined_steps = [
            f"Node 0 (Intent Router): Routed to [{decision.task_id}] based on skills: {decision.target_skills} ({decision.reason})"
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
