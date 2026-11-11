"""
Multi-Step ReAct Agent Task (Reasoning + Acting).
Standards: 0% Regex, Zero Ternary, No Typing Library, No __init__.py.
- Reads skills catalog from SkillRegistry.
- Operates ReAct loop: Thought -> Action -> Observation -> Final Answer.
- Resolves complex multi-step banking queries.
"""

from __future__ import annotations
from agent_banking.tasks.base_task import BaseBankingTask, TaskResult
from agent_banking.skills.skill_registry import SkillRegistry
from agent_banking.config.settings import Config


class ReActAgentTask(BaseBankingTask):
    """Task coordinating multi-step ReAct reasoning agent."""

    def __init__(self, skill_registry: SkillRegistry | None = None, config: Config | None = None):
        super().__init__(
            task_id="TASK_REACT_AGENT",
            task_name="Multi-Step ReAct Reasoning Agent",
            description="Reads skills from registry and iterates Thought -> Action -> Observation to resolve complex customer requests.",
            skill_registry=skill_registry,
            config=config
        )

    def _run_flow(self, user_query: str, context: dict, steps: list[str]) -> TaskResult:
        # 1. Inherited Input Guard Node
        steps.append("Node 1: Input Guardrail (Security check)")
        guard = self.node_input_guard(user_query)
        if not guard["is_safe"]:
            return self.node_format_output(answer=guard["refusal"])

        # 2. ReAct Engine Loop
        steps.append("Node 2: ReAct Engine (Reasoning + Acting loop)")
        react_res = self.node_react_engine(user_query=user_query, context=context, max_turns=4)

        # Log reasoning steps to trace
        for react_step in react_res.get("steps", []):
            steps.append(react_step)

        # 3. Inherited Format Output Node
        steps.append("Node 3: Format Output")
        return self.node_format_output(
            answer=react_res["answer"],
            citations=react_res.get("citations", []),
            sources=react_res.get("sources", []),
            structured_data={"react_turns": len(react_res.get("steps", []))}
        )
