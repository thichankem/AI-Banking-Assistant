"""
Loan Advisory & Declining Balance Amortization Task.
Inherits full node library from BaseBankingTask:
- Node 1: node_input_guard (Security check)
- Node 2: node_format_history (Format chat history)
- Node 3: node_parameter_extractor & node_banking_tool (Calculate declining balance amortization)
- Node 4: node_knowledge_retrieval (Retrieve lending policy & collateral requirements from RAG)
- Node 5: node_llm_generate (Generate professional loan advisory report)
- Node 6: node_format_output (Package output result)

RULES: 0% Regex, Zero Ternary, No Typing Library, No __init__.py.
"""

from __future__ import annotations
from agent_banking.tasks.base_task import BaseBankingTask, TaskResult
from agent_banking.skills.skill_registry import SkillRegistry
from agent_banking.config.settings import Config


class LoanAdvisoryTask(BaseBankingTask):
    """Task for loan advisory and detailed amortization schedule calculation."""

    def __init__(self, skill_registry: SkillRegistry | None = None, config: Config | None = None):
        super().__init__(
            task_id="TASK_LOAN_ADVISORY",
            task_name="Loan Advisory & Declining Balance Amortization",
            description="Advises on borrowing conditions, mortgage profiles, and calculates amortization schedules.",
            skill_registry=skill_registry,
            config=config
        )

    def _run_flow(self, user_query: str, context: dict, steps: list[str]) -> TaskResult:
        # 1. Inherited Input Guard Node
        steps.append("Node 1: Input Guardrail (Security check)")
        guard = self.node_input_guard(user_query)
        if not guard["is_safe"]:
            return self.node_format_output(answer=guard["refusal"])

        # 2. Inherited Format History Node
        steps.append("Node 2: Format History")
        history_str = self.node_format_history(context.get("chat_history"))

        # 3. Inherited Parameter Extractor & Banking Tool Nodes
        steps.append("Node 3: Parameter Extractor & Banking Tool (Calculate amortization)")
        extracted = self.node_parameter_extractor(user_query)
        principal = context.get("principal", extracted.get("principal", 500000000.0))
        term_months = context.get("term_months", extracted.get("term_months", 12))
        annual_rate = context.get("annual_rate_percent", extracted.get("annual_rate_percent", 7.5))

        calc_result = self.node_banking_tool("loan_amortization", {
            "principal": principal,
            "term_months": term_months,
            "annual_rate_percent": annual_rate
        })
        calc_data = calc_result.get("calculation_data", {})
        schedule_md = calc_data.get("schedule_markdown", "")
        summary = calc_data.get("summary", {})

        # 4. Inherited Knowledge Retrieval Node (RAG)
        steps.append("Node 4: Knowledge Retrieval (RAG lending policy lookup)")
        rag_data = self.node_knowledge_retrieval("điều kiện cho vay khách hàng cá nhân tài sản bảo đảm hồ sơ thu nhập", top_k=2)
        policy_context = rag_data["context"]

        # 5. Inherited LLM Generate Node
        steps.append("Node 5: LLM Generation (Generate loan advisory report)")
        sys_prompt = (
            "Bạn là Chuyên viên Tín dụng Ngân hàng.\n"
            "Hãy tư vấn vay vốn chuyên nghiệp: giải thích phương pháp Dư nợ giảm dần, "
            "trình bày các con số tài chính và đính kèm bảng lịch trả nợ mẫu."
        )
        user_prompt = (
            f"LỊCH SỬ CHAT:\n{history_str}\n\n"
            f"CÂU HỎI: {user_query}\n\n"
            f"QUY CHẾ VAY (RAG):\n{policy_context}\n\n"
            f"BẢNG LỊCH TRẢ NỢ:\n{schedule_md}\n\n"
            f"TỔNG KẾT: {summary}"
        )
        answer = self.node_llm_generate(sys_prompt, user_prompt, temperature=0.2)
        if not answer or answer == user_prompt:
            answer = schedule_md
        answer += "\n\n> ⚠️ **Lưu ý**: Lãi suất mang tính chất tham khảo. Quý khách vui lòng cân đối thu nhập để trả nợ đúng hạn, tránh ảnh hưởng lịch sử tín dụng tại CIC."

        # 6. Inherited Format Output Node
        steps.append("Node 6: Format Output")
        return self.node_format_output(
            answer=answer,
            citations=rag_data["citations"],
            sources=rag_data["sources"],
            structured_data={"principal": principal, "term_months": term_months, "annual_rate": annual_rate, "summary": summary}
        )
