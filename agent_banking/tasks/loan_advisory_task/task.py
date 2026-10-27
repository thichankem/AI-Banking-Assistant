"""
Nhiệm vụ Tư vấn Gói Vay & Lịch Trả Nợ Giảm Dần (Loan Advisory Task).
Kế thừa toàn bộ kho Node từ BaseBankingTask:
- Node 1: node_input_guard (Kiểm tra an toàn)
- Node 2: node_format_history (Format lịch sử hội thoại)
- Node 3: node_parameter_extractor & node_banking_tool (Tính khấu hao dư nợ giảm dần)
- Node 4: node_knowledge_retrieval (Kéo quy chế cho vay & tài sản thế chấp từ RAG)
- Node 5: node_llm_generate (Sinh báo cáo tư vấn vay chuyên nghiệp)
- Node 6: node_format_output (Đóng gói kết quả)

NGUYÊN TẮC: 0% Regex, Zero Ternary, No Typing Library, No __init__.py.
"""

from __future__ import annotations
from agent_banking.tasks.base_task import BaseBankingTask, TaskResult
from agent_banking.skills.skill_registry import SkillRegistry
from agent_banking.config import Config


class LoanAdvisoryTask(BaseBankingTask):
    """Task tư vấn vay vốn và tính bảng lịch trả nợ chi tiết."""

    def __init__(self, skill_registry: SkillRegistry | None = None, config: Config | None = None):
        super().__init__(
            task_id="TASK_LOAN_ADVISORY",
            task_name="Tư vấn Gói Vay & Lịch Trả Nợ Dư Nợ Giảm Dần",
            description="Tư vấn điều kiện cho vay, hồ sơ thế chấp và tự động lập bảng tính khấu hao dư nợ giảm dần.",
            skill_registry=skill_registry,
            config=config
        )

    def _run_flow(self, user_query: str, context: dict, steps: list[str]) -> TaskResult:
        # 1. Kế thừa Node Input Guard
        steps.append("Node 1: Input Guardrail (Kiểm duyệt an toàn đầu vào)")
        guard = self.node_input_guard(user_query)
        if not guard["is_safe"]:
            return self.node_format_output(answer=guard["refusal"])

        # 2. Kế thừa Node Format History
        steps.append("Node 2: Format History (Chuẩn hóa lịch sử hội thoại)")
        history_str = self.node_format_history(context.get("chat_history"))

        # 3. Kế thừa Node Parameter Extractor & Banking Tool
        steps.append("Node 3: Parameter Extractor (Bóc tách tham số tài chính) & Banking Tool (Tính khấu hao)")
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

        # 4. Kế thừa Node Knowledge Retrieval (Kéo quy chế cho vay từ RAG)
        steps.append("Node 4: Knowledge Retrieval (RAG tra cứu điều kiện vay thế chấp)")
        rag_data = self.node_knowledge_retrieval("điều kiện cho vay khách hàng cá nhân tài sản bảo đảm hồ sơ thu nhập", top_k=2)
        policy_context = rag_data["context"]

        # 5. Kế thừa Node LLM Generate
        steps.append("Node 5: LLM Generation (Sinh báo cáo tư vấn vay vốn)")
        sys_prompt = (
            "Bạn là Chuyên viên Tín dụng Vietcombank.\n"
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

        # 6. Kế thừa Node Format Output
        steps.append("Node 6: Format Output (Đóng gói kết quả đầu ra)")
        return self.node_format_output(
            answer=answer,
            citations=rag_data["citations"],
            sources=rag_data["sources"],
            structured_data={"principal": principal, "term_months": term_months, "annual_rate": annual_rate, "summary": summary}
        )
