"""
Card Service & Rewards Advisory Task.
Inherits full node library from BaseBankingTask:
- Node 1: node_input_guard (Security check)
- Node 2: node_format_history (Format chat history)
- Node 3: node_knowledge_retrieval (Retrieve card tariffs and credit limits from RAG)
- Node 4: node_llm_generate (Generate card advisory report with citations)
- Node 5: node_format_output (Package output result)

RULES: 0% Regex, Zero Ternary, No Typing Library, No __init__.py.
"""

from __future__ import annotations
from agent_banking.tasks.base_task import BaseBankingTask, TaskResult
from agent_banking.skills.skill_registry import SkillRegistry
from agent_banking.config.settings import Config


class CardServiceTask(BaseBankingTask):
    """Task for card issuance advisory, international credit cards, and tariffs."""

    def __init__(self, skill_registry: SkillRegistry | None = None, config: Config | None = None):
        super().__init__(
            task_id="TASK_CARD_SERVICE",
            task_name="Card Services & Advisory Support",
            description="Advises on credit card eligibility, debit card benefits, annual fees, and rewards programs.",
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

        # Card type classification
        q_lower = user_query.lower()
        card_type = "Thẻ Quốc tế"
        if "nội địa" in q_lower:
            card_type = "Thẻ Ghi nợ Nội địa"
        elif "tín dụng" in q_lower or "visa" in q_lower:
            card_type = "Thẻ Tín dụng Quốc tế"

        # 3. Inherited Knowledge Retrieval Node (RAG)
        steps.append("Node 3: Knowledge Retrieval (Retrieve tariffs & card limits from RAG)")
        rag_data = self.node_knowledge_retrieval(f"{user_query} biểu phí phát hành phí thường niên hạn mức thẻ", top_k=3)
        specs_context = rag_data["context"]

        # 4. Inherited LLM Generate Node
        steps.append("Node 4: LLM Generation (Generate card advisory report with citations)")
        sys_prompt = (
            "Bạn là Chuyên gia Tư vấn Thẻ Ngân hàng.\n"
            "Tư vấn điều kiện mở thẻ, biểu phí thường niên, hạn mức và ưu đãi điểm thưởng."
        )
        user_prompt = f"LỊCH SỬ CHAT:\n{history_str}\n\nCÂU HỎI: {user_query}\n\nTÀI LIỆU BIỂU PHÍ RAG:\n{specs_context}"
        answer = self.node_llm_generate(sys_prompt, user_prompt, temperature=0.2)
        if not answer or answer == user_prompt:
            answer = f"Căn cứ biểu phí dịch vụ thẻ ngân hàng:\n{specs_context}"
        answer += "\n\n> 🔒 **Cảnh báo an toàn**: Tuyệt đối không cung cấp số CVV/CVC, mã PIN hoặc mã OTP cho bất kỳ ai."

        # 5. Inherited Format Output Node
        steps.append("Node 5: Format Output")
        return self.node_format_output(
            answer=answer,
            citations=rag_data["citations"],
            sources=rag_data["sources"],
            structured_data={"card_category": card_type}
        )
