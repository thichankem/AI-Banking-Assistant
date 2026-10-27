"""
Nhiệm vụ Tư vấn Dịch vụ Thẻ & Ưu đãi Điểm thưởng (Card Service Task).
Kế thừa toàn bộ kho Node từ BaseBankingTask:
- Node 1: node_input_guard (Kiểm tra an toàn)
- Node 2: node_format_history (Format lịch sử hội thoại)
- Node 3: node_knowledge_retrieval (Kéo biểu phí & hạn mức phát hành thẻ từ RAG)
- Node 4: node_llm_generate (Sinh báo cáo tư vấn thẻ có trích dẫn)
- Node 5: node_format_output (Đóng gói kết quả)

NGUYÊN TẮC: 0% Regex, Zero Ternary, No Typing Library, No __init__.py.
"""

from __future__ import annotations
from agent_banking.tasks.base_task import BaseBankingTask, TaskResult
from agent_banking.skills.skill_registry import SkillRegistry
from agent_banking.config import Config


class CardServiceTask(BaseBankingTask):
    """Task tư vấn phát hành thẻ ghi nợ, thẻ tín dụng quốc tế và biểu phí Vietcombank."""

    def __init__(self, skill_registry: SkillRegistry | None = None, config: Config | None = None):
        super().__init__(
            task_id="TASK_CARD_SERVICE",
            task_name="Tư vấn & Hỗ trợ Dịch vụ Thẻ Vietcombank",
            description="Tư vấn điều kiện mở thẻ tín dụng, quyền lợi thẻ ghi nợ, biểu phí thường niên và chương trình điểm thưởng VCB Rewards.",
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

        # Phân loại thẻ
        q_lower = user_query.lower()
        card_type = "Thẻ Quốc tế"
        if "nội địa" in q_lower:
            card_type = "Thẻ Ghi nợ Nội địa"
        elif "tín dụng" in q_lower or "visa" in q_lower:
            card_type = "Thẻ Tín dụng Quốc tế"

        # 3. Kế thừa Node Knowledge Retrieval (RAG)
        steps.append("Node 3: Knowledge Retrieval (Kéo biểu phí & hạn mức phát hành thẻ từ RAG)")
        rag_data = self.node_knowledge_retrieval(f"{user_query} biểu phí phát hành phí thường niên hạn mức thẻ", top_k=3)
        specs_context = rag_data["context"]

        # 4. Kế thừa Node LLM Generate
        steps.append("Node 4: LLM Generation (Sinh báo cáo tư vấn thẻ có trích dẫn)")
        sys_prompt = (
            "Bạn là Chuyên gia Tư vấn Thẻ Vietcombank.\n"
            "Tư vấn điều kiện mở thẻ, biểu phí thường niên, hạn mức và ưu đãi điểm thưởng VCB Rewards."
        )
        user_prompt = f"LỊCH SỬ CHAT:\n{history_str}\n\nCÂU HỎI: {user_query}\n\nTÀI LIỆU BIỂU PHÍ RAG:\n{specs_context}"
        answer = self.node_llm_generate(sys_prompt, user_prompt, temperature=0.2)
        if not answer or answer == user_prompt:
            answer = f"Căn cứ biểu phí dịch vụ thẻ Vietcombank:\n{specs_context}"
        answer += "\n\n> 🔒 **Cảnh báo an toàn**: Tuyệt đối không cung cấp số CVV/CVC, mã PIN hoặc mã OTP cho bất kỳ ai."

        # 5. Kế thừa Node Format Output
        steps.append("Node 5: Format Output (Đóng gói kết quả đầu ra)")
        return self.node_format_output(
            answer=answer,
            citations=rag_data["citations"],
            sources=rag_data["sources"],
            structured_data={"card_category": card_type}
        )
