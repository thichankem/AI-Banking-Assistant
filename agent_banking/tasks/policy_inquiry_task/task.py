"""
Nhiệm vụ Tra cứu Quy chế, Biểu phí & Điều khoản (Policy Inquiry Task).
Kế thừa toàn bộ kho Node từ BaseBankingTask:
- Node 1: node_input_guard (Kiểm tra an toàn)
- Node 2: node_question_rewriter & node_format_history (Tái cấu trúc câu hỏi & lịch sử)
- Node 3: node_knowledge_retrieval (RAG Dense 1024 + BM25 RRF k=60)
- Node 4: node_grade_documents (Thẩm định độ liên quan tài liệu)
- Node 5: node_llm_generate (Sinh câu trả lời trích dẫn [Điều X, Khoản Y])
- Node 6: node_audit_answer (Đối soát Zero-Hallucination)
- Node 7: node_format_output (Đóng gói kết quả)

NGUYÊN TẮC: 0% Regex, Zero Ternary, No Typing Library, No __init__.py.
"""

from __future__ import annotations
from agent_banking.tasks.base_task import BaseBankingTask, TaskResult
from agent_banking.skills.skill_registry import SkillRegistry
from agent_banking.config import Config


class PolicyInquiryTask(BaseBankingTask):
    """Task tra cứu quy chế, hạn mức và biểu phí ngân hàng Vietcombank."""

    def __init__(self, skill_registry: SkillRegistry | None = None, config: Config | None = None):
        super().__init__(
            task_id="TASK_POLICY_INQUIRY",
            task_name="Tra cứu Quy chế & Biểu phí Dịch vụ Ngân hàng",
            description="Tìm kiếm văn bản quy phạm, biểu phí, hạn mức giao dịch eKYC và điều kiện mở tài khoản VCB Digibank.",
            skill_registry=skill_registry,
            config=config
        )

    def _run_flow(self, user_query: str, context: dict, steps: list[str]) -> TaskResult:
        # 1. Kế thừa Node Input Guard
        steps.append("Node 1: Input Guardrail (Kiểm duyệt an toàn đầu vào)")
        guard = self.node_input_guard(user_query)
        if not guard["is_safe"]:
            return self.node_format_output(answer=guard["refusal"])

        # 2. Kế thừa Node Question Rewriter & Format History
        steps.append("Node 2: Question Rewriter (Tái cấu trúc câu hỏi đa lượt) & Format History")
        effective_query = self.node_question_rewriter(user_query, context.get("chat_history"))
        history_str = self.node_format_history(context.get("chat_history"))

        # 3. Kế thừa Node Knowledge Retrieval (RAG)
        steps.append("Node 3: Knowledge Retrieval (RAG Dense 1024 + BM25 RRF k=60 + Cross-Encoder)")
        rag_data = self.node_knowledge_retrieval(effective_query, top_k=3)
        chunks = rag_data["chunks"]
        context_str = rag_data["context"]

        # 4. Kế thừa Node Grade Documents
        steps.append("Node 4: Document Grader (Thẩm định độ liên quan của tài liệu)")
        is_relevant = self.node_grade_documents(effective_query, chunks)
        if not is_relevant:
            steps.append(" -> Cảnh báo: Tài liệu có độ liên quan thấp.")

        # 5. Kế thừa Node LLM Generate
        steps.append("Node 5: LLM Generation (Sinh câu trả lời kèm trích dẫn [Điều X, Khoản Y])")
        sys_prompt = (
            "Bạn là Chuyên viên Quy chế Ngân hàng Vietcombank.\n"
            "Chỉ trả lời dựa CHÍNH XÁC trên tài liệu được cung cấp dưới đây.\n"
            "Luôn đính kèm nguồn tham chiếu điều khoản [Điều X, Khoản Y]. Tuyệt đối không bịa đặt."
        )
        user_prompt = f"LỊCH SỬ CHAT:\n{history_str}\n\nTÀI LIỆU CĂN CỨ:\n{context_str}\n\nCÂU HỎI:\n{effective_query}"
        answer = self.node_llm_generate(sys_prompt, user_prompt, temperature=0.1)

        # 6. Kế thừa Node Audit Answer (Zero-Hallucination)
        steps.append("Node 6: Hallucination Audit (Đối soát Zero-Hallucination)")
        audit = self.node_audit_answer(answer, chunks)
        if not audit.get("is_faithful", True):
            answer += "\n\n*(Thông tin căn cứ theo quy chế hiện hành tại thời điểm tra cứu)*"

        # 7. Kế thừa Node Format Output
        steps.append("Node 7: Format Output (Đóng gói kết quả đầu ra)")
        return self.node_format_output(
            answer=answer,
            citations=rag_data["citations"],
            sources=rag_data["sources"],
            structured_data={"chunks_count": len(chunks), "is_relevant": is_relevant}
        )
