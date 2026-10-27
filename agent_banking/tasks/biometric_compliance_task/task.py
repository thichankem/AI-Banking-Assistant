"""
Nhiệm vụ Thẩm định Sinh trắc học & Tuân thủ QĐ 2345/QĐ-NHNN (Biometric Compliance Task).
Kế thừa toàn bộ kho Node từ BaseBankingTask:
- Node 1: node_input_guard (Kiểm tra an toàn)
- Node 2: node_format_history (Format lịch sử hội thoại)
- Node 3: node_parameter_extractor & node_banking_tool (Thẩm định ngưỡng QĐ 2345)
- Node 4: node_knowledge_retrieval (Kéo văn bản quy phạm từ RAG)
- Node 5: node_llm_generate (Sinh chỉ dẫn Facepay & căn cứ pháp lý)
- Node 6: node_format_output (Đóng gói kết quả)

NGUYÊN TẮC: 0% Regex, Zero Ternary, No Typing Library, No __init__.py.
"""

from __future__ import annotations
from agent_banking.tasks.base_task import BaseBankingTask, TaskResult
from agent_banking.skills.skill_registry import SkillRegistry
from agent_banking.config import Config


class BiometricComplianceTask(BaseBankingTask):
    """Task thẩm định sinh trắc học và an toàn thanh toán số."""

    def __init__(self, skill_registry: SkillRegistry | None = None, config: Config | None = None):
        super().__init__(
            task_id="TASK_BIOMETRIC_COMPLIANCE",
            task_name="Thẩm định Sinh trắc học & Pháp chế QĐ 2345",
            description="Thẩm định điều kiện bắt buộc xác thực khuôn mặt Facepay theo Quyết định 2345/QĐ-NHNN và Thông tư 50/2024/TT-NHNN.",
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
        steps.append("Node 3: Parameter Extractor (Bóc tách số tiền giao dịch) & Banking Tool (Thẩm định ngưỡng QĐ 2345)")
        extracted = self.node_parameter_extractor(user_query)
        amount = context.get("transaction_amount", extracted.get("principal", 0.0))
        daily_amount = context.get("daily_accumulated_amount", 0.0)
        is_new_device = context.get("is_new_device", False)

        q_lower = user_query.lower()
        if "thiết bị mới" in q_lower or "đổi điện thoại" in q_lower or "máy mới" in q_lower:
            is_new_device = True

        comp_data = self.node_banking_tool("compliance_qd2345", {
            "transaction_amount": amount,
            "daily_accumulated_amount": daily_amount,
            "is_new_device": is_new_device
        })
        action_required = comp_data.get("action_required", "STANDARD_OTP")
        legal_basis = comp_data.get("legal_basis", "Quyết định 2345/QĐ-NHNN")

        # 4. Kế thừa Node Knowledge Retrieval (RAG)
        steps.append("Node 4: Knowledge Retrieval (Kéo văn bản quy phạm từ RAG)")
        rag_data = self.node_knowledge_retrieval(f"Quyết định 2345 sinh trắc học khuôn mặt {user_query}", top_k=2)
        law_context = rag_data["context"]

        # 5. Kế thừa Node LLM Generate
        steps.append("Node 5: LLM Generation (Sinh chỉ dẫn Facepay & căn cứ pháp lý)")
        sys_prompt = (
            "Bạn là Chuyên gia Pháp chế & An toàn Bảo mật Ngân hàng Vietcombank.\n"
            "Hãy giải thích rõ ràng căn cứ Quyết định 2345/QĐ-NHNN, nêu bật các mốc quan trọng "
            "(trên 10 triệu đồng/giao dịch hoặc trên 20 triệu đồng cộng dồn/ngày phải quét khuôn mặt qua CCCD gắn chip NFC)."
        )
        user_prompt = (
            f"LỊCH SỬ CHAT:\n{history_str}\n\n"
            f"CÂU HỎI: {user_query}\n\n"
            f"KẾT QUẢ THẨM ĐỊNH TỰ ĐỘNG: {action_required} (Căn cứ: {legal_basis})\n\n"
            f"VĂN BẢN QUY PHẠM (RAG):\n{law_context}"
        )
        answer = self.node_llm_generate(sys_prompt, user_prompt, temperature=0.1)
        if not answer or answer == user_prompt:
            answer = f"Căn cứ [{legal_basis}]: Yêu cầu xác thực sinh trắc học khuôn mặt qua CCCD gắn chip NFC đối với giao dịch trên 10 triệu đồng hoặc chuyển tiền cộng dồn trong ngày vượt 20 triệu đồng."

        # 6. Kế thừa Node Format Output
        steps.append("Node 6: Format Output (Đóng gói kết quả đầu ra)")
        return self.node_format_output(
            answer=answer,
            citations=rag_data["citations"],
            sources=rag_data["sources"],
            structured_data=comp_data
        )
