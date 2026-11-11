"""
Biometric Compliance Task (Decision 2345/QD-NHNN).
Inherits full node library from BaseBankingTask:
- Node 1: node_input_guard (Security check)
- Node 2: node_format_history (Format chat history)
- Node 3: node_parameter_extractor & node_banking_tool (Evaluate Decision 2345 thresholds)
- Node 4: node_knowledge_retrieval (Retrieve regulatory documents from RAG)
- Node 5: node_llm_generate (Generate biometric guidance & legal basis)
- Node 6: node_format_output (Package output result)

RULES: 0% Regex, Zero Ternary, No Typing Library, No __init__.py.
"""

from __future__ import annotations
from agent_banking.tasks.base_task import BaseBankingTask, TaskResult
from agent_banking.skills.skill_registry import SkillRegistry
from agent_banking.config.settings import Config


class BiometricComplianceTask(BaseBankingTask):
    """Task for biometric authentication compliance and digital payment security."""

    def __init__(self, skill_registry: SkillRegistry | None = None, config: Config | None = None):
        super().__init__(
            task_id="TASK_BIOMETRIC_COMPLIANCE",
            task_name="Biometric Compliance & Legal Assessment",
            description="Evaluates mandatory facial biometric requirements under Decision 2345/QD-NHNN and Circular 50/2024/TT-NHNN.",
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
        steps.append("Node 3: Parameter Extractor & Banking Tool (Evaluate Decision 2345)")
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

        # 4. Inherited Knowledge Retrieval Node (RAG)
        steps.append("Node 4: Knowledge Retrieval (Retrieve regulatory documents from RAG)")
        rag_data = self.node_knowledge_retrieval(f"Quyết định 2345 sinh trắc học khuôn mặt {user_query}", top_k=2)
        law_context = rag_data["context"]

        # 5. Inherited LLM Generate Node
        steps.append("Node 5: LLM Generation (Generate biometric guidance & legal basis)")
        sys_prompt = (
            "Bạn là Chuyên gia Pháp chế & An toàn Bảo mật Ngân hàng.\n"
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

        # 6. Inherited Format Output Node
        steps.append("Node 6: Format Output")
        return self.node_format_output(
            answer=answer,
            citations=rag_data["citations"],
            sources=rag_data["sources"],
            structured_data=comp_data
        )
