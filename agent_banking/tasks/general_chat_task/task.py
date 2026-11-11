"""
General Chat & Customer Care Task.
Inherits full node library from BaseBankingTask:
- Node 1: node_input_guard (Security check)
- Node 2: node_format_history (Format chat history)
- Node 3: node_llm_generate (Generate banking-appropriate courteous response)
- Node 4: node_format_output (Package output result)

RULES: 0% Regex, Zero Ternary, No Typing Library, No __init__.py.
"""

from __future__ import annotations
from agent_banking.tasks.base_task import BaseBankingTask, TaskResult
from agent_banking.skills.skill_registry import SkillRegistry
from agent_banking.config.settings import Config


class GeneralChatTask(BaseBankingTask):
    """Task handling greetings, gratitude, and general banking service inquiries."""

    def __init__(self, skill_registry: SkillRegistry | None = None, config: Config | None = None):
        super().__init__(
            task_id="TASK_GENERAL_CHAT",
            task_name="General Chat & Customer Care",
            description="Handles greetings, user guidance, and general banking service assistance.",
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

        # Fast path for common greetings
        clean_q = user_query.strip().lower()
        greetings = ["xin chào", "chào bạn", "hello", "hi", "chào em", "alo", "chào"]
        for g in greetings:
            if clean_q == g or clean_q.startswith(g):
                answer = (
                    "Kính chào Quý khách! Em là Trợ lý AI Thông minh của Ngân hàng.\n\n"
                    "Em có thể hỗ trợ Quý khách giải quyết các nhu cầu tài chính đa dạng:\n"
                    "1. 📖 **Tra cứu quy chế & biểu phí**: Hạn mức giao dịch eKYC, phí thường niên thẻ, điều khoản mở tài khoản.\n"
                    "2. 💰 **Tư vấn vay vốn & Lập lịch trả nợ**: Tính khấu hao dư nợ giảm dần chi tiết từng tháng.\n"
                    "3. 🛡️ **Thẩm định sinh trắc học QĐ 2345**: Kiểm tra điều kiện quét khuôn mặt Facepay khi chuyển tiền trên 10 triệu.\n"
                    "4. 💳 **Tư vấn dịch vụ Thẻ**: Hạn mức thẻ tín dụng, ưu đãi điểm thưởng.\n"
                    "5. 🚨 **Xử lý sự cố khẩn cấp**: Khóa thẻ khẩn cấp, tiếp nhận sự cố nuốt thẻ ATM và cấp mã phiếu tra soát.\n\n"
                    "Quý khách đang cần em hỗ trợ nội dung nào ạ?"
                )
                steps.append("Node 3: Quick Greeting Response")
                return self.node_format_output(answer=answer)

        # 3. Inherited LLM Generate Node for general conversational inquiries
        steps.append("Node 3: LLM Generation (Generate banking-appropriate response)")
        sys_prompt = (
            "Bạn là Trợ lý AI Chăm sóc Khách hàng của Ngân hàng.\n"
            "Hãy giao tiếp lịch sự, chuẩn mực, ân cần và tôn trọng khách hàng."
        )

        user_prompt = user_query
        if history_str:
            user_prompt = f"LỊCH SỬ CHAT:\n{history_str}\n\nTin nhắn hiện tại của khách hàng: {user_query}"

        final_answer = self.node_llm_generate(
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=300
        )

        # 4. Inherited Format Output Node
        steps.append("Node 4: Format Output")
        return self.node_format_output(answer=final_answer)
