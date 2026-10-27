"""
Nhiệm vụ Tư vấn Tổng quan & Chăm sóc Khách hàng (General Chat Task).
Kế thừa toàn bộ kho Node từ BaseBankingTask:
- Node 1: node_input_guard (Kiểm tra an toàn)
- Node 2: node_format_history (Format lịch sử hội thoại)
- Node 3: node_llm_generate (Sinh câu trả lời chuẩn mực văn hóa Vietcombank)
- Node 4: node_format_output (Đóng gói kết quả)

NGUYÊN TẮC: 0% Regex, Zero Ternary, No Typing Library, No __init__.py.
"""

from __future__ import annotations
from agent_banking.tasks.base_task import BaseBankingTask, TaskResult
from agent_banking.skills.skill_registry import SkillRegistry
from agent_banking.config import Config


class GeneralChatTask(BaseBankingTask):
    """Task xử lý hội thoại chào hỏi, cảm ơn và tư vấn thông tin dịch vụ chung."""

    def __init__(self, skill_registry: SkillRegistry | None = None, config: Config | None = None):
        super().__init__(
            task_id="TASK_GENERAL_CHAT",
            task_name="Tư vấn Tổng quan & Chăm sóc Khách hàng",
            description="Chào hỏi, hướng dẫn sử dụng trợ lý AI và giải đáp thắc mắc dịch vụ tổng quan của Vietcombank.",
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

        # Phản hồi nhanh các mẫu chào hỏi phổ biến
        clean_q = user_query.strip().lower()
        greetings = ["xin chào", "chào bạn", "hello", "hi", "chào em", "alo", "chào"]
        for g in greetings:
            if clean_q == g or clean_q.startswith(g):
                answer = (
                    "Kính chào Quý khách! Em là Trợ lý AI Thông minh của Ngân hàng TMCP Ngoại thương Việt Nam (Vietcombank).\n\n"
                    "Em có thể hỗ trợ Quý khách giải quyết các nhu cầu tài chính đa dạng:\n"
                    "1. 📖 **Tra cứu quy chế & biểu phí**: Hạn mức giao dịch eKYC, phí thường niên thẻ, điều khoản mở tài khoản.\n"
                    "2. 💰 **Tư vấn vay vốn & Lập lịch trả nợ**: Tính khấu hao dư nợ giảm dần chi tiết từng tháng.\n"
                    "3. 🛡️ **Thẩm định sinh trắc học QĐ 2345**: Kiểm tra điều kiện quét khuôn mặt Facepay khi chuyển tiền trên 10 triệu.\n"
                    "4. 💳 **Tư vấn dịch vụ Thẻ**: Hạn mức thẻ tín dụng, ưu đãi điểm thưởng VCB Rewards.\n"
                    "5. 🚨 **Xử lý sự cố khẩn cấp**: Khóa thẻ khẩn cấp, tiếp nhận sự cố nuốt thẻ ATM và cấp mã phiếu tra soát.\n\n"
                    "Quý khách đang cần em hỗ trợ nội dung nào ạ?"
                )
                steps.append("Node 3: Quick Greeting Response")
                return self.node_format_output(answer=answer)

        # 3. Kế thừa Node LLM Generate cho các câu hỏi xã giao khác
        steps.append("Node 3: LLM Generation (Sinh câu trả lời chuẩn mực văn hóa Vietcombank)")
        sys_prompt = (
            "Bạn là Trợ lý AI Chăm sóc Khách hàng của Ngân hàng Ngoại thương Việt Nam (Vietcombank).\n"
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

        # 4. Kế thừa Node Format Output
        steps.append("Node 4: Format Output (Đóng gói kết quả)")
        return self.node_format_output(answer=final_answer)
