"""
Nhiệm vụ Xử lý Sự cố Khẩn cấp & Cấp Phiếu Tra Soát (Incident Support Task).
Kế thừa toàn bộ kho Node từ BaseBankingTask:
- Node 1: node_input_guard (Kiểm tra an toàn)
- Node 2: node_banking_tool (Khởi tạo mã phiếu hỗ trợ Ticket ID trên hệ thống)
- Node 3: node_format_output (Đóng gói thông điệp tiếp nhận & hướng dẫn khẩn cấp)

NGUYÊN TẮC: 0% Regex, Zero Ternary, No Typing Library, No __init__.py.
"""

from __future__ import annotations
from agent_banking.tasks.base_task import BaseBankingTask, TaskResult
from agent_banking.skills.skill_registry import SkillRegistry
from agent_banking.config import Config


class IncidentSupportTask(BaseBankingTask):
    """Task xử lý sự cố nuốt thẻ, mất thẻ và cấp phiếu tra soát."""

    def __init__(self, skill_registry: SkillRegistry | None = None, config: Config | None = None):
        super().__init__(
            task_id="TASK_INCIDENT_SUPPORT",
            task_name="Xử lý Sự cố Khẩn cấp & Cấp Phiếu Tra Soát",
            description="Ứng phó sự cố nuốt thẻ ATM, mất ví/thẻ, nghi ngờ giao dịch gian lận, hướng dẫn khóa thẻ và khởi tạo mã tra soát.",
            skill_registry=skill_registry,
            config=config
        )

    def _run_flow(self, user_query: str, context: dict, steps: list[str]) -> TaskResult:
        # 1. Kế thừa Node Input Guard
        steps.append("Node 1: Input Guardrail (Kiểm duyệt an toàn)")
        guard = self.node_input_guard(user_query)
        if not guard["is_safe"]:
            return self.node_format_output(answer=guard["refusal"])

        # Phân loại mức độ khẩn cấp
        q_lower = user_query.lower()
        issue_type = "tra_soat_giao_dich"
        urgency = "HIGH"

        if "nuốt" in q_lower:
            issue_type = "nuot_the_atm"
            urgency = "HIGH"
        elif "mất thẻ" in q_lower or "mất ví" in q_lower:
            issue_type = "mat_the_khoa_the"
            urgency = "CRITICAL"
        elif "lừa đảo" in q_lower or "gian lận" in q_lower or "hack" in q_lower:
            issue_type = "nghi_ngo_gian_lan"
            urgency = "CRITICAL"

        contact = context.get("user_contact", "KHACH_HANG_HOTLINE")
        card_num = context.get("card_number", "")

        # 2. Hướng dẫn hành động tức thời
        if urgency == "CRITICAL":
            action_guide = (
                "🚨 **HÀNH ĐỘNG KHẨN CẤP**:\n"
                "1. **Khóa thẻ trên app**: Mở VCB Digibank > Dịch vụ thẻ > Khóa thẻ tạm thời.\n"
                "2. **Gọi Hotline 24/7**: Liên hệ 1900 54 54 13 để nhân viên hỗ trợ phong tỏa ngay lập tức."
            )
        else:
            action_guide = (
                "ℹ️ **HƯỚNG DẪN XỬ LÝ NUỐT THẺ TẠI CÂY ATM**:\n"
                "1. Chờ 1-2 phút tại cây ATM để chắc chắn máy không nhả thẻ.\n"
                "2. Mở VCB Digibank > Khóa thẻ tạm thời để đảm bảo an toàn.\n"
                "3. Mang CCCD đến Chi nhánh quản lý cây ATM vào giờ hành chính để nhận lại thẻ."
            )

        # 3. Kế thừa Node Banking Tool (Tạo phiếu hỗ trợ Ticket ID)
        steps.append("Node 2: Banking Tool (Khởi tạo mã phiếu hỗ trợ tra soát trên hệ thống)")
        ticket_res = self.node_banking_tool("create_ticket", {
            "issue_type": issue_type,
            "user_contact": contact,
            "card_number": card_num,
            "details": user_query,
            "urgency": urgency
        })
        ticket_data = ticket_res.get("ticket_info", {})
        ticket_id = ticket_data.get("ticket_id", "TCK-VCB-HOTLINE")
        created_at = ticket_data.get("created_at", "Hôm nay")

        # 4. Kế thừa Node Format Output
        steps.append("Node 3: Format Output (Đóng gói thông điệp tiếp nhận & cam kết xử lý)")
        lines = [
            "Kính gửi Quý khách, Vietcombank đã tiếp nhận yêu cầu hỗ trợ sự cố của Quý khách.\n",
            action_guide,
            "\n📋 **THÔNG TIN PHIẾU TIẾP NHẬN TRA SOÁT**:",
            f"- Mã hồ sơ (Ticket ID): `{ticket_id}`",
            f"- Loại sự cố: `{issue_type}`",
            f"- Mức độ ưu tiên: `{urgency}`",
            f"- Thời gian ghi nhận: `{created_at}`",
            "- Tổng đài hỗ trợ 24/7: **1900 54 54 13**",
            "\nQuý khách an tâm, Vietcombank sẽ bảo vệ tối đa quyền lợi hợp pháp của Quý khách."
        ]
        return self.node_format_output(
            answer="\n".join(lines),
            structured_data={"ticket_id": ticket_id, "issue_type": issue_type, "urgency": urgency}
        )
