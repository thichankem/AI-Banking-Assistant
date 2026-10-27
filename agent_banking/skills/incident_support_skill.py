"""
Kỹ năng Hỗ trợ Xử lý Sự cố Khẩn cấp & Tiếp nhận Tra soát (Incident Support Skill).
Thiết kế tối giản, trực quan, dễ hiểu theo chuẩn Chunking:
- Triage sự cố: Nuốt thẻ tại cây ATM, mất ví/thẻ, nghi ngờ giao dịch lừa đảo.
- Hướng dẫn phong tỏa tài khoản tức thì trên VCB Digibank hoặc tổng đài.
- Tự động cấp mã định danh hồ sơ tra soát (Ticket ID).

NGUYÊN TẮC BẮT BUỘC:
1. KHÔNG DÙNG __INIT__.PY: Import trực tiếp file module.
2. KHÔNG DÙNG THƯ VIỆN TYPING: Thuần túy dùng kiểu gốc của Python.
3. ZERO TERNARY: Bắt buộc dùng if/else tường minh, không dùng ?: hoặc toán tử 3 ngôi.
4. 0% REGEX LAW: Tuyệt đối không dùng regex.
5. HẠN CHẾ TỐI ĐA THỪA THÃI: Code ngắn gọn, trực diện, dễ hiểu.
"""

from __future__ import annotations

from .base_skill import BaseSkill, SkillMetadata
from agent_banking.tools.banking_api_tools import create_support_ticket


class IncidentSupportSkill(BaseSkill):
    """Kỹ năng tiếp nhận khiếu nại sự cố và cấp phiếu tra soát."""

    def __init__(self, config: object | None = None):
        meta = SkillMetadata(
            skill_id="incident_support_ticket",
            display_name="Xử lý Sự cố Khẩn cấp & Cấp Phiếu Tra Soát",
            description="Tiếp nhận tình huống khẩn cấp: nuốt thẻ, mất ví/thẻ, nghi ngờ gian lận, hướng dẫn khóa thẻ và cấp mã phiếu tra soát.",
            capabilities=[
                "emergency_card_lock_guidance",
                "atm_swallowed_card_triage",
                "support_ticket_creation"
            ],
            intent_triggers=[
                "nuốt thẻ", "bị nuốt thẻ", "mất thẻ", "mất ví",
                "khóa thẻ khẩn cấp", "lừa đảo", "gian lận", "tra soát"
            ],
            category="INCIDENT"
        )
        super().__init__(metadata=meta, config=config)

    def _run(self, params: dict) -> dict:
        issue_type = params.get("issue_type", "tra_soat_giao_dich")
        contact = params.get("user_contact", "KHACH_HANG_HOTLINE")
        card_num = params.get("card_number", "")
        details = params.get("details", "Tiếp nhận từ trợ lý AI.")
        urgency = params.get("urgency", "HIGH")

        issue_lower = issue_type.lower()
        if "mat_the" in issue_lower or "gian_lan" in issue_lower or "hack" in issue_lower:
            urgency = "CRITICAL"

        desc = details
        if card_num:
            desc = f"{details} | Thẻ/TK: {card_num}"

        priority_val = urgency
        if urgency == "CRITICAL":
            priority_val = "URGENT"

        ticket_data = create_support_ticket(
            customer_name=contact,
            issue_type=issue_type,
            description=desc,
            priority=priority_val
        )

        if urgency == "CRITICAL":
            guide = (
                "Quý khách vui lòng mở VCB Digibank > Dịch vụ thẻ > Khóa thẻ tạm thời ngay lập tức, "
                "hoặc gọi Tổng đài 1900 54 54 13 để khóa khẩn cấp."
            )
        else:
            guide = (
                "Hồ sơ tra soát đã được tiếp nhận thành công. Quý khách vui lòng lưu lại Mã hồ sơ "
                "để theo dõi tiến độ xử lý."
            )

        return {
            "ticket_info": ticket_data,
            "immediate_action_guide": guide,
            "hotline_support": "1900 54 54 13 (24/7)"
        }
