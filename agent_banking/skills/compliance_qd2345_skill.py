"""
Kỹ năng Thẩm định Sinh trắc học QĐ 2345 & TT 50 (Compliance QD2345 Skill).
Thiết kế tối giản, trực quan, dễ hiểu theo chuẩn Chunking:
- Thẩm định ngưỡng chuyển tiền trực tuyến: >10 triệu/lần hoặc >20 triệu/ngày.
- Kiểm tra thiết bị mới hoặc kích hoạt tài khoản eKYC.
- Đối soát nghĩa vụ quét khuôn mặt (Facepay) qua CCCD gắn chip NFC.

NGUYÊN TẮC BẮT BUỘC:
1. KHÔNG DÙNG __INIT__.PY: Import trực tiếp file module.
2. KHÔNG DÙNG THƯ VIỆN TYPING: Thuần túy dùng kiểu gốc của Python.
3. ZERO TERNARY: Bắt buộc dùng if/else tường minh, không dùng ?: hoặc toán tử 3 ngôi.
4. 0% REGEX LAW: Tuyệt đối không dùng regex.
5. HẠN CHẾ TỐI ĐA THỪA THÃI: Code ngắn gọn, trực diện, dễ hiểu.
"""

from __future__ import annotations

from .base_skill import BaseSkill, SkillMetadata
from agent_banking.tools.banking_api_tools import check_biometric_compliance_qd2345


class ComplianceQD2345Skill(BaseSkill):
    """Kỹ năng thẩm định tuân thủ sinh trắc học và an toàn thanh toán số."""

    def __init__(self, config: object | None = None):
        meta = SkillMetadata(
            skill_id="compliance_qd2345_audit",
            display_name="Thẩm định Sinh trắc học QĐ 2345 & TT 50",
            description="Kiểm tra ngưỡng giao dịch (>10tr hoặc >20tr/ngày), thiết bị mới để xác định nghĩa vụ quét khuôn mặt CCCD gắn chip qua NFC theo quy chuẩn NHNN.",
            capabilities=[
                "biometric_threshold_validation",
                "new_device_security_audit",
                "chip_id_nfc_status_check"
            ],
            intent_triggers=[
                "sinh trắc học", "khuôn mặt", "facepay", "2345", "thông tư 50",
                "trên 10 triệu", "trên 20 triệu", "cccd gắn chip", "nfc", "thiết bị mới"
            ],
            category="COMPLIANCE"
        )
        super().__init__(metadata=meta, config=config)

    def _run(self, params: dict) -> dict:
        amount = params.get("transaction_amount", 0.0)
        daily_amount = params.get("daily_accumulated_amount", 0.0)
        is_new_device = params.get("is_new_device", False)
        has_cccd = params.get("has_cccd_chip_nfc", True)

        audit_result = check_biometric_compliance_qd2345(
            single_tx_amount_vnd=float(amount),
            daily_cumulative_amount_vnd=float(daily_amount),
            is_first_time_new_device=bool(is_new_device),
            has_verified_chip_cccd=bool(has_cccd)
        )

        return {
            "compliance_status": "SUCCESS",
            "requires_biometric": audit_result.get("requires_biometric", False),
            "action_required": audit_result.get("mandatory_action", ""),
            "regulatory_rules_triggered": audit_result.get("trigger_reasons", []),
            "legal_basis": audit_result.get("regulation_reference", "Quyết định 2345/QĐ-NHNN"),
            "full_audit_details": audit_result
        }
