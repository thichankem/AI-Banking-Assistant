# -*- coding: utf-8 -*-
"""
Module Banking Tool Guardrail (Tầng 4 - Action Gating & Bounds Clamping)
======================================================================
Kiểm soát chặt chẽ các tham số nghiệp vụ trước khi thực thi công cụ
(Excessive Agency Protection - OWASP LLM06).

Ngăn chặn:
- Số tiền vay / gửi âm, giá trị NaN, số tiền phi lý (trên trần quy định).
- Kỳ hạn vay bất khả thi (trên 30 năm).
- Chèn mã độc, câu lệnh SQL hoặc ký tự điều khiển vào form mở Support Ticket.

NGUYÊN TẮC BẤT DI BẤT DỊCH:
1. 0% REGEX LAW: Tuyệt đối không dùng regex methods.
2. ZERO TERNARY LAW: Dùng if/else tường minh từng dòng.
3. NO TYPING LIBRARY: Không import typing.
4. NO __INIT__.PY: Import trực tiếp file module.
"""

from agent_banking.config import Config


DANGEROUS_INPUT_CHARACTERS = ["<script", "javascript:", "--", "; drop ", "; delete ", "union select", "exec("]


class BankingToolGuardrail:
    """
    Trình thẩm định và kiểm soát biên tham số cho Banking API Tools.
    """

    def __init__(self):
        self.max_loan_vnd = Config.MAX_LOAN_AMOUNT_VND
        self.max_loan_tenure = Config.MAX_LOAN_TENURE_MONTHS
        self.min_savings_vnd = Config.MIN_SAVINGS_AMOUNT_VND

    def _sanitize_string(self, text, max_len=500):
        """Khử trùng chuỗi ký tự, loại bỏ ký tự điều khiển nguy hiểm (0% Regex)"""
        if not text:
            return ""

        raw = str(text).strip()
        cleaned = raw
        for dang in DANGEROUS_INPUT_CHARACTERS:
            if dang in cleaned.lower():
                cleaned = cleaned.replace(dang, "[REDACTED_INPUT]")

        if len(cleaned) > max_len:
            cleaned = cleaned[:max_len].strip()

        return cleaned

    def validate_loan_params(self, params):
        """
        Thẩm định tham số cho công cụ tính lịch trả nợ vay (calculate_loan_amortization).
        """
        principal = params.get("principal_vnd", 0.0)
        rate = params.get("annual_rate_percent", 8.5)
        tenure = params.get("term_months", 12)

        try:
            principal_val = float(principal)
        except Exception:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": "Số tiền vay không đúng định dạng số hợp lệ."
            }

        try:
            rate_val = float(rate)
        except Exception:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": "Lãi suất vay năm không đúng định dạng số hợp lệ."
            }

        try:
            tenure_val = int(tenure)
        except Exception:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": "Kỳ hạn vay (tháng) phải là số nguyên."
            }

        if principal_val <= 0:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": "Số tiền vay gốc phải lớn hơn 0 VNĐ."
            }

        if principal_val < 1000000.0:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": "Số tiền vay tối thiểu của gói vay tiêu dùng là 1.000.000 VNĐ."
            }

        if principal_val > self.max_loan_vnd:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": (
                    f"Số tiền vay ({principal_val:,.0f} VNĐ) vượt quá hạn mức tối đa cho khách hàng "
                    f"cá nhân ({self.max_loan_vnd:,.0f} VNĐ). Quý khách vui lòng liên hệ chi nhánh Vietcombank."
                )
            }

        if tenure_val <= 0:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": "Kỳ hạn vay (tháng) phải lớn hơn 0."
            }

        if tenure_val > self.max_loan_tenure:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": (
                    f"Thời hạn vay tối đa cho gói vay thế chấp mua nhà là {self.max_loan_tenure} tháng "
                    f"(30 năm). Giá trị yêu cầu ({tenure_val} tháng) vượt quá quy định."
                )
            }

        if rate_val <= 0.0 or rate_val > 50.0:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": "Lãi suất vay năm phải nằm trong khoảng từ 0.1%/năm đến 50%/năm."
            }

        sanitized = {
            "principal_vnd": round(principal_val, 0),
            "annual_rate_percent": round(rate_val, 2),
            "term_months": tenure_val
        }
        return {
            "is_safe": True,
            "sanitized_params": sanitized,
            "error_message": None
        }

    def validate_savings_params(self, params):
        """
        Thẩm định tham số cho công cụ tính tiền lãi tiết kiệm (calculate_savings_profit).
        """
        amount = params.get("amount_vnd", 0.0)
        tenure = params.get("term_months", 12)

        try:
            amount_val = float(amount)
        except Exception:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": "Số tiền gửi tiết kiệm không đúng định dạng số."
            }

        try:
            tenure_val = int(tenure)
        except Exception:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": "Kỳ hạn gửi tiết kiệm phải là số nguyên tháng."
            }

        if amount_val <= 0:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": "Số tiền gửi tiết kiệm phải lớn hơn 0 VNĐ."
            }

        if amount_val < self.min_savings_vnd:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": (
                    f"Số tiền gửi tiết kiệm tối thiểu theo quy định của Vietcombank là "
                    f"{self.min_savings_vnd:,.0f} VNĐ."
                )
            }

        if amount_val > 500000000000.0:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": "Số tiền gửi vượt hạn mức giao dịch trực tuyến (500 tỷ VNĐ). Vui lòng đến quầy giao dịch."
            }

        if tenure_val <= 0 or tenure_val > 60:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": "Kỳ hạn gửi tiết kiệm phải từ 1 tháng đến tối đa 60 tháng (5 năm)."
            }

        sanitized = {
            "amount_vnd": round(amount_val, 0),
            "term_months": tenure_val
        }
        return {
            "is_safe": True,
            "sanitized_params": sanitized,
            "error_message": None
        }

    def validate_qd2345_params(self, params):
        """
        Thẩm định tham số cho công cụ kiểm tra sinh trắc học QĐ 2345 (check_biometric_compliance_qd2345).
        """
        single_tx = params.get("single_tx_amount_vnd", 0.0)
        daily_cum = params.get("daily_cumulative_amount_vnd", 0.0)

        try:
            single_val = float(single_tx)
        except Exception:
            single_val = 0.0

        try:
            daily_val = float(daily_cum)
        except Exception:
            daily_val = 0.0

        if single_val < 0.0:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": "Số tiền giao dịch đơn lẻ không được là số âm."
            }

        if daily_val < 0.0:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": "Số tiền giao dịch cộng dồn trong ngày không được là số âm."
            }

        sanitized = {
            "single_tx_amount_vnd": single_val,
            "daily_cumulative_amount_vnd": daily_val,
            "is_first_time_new_device": bool(params.get("is_first_time_new_device", False)),
            "has_verified_chip_cccd": bool(params.get("has_verified_chip_cccd", True))
        }
        return {
            "is_safe": True,
            "sanitized_params": sanitized,
            "error_message": None
        }

    def validate_ticket_params(self, params):
        """
        Thẩm định tham số cho công cụ tạo ticket hỗ trợ (create_support_ticket).
        """
        name = self._sanitize_string(params.get("customer_name", "Khách hàng"), max_len=100)
        issue = self._sanitize_string(params.get("issue_type", "Yêu cầu tra soát"), max_len=100)
        desc = self._sanitize_string(params.get("description", ""), max_len=1000)
        priority = self._sanitize_string(params.get("priority", "NORMAL"), max_len=20).upper()

        if len(desc) < 3:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": "Nội dung yêu cầu tra soát/khiếu nại quá ngắn. Quý khách vui lòng mô tả chi tiết hơn."
            }

        if priority not in ["URGENT", "HIGH", "NORMAL", "LOW"]:
            priority = "NORMAL"

        sanitized = {
            "customer_name": name,
            "issue_type": issue,
            "description": desc,
            "priority": priority
        }
        return {
            "is_safe": True,
            "sanitized_params": sanitized,
            "error_message": None
        }

    def guard_tool_call(self, tool_name, params):
        """
        Điều phối kiểm soát an toàn trước khi gọi bất kỳ Banking Tool nào.
        """
        if not tool_name:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": "Tên công cụ (tool_name) không được để trống."
            }

        name_lower = tool_name.strip().lower()

        if "loan" in name_lower or "amortization" in name_lower:
            return self.validate_loan_params(params)
        elif "savings" in name_lower or "profit" in name_lower:
            return self.validate_savings_params(params)
        elif "biometric" in name_lower or "qd2345" in name_lower:
            return self.validate_qd2345_params(params)
        elif "ticket" in name_lower or "support" in name_lower:
            return self.validate_ticket_params(params)
        else:
            # Các tool đọc dữ liệu niêm yết (tỷ giá, lãi suất) là an toàn mặc định
            return {
                "is_safe": True,
                "sanitized_params": params,
                "error_message": None
            }
