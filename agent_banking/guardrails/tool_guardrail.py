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

import math
from agent_banking.config.settings import Config


APPROVED_READONLY_TOOLS = [
    "get_interest_rates",
    "get_foreign_exchange_rates",
    "parse_vietnamese_financial_params",
    "check_complain_count",
    "transfer_to_agent",
    "rate_agent_csat",
    "deflect_competitor_comparison",
    "search_banking_knowledge"
]

SENSITIVE_UNAPPROVED_ACTIONS = [
    "transfer_money",
    "make_transfer",
    "send_money",
    "execute_payment",
    "withdraw_cash",
    "delete_account",
    "change_password",
    "reset_pin",
    "chuyen_khoan",
    "thanh_toan",
    "rut_tien",
    "xoa_tai_khoan"
]

DANGEROUS_INPUT_CHARACTERS = [
    "<script", "</script", "javascript:", "<iframe>", "</iframe>",
    "<style>", "</style>", "onerror=", "onload=", "window.location",
    "--", "; drop ", "; delete ", "union select", "exec(", "' or 1=1",
    "' or '1'='1", "/*", "*/", "xp_"
]


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

    def _sanitize_numeric(self, val, field_name, min_val=None, max_val=None, is_integer=False):
        """
        Xác thực và chuẩn hóa giá trị số học, triệt tiêu hoàn toàn NaN, Inf, -Inf, bool và chuỗi lỗi.
        Tuân thủ 100% GEMINI.md (0% regex, no typing, explicit if/else).
        Trả về: (is_valid, sanitized_val, error_message)
        """
        if val is None:
            return False, None, f"Trường '{field_name}' không được để trống (None)."

        # Trong Python, bool là lớp con của int (isinstance(True, int) là True). Bắt buộc chặn bool.
        if isinstance(val, bool):
            return False, None, f"Trường '{field_name}' không được là giá trị logic (True/False)."

        # Kiểm tra ép kiểu số thực
        try:
            f_val = float(val)
        except Exception:
            return False, None, f"Trường '{field_name}' không đúng định dạng số hợp lệ."

        # Kiểm tra NaN (val != val là đặc tính duy nhất của NaN trong chuẩn IEEE 754)
        if f_val != f_val:
            return False, None, f"Trường '{field_name}' không đúng định dạng số hợp lệ (phát hiện giá trị NaN)."

        if math.isnan(f_val):
            return False, None, f"Trường '{field_name}' không đúng định dạng số hợp lệ (phát hiện giá trị NaN)."

        # Kiểm tra vô cực (Inf / -Inf)
        if math.isinf(f_val):
            return False, None, f"Trường '{field_name}' không đúng định dạng số hợp lệ (phát hiện giá trị vô cực Inf)."

        # Nếu yêu cầu số nguyên (term_months)
        if is_integer is True:
            try:
                int_val = int(val)
            except Exception:
                return False, None, f"Trường '{field_name}' phải là số nguyên."
            # Đảm bảo không truyền số thực lẻ như 12.5 vào trường số nguyên
            if float(int_val) != f_val:
                return False, None, f"Trường '{field_name}' phải là số nguyên, không được chứa phần thập phân."
            target_val = int_val
        else:
            target_val = f_val

        # Kiểm tra cận dưới
        if min_val is not None:
            if target_val < min_val:
                return False, None, f"Trường '{field_name}' ({target_val}) nhỏ hơn giá trị tối thiểu cho phép ({min_val})."

        # Kiểm tra cận trên
        if max_val is not None:
            if target_val > max_val:
                return False, None, f"Trường '{field_name}' ({target_val}) vượt quá giá trị tối đa cho phép ({max_val})."

        return True, target_val, None

    def validate_loan_params(self, params):
        """
        Thẩm định tham số cho công cụ tính lịch trả nợ vay (calculate_loan_amortization).
        """
        if not isinstance(params, dict):
            return {
                "is_safe": False,
                "sanitized_params": {},
                "error_message": "Tham số đầu vào phải là một từ điển (dict)."
            }

        principal = params.get("principal_vnd", 0.0)
        rate = params.get("annual_rate_percent", 8.5)
        tenure = params.get("term_months", 12)

        # 1. Thẩm định số tiền vay: tối thiểu 1 triệu VNĐ, tối đa self.max_loan_vnd (50 tỷ)
        valid_p, p_val, err_p = self._sanitize_numeric(
            principal, "principal_vnd", min_val=1000000.0, max_val=self.max_loan_vnd, is_integer=False
        )
        if valid_p is False:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": err_p
            }

        # 2. Thẩm định kỳ hạn vay: tối thiểu 1 tháng, tối đa self.max_loan_tenure (360 tháng)
        valid_t, t_val, err_t = self._sanitize_numeric(
            tenure, "term_months", min_val=1, max_val=self.max_loan_tenure, is_integer=True
        )
        if valid_t is False:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": err_t
            }

        # 3. Thẩm định lãi suất năm: > 0.0% và <= 50.0%
        valid_r, r_val, err_r = self._sanitize_numeric(
            rate, "annual_rate_percent", min_val=0.01, max_val=50.0, is_integer=False
        )
        if valid_r is False:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": err_r
            }

        sanitized = {
            "principal_vnd": round(p_val, 0),
            "annual_rate_percent": round(r_val, 2),
            "term_months": t_val
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
        if not isinstance(params, dict):
            return {
                "is_safe": False,
                "sanitized_params": {},
                "error_message": "Tham số đầu vào phải là một từ điển (dict)."
            }

        amount = params.get("amount_vnd", 0.0)
        tenure = params.get("term_months", 12)

        # 1. Thẩm định số tiền gửi: tối thiểu self.min_savings_vnd (1 triệu VNĐ), tối đa 500 tỷ VNĐ
        valid_a, a_val, err_a = self._sanitize_numeric(
            amount, "amount_vnd", min_val=self.min_savings_vnd, max_val=500000000000.0, is_integer=False
        )
        if valid_a is False:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": err_a
            }

        # 2. Thẩm định kỳ hạn gửi: tối thiểu 1 tháng, tối đa 60 tháng (5 năm)
        valid_t, t_val, err_t = self._sanitize_numeric(
            tenure, "term_months", min_val=1, max_val=60, is_integer=True
        )
        if valid_t is False:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": err_t
            }

        sanitized = {
            "amount_vnd": round(a_val, 0),
            "term_months": t_val
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
        if not isinstance(params, dict):
            return {
                "is_safe": False,
                "sanitized_params": {},
                "error_message": "Tham số đầu vào phải là một từ điển (dict)."
            }

        single_tx = params.get("single_tx_amount_vnd", 0.0)
        daily_cum = params.get("daily_cumulative_amount_vnd", 0.0)

        # 1. Thẩm định số tiền giao dịch đơn lẻ: phải là số không âm, tối đa 500 tỷ
        valid_s, s_val, err_s = self._sanitize_numeric(
            single_tx, "single_tx_amount_vnd", min_val=0.0, max_val=500000000000.0, is_integer=False
        )
        if valid_s is False:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": err_s
            }

        # 2. Thẩm định số tiền cộng dồn trong ngày: phải là số không âm, tối đa 500 tỷ
        valid_d, d_val, err_d = self._sanitize_numeric(
            daily_cum, "daily_cumulative_amount_vnd", min_val=0.0, max_val=500000000000.0, is_integer=False
        )
        if valid_d is False:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": err_d
            }

        sanitized = {
            "single_tx_amount_vnd": s_val,
            "daily_cumulative_amount_vnd": d_val,
            "is_first_time_new_device": bool(params.get("is_first_time_new_device", False)),
            "has_verified_chip_cccd": bool(params.get("has_verified_chip_cccd", True))
        }
        return {
            "is_safe": True,
            "sanitized_params": sanitized,
            "error_message": None
        }

    def validate_decision_2345_transfer(self, params):
        """Alias tương thích ngược cho validate_qd2345_params theo Quyết định 2345/QĐ-NHNN."""
        return self.validate_qd2345_params(params)

    def validate_ticket_params(self, params):
        """
        Thẩm định tham số cho công cụ tạo ticket hỗ trợ (create_support_ticket).
        """
        if not isinstance(params, dict):
            return {
                "is_safe": False,
                "sanitized_params": {},
                "error_message": "Tham số đầu vào phải là một từ điển (dict)."
            }

        name = self._sanitize_string(params.get("customer_name", "Khách hàng"), max_len=100)
        issue = self._sanitize_string(params.get("issue_type", "Yêu cầu tra soát"), max_len=100)
        desc = self._sanitize_string(params.get("description", ""), max_len=1000)
        priority_raw = str(params.get("priority", "NORMAL")).strip().upper()

        if len(desc) < 3:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": "Nội dung yêu cầu tra soát/khiếu nại quá ngắn. Quý khách vui lòng mô tả chi tiết hơn."
            }

        if priority_raw not in ["URGENT", "HIGH", "NORMAL", "LOW"]:
            priority = "NORMAL"
        else:
            priority = priority_raw

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
        Áp dụng chính sách Fail-Closed tuyệt đối cho các công cụ ngoài danh mục và hành vi chuyển tiền.
        """
        if not tool_name:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": "Tên công cụ (tool_name) không được để trống."
            }

        name_lower = str(tool_name).strip().lower()

        # 1. Chặn các hành vi chuyển tiền hoặc thao tác nhạy cảm chưa được phê chuẩn
        for sensitive_act in SENSITIVE_UNAPPROVED_ACTIONS:
            if sensitive_act in name_lower:
                return {
                    "is_safe": False,
                    "sanitized_params": params,
                    "error_message": (
                        f"Hành vi giao dịch tài chính hoặc công cụ nhạy cảm '{tool_name}' "
                        f"yêu cầu xác thực bảo mật đa yếu tố và không được phép kích hoạt tự động qua AI Assistant."
                    )
                }

        # 2. Định tuyến theo nhóm công cụ tính toán và xử lý nghiệp vụ
        if "loan" in name_lower or "amortization" in name_lower:
            return self.validate_loan_params(params)
        elif "savings" in name_lower or "profit" in name_lower:
            return self.validate_savings_params(params)
        elif "biometric" in name_lower or "qd2345" in name_lower or "decision_2345" in name_lower:
            return self.validate_qd2345_params(params)
        elif "ticket" in name_lower or "support" in name_lower:
            return self.validate_ticket_params(params)

        # 3. Kiểm tra danh mục công cụ tra cứu / helper đã được phê duyệt (Approved Read-Only)
        is_approved_readonly = False
        for approved_tool in APPROVED_READONLY_TOOLS:
            if approved_tool in name_lower:
                is_approved_readonly = True
                break

        if is_approved_readonly is True:
            # Làm sạch chuỗi trong params nếu có
            sanitized_params = {}
            if isinstance(params, dict):
                for k, v in params.items():
                    if isinstance(v, str):
                        sanitized_params[k] = self._sanitize_string(v, max_len=500)
                    else:
                        sanitized_params[k] = v
            else:
                sanitized_params = params

            return {
                "is_safe": True,
                "sanitized_params": sanitized_params,
                "error_message": None
            }

        # 4. Fail-Closed: Chặn mọi công cụ không nhận diện được
        return {
            "is_safe": False,
            "sanitized_params": params,
            "error_message": (
                f"Công cụ '{tool_name}' không nằm trong danh mục công cụ ngân hàng được phê duyệt "
                f"(Chính sách Fail-Closed). Lời gọi công cụ bị từ chối an toàn."
            )
        }


_DEFAULT_TOOL_GUARDRAIL = None


def get_tool_guardrail():
    """Lấy hoặc khởi tạo singleton instance của BankingToolGuardrail."""
    global _DEFAULT_TOOL_GUARDRAIL
    if _DEFAULT_TOOL_GUARDRAIL is None:
        _DEFAULT_TOOL_GUARDRAIL = BankingToolGuardrail()
    return _DEFAULT_TOOL_GUARDRAIL


def guard_tool_call(tool_name, params):
    """Module-level wrapper cho guard_tool_call."""
    guardrail = get_tool_guardrail()
    return guardrail.guard_tool_call(tool_name, params)


def validate_loan_params(params):
    """Module-level wrapper cho validate_loan_params."""
    guardrail = get_tool_guardrail()
    return guardrail.validate_loan_params(params)


def validate_savings_params(params):
    """Module-level wrapper cho validate_savings_params."""
    guardrail = get_tool_guardrail()
    return guardrail.validate_savings_params(params)


def validate_qd2345_params(params):
    """Module-level wrapper cho validate_qd2345_params."""
    guardrail = get_tool_guardrail()
    return guardrail.validate_qd2345_params(params)


def validate_decision_2345_transfer(params):
    """Module-level wrapper cho validate_decision_2345_transfer."""
    guardrail = get_tool_guardrail()
    return guardrail.validate_decision_2345_transfer(params)

