# -*- coding: utf-8 -*-
"""
Banking Tool Guardrail Module (Layer 4 - Action Gating & Bounds Clamping)
========================================================================
Validates business parameters before executing tools (Excessive Agency Protection - OWASP LLM06).

Prevents:
- Negative loan / deposit amounts, NaN, Inf, unreasonable amounts exceeding ceiling.
- Impossible loan tenures (> 30 years).
- Code injection, SQL commands, or dangerous control characters in ticket forms.

DESIGN INVARIANTS:
1. 0% REGEX LAW: No regex methods or import re.
2. ZERO TERNARY LAW: Explicit if/else statements only.
3. NO TYPING LIBRARY: No typing library imports.
4. NO __INIT__.PY: Direct module imports.
"""

import math
from agent_banking.core.settings import Config


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
    Validator and parameter bounds clamper for Banking API Tools.
    """

    def __init__(self):
        self.max_loan_vnd = Config.MAX_LOAN_AMOUNT_VND
        self.max_loan_tenure = Config.MAX_LOAN_TENURE_MONTHS
        self.min_savings_vnd = Config.MIN_SAVINGS_AMOUNT_VND

    def _sanitize_string(self, text, max_len=500):
        """Sanitize text string, removing dangerous control characters (0% Regex)."""
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
        Validate and normalize numerical values, rejecting NaN, Inf, -Inf, bool, and invalid formats.
        Returns: (is_valid, sanitized_val, error_message)
        """
        if val is None:
            return False, None, f"Field '{field_name}' cannot be None."

        if isinstance(val, bool):
            return False, None, f"Field '{field_name}' cannot be a boolean (True/False)."

        try:
            f_val = float(val)
        except Exception:
            return False, None, f"Field '{field_name}' is not a valid number."

        if f_val != f_val:
            return False, None, f"Field '{field_name}' contains NaN value."

        if math.isnan(f_val):
            return False, None, f"Field '{field_name}' contains NaN value."

        if math.isinf(f_val):
            return False, None, f"Field '{field_name}' contains infinite value (Inf)."

        if is_integer is True:
            try:
                int_val = int(val)
            except Exception:
                return False, None, f"Field '{field_name}' must be an integer."
            if float(int_val) != f_val:
                return False, None, f"Field '{field_name}' must be an integer, cannot contain fractional parts."
            target_val = int_val
        else:
            target_val = f_val

        if min_val is not None:
            if target_val < min_val:
                return False, None, f"Field '{field_name}' ({target_val}) is below minimum allowed value ({min_val}) - dưới mức tối thiểu cho phép."

        if max_val is not None:
            if target_val > max_val:
                return False, None, f"Field '{field_name}' ({target_val}) vượt quá hạn mức tối đa / vượt quá quy định ({max_val})."

        return True, target_val, None

    def validate_loan_params(self, params):
        """
        Validate parameters for loan amortization calculator (calculate_loan_amortization).
        """
        if not isinstance(params, dict):
            return {
                "is_safe": False,
                "sanitized_params": {},
                "error_message": "Input parameters must be a dictionary."
            }

        principal = params.get("principal_vnd", 0.0)
        rate = params.get("annual_rate_percent", 8.5)
        tenure = params.get("term_months", 12)

        valid_p, p_val, err_p = self._sanitize_numeric(
            principal, "principal_vnd", min_val=1000000.0, max_val=self.max_loan_vnd, is_integer=False
        )
        if valid_p is False:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": err_p
            }

        valid_t, t_val, err_t = self._sanitize_numeric(
            tenure, "term_months", min_val=1, max_val=self.max_loan_tenure, is_integer=True
        )
        if valid_t is False:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": err_t
            }

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
        Validate parameters for savings profit calculator (calculate_savings_profit).
        """
        if not isinstance(params, dict):
            return {
                "is_safe": False,
                "sanitized_params": {},
                "error_message": "Input parameters must be a dictionary."
            }

        amount = params.get("amount_vnd", 0.0)
        tenure = params.get("term_months", 12)

        valid_a, a_val, err_a = self._sanitize_numeric(
            amount, "amount_vnd", min_val=self.min_savings_vnd, max_val=500000000000.0, is_integer=False
        )
        if valid_a is False:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": err_a
            }

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
        Validate parameters for biometric compliance check (check_biometric_compliance_qd2345).
        """
        if not isinstance(params, dict):
            return {
                "is_safe": False,
                "sanitized_params": {},
                "error_message": "Input parameters must be a dictionary."
            }

        single_tx = params.get("single_tx_amount_vnd", 0.0)
        daily_cum = params.get("daily_cumulative_amount_vnd", 0.0)

        valid_s, s_val, err_s = self._sanitize_numeric(
            single_tx, "single_tx_amount_vnd", min_val=0.0, max_val=500000000000.0, is_integer=False
        )
        if valid_s is False:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": err_s
            }

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
        """Backward compatibility alias for validate_qd2345_params."""
        return self.validate_qd2345_params(params)

    def validate_ticket_params(self, params):
        """
        Validate parameters for support ticket creation (create_support_ticket).
        """
        if not isinstance(params, dict):
            return {
                "is_safe": False,
                "sanitized_params": {},
                "error_message": "Input parameters must be a dictionary."
            }

        name = self._sanitize_string(params.get("customer_name", "Customer"), max_len=100)
        issue = self._sanitize_string(params.get("issue_type", "Inquiry"), max_len=100)
        desc = self._sanitize_string(params.get("description", ""), max_len=1000)
        priority_raw = str(params.get("priority", "NORMAL")).strip().upper()

        if len(desc) < 3:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": "Ticket description is too short. Please provide more details."
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
        Coordinates safety checks before executing any Banking Tool.
        Enforces strict Fail-Closed policy for unauthorized tools and sensitive transactions.
        """
        if not tool_name:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": "Tool name cannot be empty."
            }

        name_lower = str(tool_name).strip().lower()

        # 1. Block unauthorized money transfer or sensitive financial execution
        for sensitive_act in SENSITIVE_UNAPPROVED_ACTIONS:
            if sensitive_act in name_lower:
                return {
                    "is_safe": False,
                    "sanitized_params": params,
                    "error_message": (
                        f"Financial transaction or sensitive tool '{tool_name}' "
                        f"requires multi-factor authentication and cannot be triggered automatically by AI Assistant."
                    )
                }

        # 2. Route by tool family
        if "loan" in name_lower or "amortization" in name_lower:
            return self.validate_loan_params(params)
        elif "savings" in name_lower or "profit" in name_lower:
            return self.validate_savings_params(params)
        elif "biometric" in name_lower or "qd2345" in name_lower or "decision_2345" in name_lower:
            return self.validate_qd2345_params(params)
        elif "ticket" in name_lower or "support" in name_lower:
            return self.validate_ticket_params(params)

        # 3. Check approved read-only tools
        is_approved_readonly = False
        for approved_tool in APPROVED_READONLY_TOOLS:
            if approved_tool in name_lower:
                is_approved_readonly = True
                break

        if is_approved_readonly is True:
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

        # 4. Fail-Closed: Reject unrecognized tools
        return {
            "is_safe": False,
            "sanitized_params": params,
            "error_message": (
                f"Tool '{tool_name}' is not in the approved banking tool whitelist "
                f"(Fail-Closed policy). Tool call safely rejected."
            )
        }


_DEFAULT_TOOL_GUARDRAIL = None


def get_tool_guardrail():
    """Get or initialize singleton instance of BankingToolGuardrail."""
    global _DEFAULT_TOOL_GUARDRAIL
    if _DEFAULT_TOOL_GUARDRAIL is None:
        _DEFAULT_TOOL_GUARDRAIL = BankingToolGuardrail()
    return _DEFAULT_TOOL_GUARDRAIL


def guard_tool_call(tool_name, params):
    """Module-level wrapper for guard_tool_call."""
    guardrail = get_tool_guardrail()
    return guardrail.guard_tool_call(tool_name, params)


def validate_loan_params(params):
    """Module-level wrapper for validate_loan_params."""
    guardrail = get_tool_guardrail()
    return guardrail.validate_loan_params(params)


def validate_savings_params(params):
    """Module-level wrapper for validate_savings_params."""
    guardrail = get_tool_guardrail()
    return guardrail.validate_savings_params(params)


def validate_qd2345_params(params):
    """Module-level wrapper for validate_qd2345_params."""
    guardrail = get_tool_guardrail()
    return guardrail.validate_qd2345_params(params)


def validate_decision_2345_transfer(params):
    """Module-level wrapper for validate_decision_2345_transfer."""
    guardrail = get_tool_guardrail()
    return guardrail.validate_decision_2345_transfer(params)
