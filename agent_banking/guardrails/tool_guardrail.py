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
4. NO __INIT__.PY: Direct module imports.
"""

import math

from agent_banking.core.settings import Config
from agent_banking.guardrails.config.settings import (
    GuardrailsConfig,
    load_guardrail_patterns,
    load_guardrail_prompts,
    load_guardrail_rules,
)


_patterns = load_guardrail_patterns()
_rules = load_guardrail_rules()
_tool_policy = dict(_rules.get("tool_policy") or {})
_family_keywords = dict(_tool_policy.get("tool_family_keywords") or {})
_bounds = dict(GuardrailsConfig.TOOL_BOUNDS_SETTINGS or {})
_prompts = load_guardrail_prompts()
_redaction_tokens = dict(getattr(_prompts, "REDACTION_TOKENS", {}) or {})
_tool_errors = dict(getattr(_prompts, "TOOL_ERROR_MESSAGES", {}) or {})

APPROVED_READONLY_TOOLS = list(_tool_policy.get("approved_readonly_tools") or [])
SENSITIVE_UNAPPROVED_ACTIONS = list(_tool_policy.get("sensitive_unapproved_actions") or [])
DANGEROUS_INPUT_CHARACTERS = list(_patterns.get("dangerous_input_characters") or [])

_SANITIZE_MAX_LEN = int(_bounds.get("sanitize_max_len", 0))
_REDACTION_TOOL_INPUT = str(_redaction_tokens.get("tool_input") or "")

_LOAN_BOUNDS = dict(_bounds.get("loan") or {})
_SAVINGS_BOUNDS = dict(_bounds.get("savings") or {})
_QD2345_BOUNDS = dict(_bounds.get("qd2345") or {})
_TICKET_BOUNDS = dict(_bounds.get("ticket") or {})


def _tool_err(key, **kwargs):
    template = str(_tool_errors.get(key) or "")
    if not template:
        return key
    return template.format(**kwargs)


def _matches_family(name_lower, family_key):
    keywords = _family_keywords.get(family_key) or []
    for kw in keywords:
        if kw in name_lower:
            return True
    return False


class BankingToolGuardrail:
    """
    Validator and parameter bounds clamper for Banking API Tools.
    """

    def __init__(self):
        self.max_loan_vnd = Config.MAX_LOAN_AMOUNT_VND
        self.max_loan_tenure = Config.MAX_LOAN_TENURE_MONTHS
        self.min_savings_vnd = Config.MIN_SAVINGS_AMOUNT_VND

    def _sanitize_string(self, text, max_len=None):
        """Sanitize text string, removing dangerous control characters (0% Regex)."""
        if not text:
            return ""

        if max_len is None:
            limit = _SANITIZE_MAX_LEN
        else:
            limit = max_len

        raw = str(text).strip()
        cleaned = raw
        for dang in DANGEROUS_INPUT_CHARACTERS:
            if dang in cleaned.lower():
                cleaned = cleaned.replace(dang, _REDACTION_TOOL_INPUT)

        if limit > 0 and len(cleaned) > limit:
            cleaned = cleaned[:limit].strip()

        return cleaned

    def _sanitize_numeric(self, val, field_name, min_val=None, max_val=None, is_integer=False):
        """
        Validate and normalize numerical values, rejecting NaN, Inf, -Inf, bool, and invalid formats.
        Returns: (is_valid, sanitized_val, error_message)
        """
        if val is None:
            return False, None, _tool_err("none_value", field_name=field_name)

        if isinstance(val, bool):
            return False, None, _tool_err("bool_value", field_name=field_name)

        try:
            f_val = float(val)
        except Exception:
            return False, None, _tool_err("invalid_number", field_name=field_name)

        if f_val != f_val:
            return False, None, _tool_err("nan_value", field_name=field_name)

        if math.isnan(f_val):
            return False, None, _tool_err("nan_value", field_name=field_name)

        if math.isinf(f_val):
            return False, None, _tool_err("inf_value", field_name=field_name)

        if is_integer is True:
            try:
                int_val = int(val)
            except Exception:
                return False, None, _tool_err("not_integer", field_name=field_name)
            if float(int_val) != f_val:
                return False, None, _tool_err("fractional_integer", field_name=field_name)
            target_val = int_val
        else:
            target_val = f_val

        if min_val is not None:
            if target_val < min_val:
                return False, None, _tool_err("below_min", field_name=field_name, target_val=target_val, min_val=min_val)

        if max_val is not None:
            if target_val > max_val:
                return False, None, _tool_err("above_max", field_name=field_name, target_val=target_val, max_val=max_val)

        return True, target_val, None

    def validate_loan_params(self, params):
        """
        Validate parameters for loan amortization calculator (calculate_loan_amortization).
        """
        if not isinstance(params, dict):
            return {
                "is_safe": False,
                "sanitized_params": {},
                "error_message": _tool_err("params_not_dict")
            }

        principal = params.get("principal_vnd", _LOAN_BOUNDS.get("default_principal_vnd"))
        rate = params.get("annual_rate_percent", _LOAN_BOUNDS.get("default_annual_rate_percent"))
        tenure = params.get("term_months", _LOAN_BOUNDS.get("default_term_months"))

        valid_p, p_val, err_p = self._sanitize_numeric(
            principal,
            "principal_vnd",
            min_val=_LOAN_BOUNDS.get("min_principal_vnd"),
            max_val=self.max_loan_vnd,
            is_integer=False
        )
        if valid_p is False:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": err_p
            }

        valid_t, t_val, err_t = self._sanitize_numeric(
            tenure,
            "term_months",
            min_val=_LOAN_BOUNDS.get("min_term_months"),
            max_val=self.max_loan_tenure,
            is_integer=True
        )
        if valid_t is False:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": err_t
            }

        valid_r, r_val, err_r = self._sanitize_numeric(
            rate,
            "annual_rate_percent",
            min_val=_LOAN_BOUNDS.get("min_annual_rate_percent"),
            max_val=_LOAN_BOUNDS.get("max_annual_rate_percent"),
            is_integer=False
        )
        if valid_r is False:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": err_r
            }

        p_digits = int(_LOAN_BOUNDS.get("principal_round_digits", 0))
        r_digits = int(_LOAN_BOUNDS.get("rate_round_digits", 0))
        sanitized = {
            "principal_vnd": round(p_val, p_digits),
            "annual_rate_percent": round(r_val, r_digits),
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
                "error_message": _tool_err("params_not_dict")
            }

        amount = params.get("amount_vnd", _SAVINGS_BOUNDS.get("default_amount_vnd"))
        tenure = params.get("term_months", _SAVINGS_BOUNDS.get("default_term_months"))

        valid_a, a_val, err_a = self._sanitize_numeric(
            amount,
            "amount_vnd",
            min_val=self.min_savings_vnd,
            max_val=_SAVINGS_BOUNDS.get("max_amount_vnd"),
            is_integer=False
        )
        if valid_a is False:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": err_a
            }

        valid_t, t_val, err_t = self._sanitize_numeric(
            tenure,
            "term_months",
            min_val=_SAVINGS_BOUNDS.get("min_term_months"),
            max_val=_SAVINGS_BOUNDS.get("max_term_months"),
            is_integer=True
        )
        if valid_t is False:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": err_t
            }

        a_digits = int(_SAVINGS_BOUNDS.get("amount_round_digits", 0))
        sanitized = {
            "amount_vnd": round(a_val, a_digits),
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
                "error_message": _tool_err("params_not_dict")
            }

        single_tx = params.get("single_tx_amount_vnd", _QD2345_BOUNDS.get("default_single_tx_amount_vnd"))
        daily_cum = params.get("daily_cumulative_amount_vnd", _QD2345_BOUNDS.get("default_daily_cumulative_amount_vnd"))

        valid_s, s_val, err_s = self._sanitize_numeric(
            single_tx,
            "single_tx_amount_vnd",
            min_val=_QD2345_BOUNDS.get("min_tx_amount_vnd"),
            max_val=_QD2345_BOUNDS.get("max_tx_amount_vnd"),
            is_integer=False
        )
        if valid_s is False:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": err_s
            }

        valid_d, d_val, err_d = self._sanitize_numeric(
            daily_cum,
            "daily_cumulative_amount_vnd",
            min_val=_QD2345_BOUNDS.get("min_tx_amount_vnd"),
            max_val=_QD2345_BOUNDS.get("max_tx_amount_vnd"),
            is_integer=False
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
            "is_first_time_new_device": bool(params.get("is_first_time_new_device", _QD2345_BOUNDS.get("default_is_first_time_new_device"))),
            "has_verified_chip_cccd": bool(params.get("has_verified_chip_cccd", _QD2345_BOUNDS.get("default_has_verified_chip_cccd")))
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
                "error_message": _tool_err("params_not_dict")
            }

        default_name = str(_TICKET_BOUNDS.get("default_customer_name") or "")
        default_issue = str(_TICKET_BOUNDS.get("default_issue_type") or "")
        default_priority = str(_TICKET_BOUNDS.get("default_priority") or "")
        allowed_priorities = list(_TICKET_BOUNDS.get("allowed_priorities") or [])

        name = self._sanitize_string(
            params.get("customer_name", default_name),
            max_len=int(_TICKET_BOUNDS.get("customer_name_max_len", 0))
        )
        issue = self._sanitize_string(
            params.get("issue_type", default_issue),
            max_len=int(_TICKET_BOUNDS.get("issue_type_max_len", 0))
        )
        desc = self._sanitize_string(
            params.get("description", ""),
            max_len=int(_TICKET_BOUNDS.get("description_max_len", 0))
        )
        priority_raw = str(params.get("priority", default_priority)).strip().upper()

        if len(desc) < int(_TICKET_BOUNDS.get("description_min_len", 0)):
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": _tool_err("ticket_description_too_short")
            }

        if priority_raw not in allowed_priorities:
            priority = default_priority
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

    def _guard_readonly(self, params):
        def scrub(value):
            if isinstance(value, str):
                return self._sanitize_string(value)
            if isinstance(value, list):
                return [scrub(item) for item in value]
            if isinstance(value, dict):
                return {key: scrub(item) for key, item in value.items()}
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Non-finite number")
            return value

        try:
            if not isinstance(params, dict):
                raise ValueError("Arguments must be an object")
            return {"is_safe": True, "sanitized_params": scrub(params), "error_message": None}
        except ValueError:
            return {"is_safe": False, "sanitized_params": {}, "error_message": _tool_err("readonly_invalid_args")}

    def guard_tool_call(self, tool_name, params, *, registered_readonly=False):
        """
        Coordinates safety checks before executing any Banking Tool.
        Enforces strict Fail-Closed policy for unauthorized tools and sensitive transactions.
        """
        if not tool_name:
            return {
                "is_safe": False,
                "sanitized_params": params,
                "error_message": _tool_err("empty_tool_name")
            }

        name_lower = str(tool_name).strip().lower()

        # 1. Block unauthorized money transfer or sensitive financial execution
        for sensitive_act in SENSITIVE_UNAPPROVED_ACTIONS:
            if sensitive_act in name_lower:
                return {
                    "is_safe": False,
                    "sanitized_params": params,
                    "error_message": _tool_err("sensitive_tool_blocked", tool_name=tool_name)
                }

        if registered_readonly:
            if name_lower == "present_visualization" and isinstance(params, dict):
                artifact = params.get("artifact")
                if isinstance(artifact, dict) and artifact.get("engine") == "mermaid" and isinstance(artifact.get("source"), str):
                    # Mermaid arrows contain SQL-like punctuation. Its dedicated schema
                    # validates the complete DSL before the tool returns it.
                    guarded = self._guard_readonly({**params, "artifact": {**artifact, "source": ""}})
                    if guarded["is_safe"]:
                        guarded["sanitized_params"]["artifact"]["source"] = artifact["source"]
                    return guarded
            return self._guard_readonly(params)

        # 2. Route by tool family
        if _matches_family(name_lower, "loan"):
            return self.validate_loan_params(params)
        elif _matches_family(name_lower, "savings"):
            return self.validate_savings_params(params)
        elif _matches_family(name_lower, "biometric"):
            return self.validate_qd2345_params(params)
        elif _matches_family(name_lower, "ticket"):
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
                        sanitized_params[k] = self._sanitize_string(v, max_len=_SANITIZE_MAX_LEN)
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
            "error_message": _tool_err("unrecognized_tool_blocked", tool_name=tool_name)
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
