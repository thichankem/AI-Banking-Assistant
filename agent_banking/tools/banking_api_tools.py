# -*- coding: utf-8 -*-
"""
Banking API Tools (Digital Banking Business Toolset v4.0)
======================================================================
Provides execution tools for AI Agents, driven by gitignored YAML/prompt configs:
1. get_interest_rates: Deposit interest rates schedule.
2. get_foreign_exchange_rates: Foreign exchange rates today.
3. calculate_savings_profit: Automated savings profit calculator.
4. calculate_loan_amortization: Declining balance loan amortization schedule.
5. check_biometric_compliance_qd2345: Biometric compliance under Decision 2345 & Circular 50.
6. create_support_ticket: Create support ticket for urgent incidents.
7. parse_vietnamese_financial_params: Parse Vietnamese financial parameters.
8. check_complain_count: Track complaint counts & suggest human escalation.
9. transfer_to_agent: Live Agent handoff during business hours.
10. rate_agent_csat: Record CSAT ratings 1-5 stars.
11. deflect_competitor_comparison: Deflect competitor comparison inquiries.

RULES:
- 0% REGEX LAW: Absolutely no regex or re library imports.
- ZERO TERNARY LAW: Explicit if/else only.
- NO TYPING LIBRARY: Native Python types.
"""

import datetime
import time

from agent_banking.tools.config.prompts import COMPETITOR_DEFLECTION_DIRECTIVE, TOOL_TEMPLATES
from agent_banking.tools.config.settings import get_banking_data, get_query_rules

_DATA = get_banking_data()
_PARSER_RULES = get_query_rules().get("financial_param_parser", {})


def get_interest_rates(deposit_type: str = "vnd") -> dict:
    """Return the deposit interest rates used in the simulation."""
    cfg = _DATA["interest_rates"]
    dep_type = deposit_type.strip().lower()
    if dep_type in cfg["vnd_aliases"]:
        return {
            "status": "success",
            "currency": cfg["vnd_currency"],
            "effective_date": cfg["effective_date"],
            "bank": cfg["bank"],
            "interest_rates": [dict(row) for row in cfg["vnd_rates"]],
            "disclaimer": TOOL_TEMPLATES["interest_rates_vnd_disclaimer"]
        }
    return {
        "status": "success",
        "currency": cfg["usd_currency"],
        "effective_date": cfg["effective_date"],
        "bank": cfg["bank"],
        "interest_rates": [dict(row) for row in cfg["usd_rates"]],
        "disclaimer": TOOL_TEMPLATES["interest_rates_usd_disclaimer"]
    }


def get_foreign_exchange_rates(currency: str = "USD") -> dict:
    """Return the foreign exchange rates used in the simulation."""
    cfg = _DATA["foreign_exchange_rates"]
    curr = currency.strip().upper()
    rate_table = {code: dict(vals) for code, vals in cfg["rates"].items()}

    if curr in rate_table:
        selected_rate = {curr: rate_table[curr]}
    else:
        selected_rate = rate_table

    return {
        "status": "success",
        "effective_date": cfg["effective_date"],
        "bank": cfg["bank"],
        "rates": selected_rate,
        "note": TOOL_TEMPLATES["fx_rates_note"]
    }


def calculate_savings_profit(amount_vnd: float, term_months: int) -> dict:
    """
    Calculate estimated savings profit and total maturity balance.
    Formula: Interest = Deposit amount * Rate (%/year) * (Months / 12)
    """
    if amount_vnd <= 0:
        return {
            "status": "ERROR",
            "message": TOOL_TEMPLATES["savings_amount_error"]
        }

    sav_cfg = _DATA["savings_profit"]
    rate = float(sav_cfg["default_rate_percent"])
    for tier in sav_cfg["tiers"]:
        if term_months < int(tier["max_term_months"]):
            rate = float(tier["rate_percent"])
            break

    interest_earned = amount_vnd * (rate / 100.0) * (term_months / 12.0)
    total_received = amount_vnd + interest_earned

    return {
        "status": "SUCCESS",
        "principal_amount_vnd": round(amount_vnd, 0),
        "term_months": term_months,
        "applicable_rate_percent": rate,
        "interest_earned_vnd": round(interest_earned, 0),
        "total_at_maturity_vnd": round(total_received, 0),
        "formatted_summary": TOOL_TEMPLATES["savings_summary"].format(
            amount_vnd=amount_vnd,
            term_months=term_months,
            rate=rate,
            interest_earned=interest_earned,
            total_received=total_received
        )
    }


def calculate_loan_amortization(
    principal_vnd: float,
    annual_rate_percent: float,
    term_months: int
) -> dict:
    """
    Calculate loan repayment schedule using declining balance amortization method.
    """
    if principal_vnd <= 0:
        return {
            "status": "ERROR",
            "message": TOOL_TEMPLATES["loan_principal_error"]
        }

    if term_months <= 0:
        return {
            "status": "ERROR",
            "message": TOOL_TEMPLATES["loan_term_error"]
        }

    preview_months = int(_DATA["loan_amortization"]["preview_leading_months"])
    monthly_principal = principal_vnd / float(term_months)
    monthly_rate = (annual_rate_percent / 100.0) / 12.0

    remaining_balance = principal_vnd
    total_interest = 0.0
    schedule_preview = []

    for month_idx in range(1, term_months + 1):
        interest_for_month = remaining_balance * monthly_rate
        total_payment_for_month = monthly_principal + interest_for_month
        total_interest = total_interest + interest_for_month

        should_record = False
        if month_idx <= preview_months:
            should_record = True
        elif month_idx == term_months:
            should_record = True

        if should_record is True:
            schedule_preview.append({
                "month": month_idx,
                "opening_balance_vnd": round(remaining_balance, 0),
                "principal_payment_vnd": round(monthly_principal, 0),
                "interest_payment_vnd": round(interest_for_month, 0),
                "total_monthly_payment_vnd": round(total_payment_for_month, 0)
            })

        remaining_balance = remaining_balance - monthly_principal

    total_payable = principal_vnd + total_interest
    first_month_payment = 0.0
    if len(schedule_preview) > 0:
        first_month_payment = schedule_preview[0]["total_monthly_payment_vnd"]

    return {
        "status": "SUCCESS",
        "calculation_method": TOOL_TEMPLATES["loan_calculation_method"],
        "principal_vnd": round(principal_vnd, 0),
        "annual_rate_percent": annual_rate_percent,
        "term_months": term_months,
        "first_month_payment_vnd": first_month_payment,
        "total_interest_vnd": round(total_interest, 0),
        "total_payable_vnd": round(total_payable, 0),
        "schedule_sample": schedule_preview
    }


def check_biometric_compliance_qd2345(
    single_tx_amount_vnd: float,
    daily_cumulative_amount_vnd: float,
    is_first_time_new_device: bool,
    has_verified_chip_cccd: bool
) -> dict:
    """
    Evaluate biometric authentication requirements under Decision 2345 & Circular 50.
    """
    cfg = _DATA["biometric_qd2345"]
    single_threshold = float(cfg["single_tx_threshold_vnd"])
    daily_threshold = float(cfg["daily_cumulative_threshold_vnd"])
    requires_biometric = False
    reasons = []

    if is_first_time_new_device is True:
        requires_biometric = True
        reasons.append(TOOL_TEMPLATES["qd2345_reason_new_device"])

    if single_tx_amount_vnd > single_threshold:
        requires_biometric = True
        reasons.append(TOOL_TEMPLATES["qd2345_reason_single_tx"])

    if daily_cumulative_amount_vnd > daily_threshold:
        requires_biometric = True
        reasons.append(TOOL_TEMPLATES["qd2345_reason_daily_cumulative"])

    can_proceed_immediately = True
    action_required = TOOL_TEMPLATES["qd2345_action_normal_otp"]

    if requires_biometric is True:
        if has_verified_chip_cccd is True:
            can_proceed_immediately = True
            action_required = TOOL_TEMPLATES["qd2345_action_facepay_otp"]
        else:
            can_proceed_immediately = False
            action_required = TOOL_TEMPLATES["qd2345_action_blocked_cccd"]

    return {
        "status": "SUCCESS",
        "regulation_reference": cfg["regulation_reference"],
        "single_tx_amount_vnd": round(single_tx_amount_vnd, 0),
        "daily_cumulative_amount_vnd": round(daily_cumulative_amount_vnd, 0),
        "requires_biometric": requires_biometric,
        "has_verified_chip_cccd": has_verified_chip_cccd,
        "can_proceed_immediately": can_proceed_immediately,
        "trigger_reasons": reasons,
        "mandatory_action": action_required
    }


def create_support_ticket(
    customer_name: str,
    issue_type: str,
    description: str,
    priority: str = "NORMAL"
) -> dict:
    """Create customer support ticket and dispatch to banking specialist."""
    cfg = _DATA["support_ticket"]
    ts = int(time.time())
    ticket_id = f"{cfg['ticket_id_prefix']}{ts}"

    clean_priority = priority.strip().upper()
    if clean_priority not in cfg["allowed_priorities"]:
        clean_priority = cfg["default_priority"]

    return {
        "status": "created",
        "ticket_id": ticket_id,
        "customer_name": customer_name.strip(),
        "issue_type": issue_type.strip(),
        "description": description.strip(),
        "priority": clean_priority,
        "created_at": datetime.datetime.now().strftime(cfg["timestamp_format"]),
        "emergency_hotline": cfg["emergency_hotline"],
        "message": TOOL_TEMPLATES["support_ticket_created_message"].format(ticket_id=ticket_id)
    }


def _clean_token_number_str(token: str) -> str:
    strip_chars = _PARSER_RULES["strip_punctuation"]
    cleaned = token.strip()
    while len(cleaned) > 0 and cleaned[-1] in strip_chars:
        cleaned = cleaned[:-1]
    while len(cleaned) > 0 and cleaned[0] in strip_chars:
        cleaned = cleaned[1:]
    return cleaned


def _parse_float_or_int(tok: str) -> float | None:
    t = tok.strip()
    if t == "":
        return None
    if t.endswith("%"):
        t = t[:-1].strip()

    if "," in t and "." not in t:
        comma_parts = t.split(",")
        if len(comma_parts) == 2:
            if len(comma_parts[1]) <= 2:
                if comma_parts[0].isdigit() and comma_parts[1].isdigit():
                    return float(comma_parts[0] + "." + comma_parts[1])
        pure_digits = t.replace(",", "")
        if pure_digits.isdigit():
            return float(pure_digits)

    if "." in t:
        dot_parts = t.split(".")
        if len(dot_parts) > 2:
            pure_digits = t.replace(".", "")
            if pure_digits.isdigit():
                return float(pure_digits)
        elif len(dot_parts) == 2:
            if len(dot_parts[1]) == 3:
                if dot_parts[0].isdigit() and dot_parts[1].isdigit():
                    return float(t.replace(".", ""))
            if dot_parts[0].isdigit() and dot_parts[1].isdigit():
                return float(t)

    if t.isdigit():
        return float(t)

    return None


def parse_vietnamese_financial_params(query: str) -> dict:
    """
    Extract Vietnamese financial parameters (0% Regex Law, Zero Ternary).
    Extracts principal/amount, term, interest percentage, device status, and chip CCCD status.
    """
    cleaned_q = query.lower().replace("/", " ")
    words = cleaned_q.split()

    amount_vnd = 0.0
    single_tx_amount_vnd = 0.0
    daily_cumulative_amount_vnd = 0.0
    term_months = 0
    annual_rate_percent = 0.0
    amounts_found = []

    percent_tokens = _PARSER_RULES["percent_tokens"]
    year_tokens = _PARSER_RULES["year_tokens"]
    month_tokens = _PARSER_RULES["month_tokens"]
    amount_multipliers = _PARSER_RULES["amount_multipliers"]
    min_raw_amount = float(_PARSER_RULES["min_raw_amount_vnd"])
    ctx_window = int(_PARSER_RULES["context_window_words"])

    for i in range(len(words)):
        raw_tok = words[i]
        clean_tok = _clean_token_number_str(raw_tok)

        # 1. Parse rate percentage
        if "%" in raw_tok:
            num = _parse_float_or_int(clean_tok)
            if num is not None:
                annual_rate_percent = num
                continue
        elif i + 1 < len(words):
            next_clean = _clean_token_number_str(words[i + 1])
            if next_clean in percent_tokens:
                num = _parse_float_or_int(clean_tok)
                if num is not None:
                    annual_rate_percent = num
                    continue

        # 2. Parse term (months / years)
        if i + 1 < len(words):
            next_clean = _clean_token_number_str(words[i + 1])
            if next_clean in year_tokens:
                num = _parse_float_or_int(clean_tok)
                if num is not None:
                    term_months = int(num * 12)
                    continue
            elif next_clean in month_tokens:
                num = _parse_float_or_int(clean_tok)
                if num is not None:
                    term_months = int(num)
                    continue

        # 3. Parse monetary amounts
        num = _parse_float_or_int(clean_tok)
        if num is not None:
            multiplier = 1.0
            has_suffix = False
            if i + 1 < len(words):
                next_clean = _clean_token_number_str(words[i + 1])
                for rule in amount_multipliers:
                    if next_clean in rule["tokens"]:
                        multiplier = float(rule["multiplier"])
                        has_suffix = True
                        break

            if has_suffix is True:
                total_val = num * multiplier
                start_context = max(0, i - ctx_window)
                context_prev = " ".join(words[start_context:i])
                amounts_found.append((total_val, context_prev))
            else:
                if num >= min_raw_amount:
                    start_context = max(0, i - ctx_window)
                    context_prev = " ".join(words[start_context:i])
                    amounts_found.append((num, context_prev))

    # Allocate single vs daily cumulative amounts
    daily_kws = _PARSER_RULES["daily_context_keywords"]
    for val, ctx in amounts_found:
        if any(kw in ctx for kw in daily_kws):
            if daily_cumulative_amount_vnd == 0.0:
                daily_cumulative_amount_vnd = val
        else:
            if single_tx_amount_vnd == 0.0:
                single_tx_amount_vnd = val
            elif daily_cumulative_amount_vnd == 0.0:
                daily_cumulative_amount_vnd = val

        if amount_vnd == 0.0:
            amount_vnd = val

    if single_tx_amount_vnd > 0.0 and daily_cumulative_amount_vnd == 0.0:
        daily_cumulative_amount_vnd = single_tx_amount_vnd
    elif daily_cumulative_amount_vnd > 0.0 and single_tx_amount_vnd == 0.0:
        single_tx_amount_vnd = daily_cumulative_amount_vnd

    # Detect new device activation
    is_first_time_new_device = False
    for dkw in _PARSER_RULES["new_device_keywords"]:
        if dkw in cleaned_q:
            is_first_time_new_device = True
            break

    # Detect chip CCCD verification status
    has_verified_chip_cccd = True
    for ncm in _PARSER_RULES["no_cccd_keywords"]:
        if ncm in cleaned_q:
            has_verified_chip_cccd = False
            break

    # Biometric requirement evaluation under Decision 2345
    qd_cfg = _DATA["biometric_qd2345"]
    requires_biometric = False
    if is_first_time_new_device is True:
        requires_biometric = True
    if single_tx_amount_vnd > float(qd_cfg["single_tx_threshold_vnd"]):
        requires_biometric = True
    if daily_cumulative_amount_vnd > float(qd_cfg["daily_cumulative_threshold_vnd"]):
        requires_biometric = True

    return {
        "amount_vnd": amount_vnd,
        "single_tx_amount_vnd": single_tx_amount_vnd,
        "daily_cumulative_amount_vnd": daily_cumulative_amount_vnd,
        "term_months": term_months,
        "annual_rate_percent": annual_rate_percent,
        "is_first_time_new_device": is_first_time_new_device,
        "has_verified_chip_cccd": has_verified_chip_cccd,
        "requires_biometric": requires_biometric
    }


def check_complain_count(complain_count: int = 0) -> dict:
    """Track complaint counts and suggest human agent escalation at threshold."""
    threshold = int(_DATA["complain_escalation"]["escalation_threshold"])
    updated_count = int(complain_count) + 1
    escalate_to_human = False
    if updated_count >= threshold:
        escalate_to_human = True

    if escalate_to_human:
        message = TOOL_TEMPLATES["complain_escalate_message"]
    else:
        message = TOOL_TEMPLATES["complain_standard_message"]

    return {
        "status": "success",
        "previous_count": complain_count,
        "complain_count": updated_count,
        "escalate_to_human": escalate_to_human,
        "message": message
    }


def transfer_to_agent(
    business_process: str = "request_human_agent",
    customer_segment: str = "individual"
) -> dict:
    """Transfer session to Live Agent based on operating hours."""
    cfg = _DATA["transfer_to_agent"]
    now_vn = datetime.datetime.now()
    current_hour = now_vn.hour

    is_working_hour = False
    if current_hour >= int(cfg["working_hour_start"]) and current_hour < int(cfg["working_hour_end"]):
        is_working_hour = True

    queue_name = cfg["individual_queue"]
    if customer_segment.lower() in cfg["corporate_aliases"]:
        queue_name = cfg["corporate_queue"]

    if is_working_hour:
        handoff_status = "queued"
        customer_notice = TOOL_TEMPLATES["transfer_queued_notice"]
    else:
        handoff_status = "offline_hours"
        customer_notice = TOOL_TEMPLATES["transfer_offline_notice"]

    return {
        "status": "success",
        "handoff_status": handoff_status,
        "is_working_hour": is_working_hour,
        "business_process": business_process,
        "customer_segment": customer_segment,
        "target_queue": queue_name,
        "hotline": cfg["hotline"],
        "customer_notice": customer_notice
    }


def rate_agent_csat(
    rating_score: int,
    user_message: str = "",
    channel: str = "web"
) -> dict:
    """Record customer CSAT rating from 1 to 5 stars."""
    cfg = _DATA["csat"]
    min_score = int(cfg["min_score"])
    max_score = int(cfg["max_score"])
    score = int(rating_score)
    if score < min_score:
        score = min_score
    if score > max_score:
        score = max_score

    feedback_map = TOOL_TEMPLATES["csat_feedback_map"]
    default_feedback = TOOL_TEMPLATES["csat_default_feedback"]
    if score in feedback_map:
        response_text = feedback_map[score]
    elif str(score) in feedback_map:
        response_text = feedback_map[str(score)]
    else:
        response_text = default_feedback

    return {
        "status": "success",
        "rating_score": score,
        "comment": user_message,
        "channel": channel,
        "final_answer": response_text
    }


def deflect_competitor_comparison(
    competitor_name: str = "",
    query: str = ""
) -> dict:
    """Deflect competitor comparison queries with bank strength highlights."""
    cfg = _DATA["competitor_deflection"]
    return {
        "status": "success",
        "strategy": cfg["strategy"],
        "directive": COMPETITOR_DEFLECTION_DIRECTIVE,
        "competitor_mentioned": competitor_name,
        "original_query": query
    }
