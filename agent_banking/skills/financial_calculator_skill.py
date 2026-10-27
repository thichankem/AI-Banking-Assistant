"""
Kỹ năng Tính toán Tài chính & Lịch Trả Nợ Vay (Financial Calculator Skill).
Thiết kế tối giản, trực quan, dễ hiểu theo chuẩn Chunking:
- Tính lịch khấu hao dư nợ giảm dần (Declining Balance Amortization).
- Tính tiền lãi gửi tiết kiệm theo kỳ hạn.
- Tra cứu bảng lãi suất niêm yết và tỷ giá ngoại tệ Vietcombank.

NGUYÊN TẮC BẮT BUỘC:
1. KHÔNG DÙNG __INIT__.PY: Import trực tiếp file module.
2. KHÔNG DÙNG THƯ VIỆN TYPING: Thuần túy dùng kiểu gốc của Python.
3. ZERO TERNARY: Bắt buộc dùng if/else tường minh, không dùng ?: hoặc toán tử 3 ngôi.
4. 0% REGEX LAW: Tuyệt đối không dùng regex.
5. HẠN CHẾ TỐI ĐA THỪA THÃI: Code ngắn gọn, trực diện, dễ hiểu.
"""

from __future__ import annotations

from .base_skill import BaseSkill, SkillMetadata
from agent_banking.tools.banking_api_tools import (
    calculate_loan_amortization,
    calculate_savings_profit,
    get_interest_rates,
    get_foreign_exchange_rates,
    parse_vietnamese_financial_params
)


class FinancialCalculatorSkill(BaseSkill):
    """Kỹ năng tính toán tài chính và tra cứu tỷ giá, lãi suất."""

    def __init__(self, config: object | None = None):
        meta = SkillMetadata(
            skill_id="financial_calculator",
            display_name="Tính toán Tài chính & Lịch Trả Nợ Vay",
            description="Tính khấu hao dư nợ vay giảm dần từng tháng, tính lãi tiết kiệm, tra cứu bảng lãi suất và tỷ giá ngoại tệ.",
            capabilities=[
                "declining_balance_loan_amortization",
                "savings_profit_calculation",
                "interest_rates_lookup",
                "foreign_exchange_rates_lookup"
            ],
            intent_triggers=[
                "lãi suất", "tỷ giá", "tính lãi", "gửi tiết kiệm",
                "vay vốn", "trả nợ", "dư nợ giảm dần", "lịch trả nợ",
                "mỗi tháng trả bao nhiêu", "usd", "ngoại tệ"
            ],
            category="CALCULATION"
        )
        super().__init__(metadata=meta, config=config)

    def _run(self, params: dict) -> dict:
        raw_text = params.get("raw_text", "")
        extracted = {}
        if raw_text:
            extracted = parse_vietnamese_financial_params(raw_text)

        operation = params.get("operation", "auto")

        # Tự động nhận diện thao tác nếu là auto
        if operation == "auto":
            text_lower = raw_text.lower()
            if "vay" in text_lower or "trả nợ" in text_lower or "khấu hao" in text_lower:
                operation = "loan_amortization"
            elif "tiết kiệm" in text_lower or "gửi" in text_lower:
                operation = "savings_profit"
            elif "tỷ giá" in text_lower or "ngoại tệ" in text_lower or "usd" in text_lower:
                operation = "fx_rates"
            else:
                operation = "interest_rates"

        # 1. Tính khấu hao dư nợ giảm dần
        if operation == "loan_amortization":
            principal = params.get("principal", extracted.get("principal", 500000000.0))
            annual_rate = params.get("annual_rate_percent", extracted.get("annual_rate_percent", 7.5))
            term_months = params.get("term_months", extracted.get("term_months", 12))

            result = calculate_loan_amortization(
                principal_vnd=float(principal),
                annual_rate_percent=float(annual_rate),
                term_months=int(term_months)
            )
            return {"operation": "loan_amortization", "calculation_data": result}

        # 2. Tính lãi tiết kiệm
        if operation == "savings_profit":
            principal = params.get("principal", extracted.get("principal", 100000000.0))
            term_months = params.get("term_months", extracted.get("term_months", 6))

            result = calculate_savings_profit(
                amount_vnd=float(principal),
                term_months=int(term_months)
            )
            return {"operation": "savings_profit", "calculation_data": result}

        # 3. Tra cứu tỷ giá ngoại tệ
        if operation == "fx_rates":
            currency = params.get("currency", "USD")
            result = get_foreign_exchange_rates(currency=currency)
            return {"operation": "fx_rates", "rates_data": result}

        # 4. Tra cứu bảng lãi suất niêm yết
        dep_type = params.get("deposit_type", "vnd")
        result = get_interest_rates(deposit_type=dep_type)
        return {"operation": "interest_rates", "rates_data": result}
