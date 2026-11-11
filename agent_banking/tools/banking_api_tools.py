# -*- coding: utf-8 -*-
"""
Banking API Tools (Digital Banking Business Toolset v4.0)
======================================================================
Provides execution tools for AI Agents:
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

from __future__ import annotations
import time
import datetime


def get_interest_rates(deposit_type: str = "vnd") -> dict:
    """Look up listed deposit interest rates."""
    dep_type = deposit_type.strip().lower()
    if dep_type == "vnd" or dep_type == "dong":
        rates = [
            {"term": "Không kỳ hạn", "rate_percent_per_year": 0.1, "note": "Rút linh hoạt mọi lúc"},
            {"term": "1 tháng", "rate_percent_per_year": 1.6, "note": "Lãi cuối kỳ"},
            {"term": "3 tháng", "rate_percent_per_year": 1.9, "note": "Lãi cuối kỳ"},
            {"term": "6 tháng", "rate_percent_per_year": 2.9, "note": "Lãi cuối kỳ"},
            {"term": "9 tháng", "rate_percent_per_year": 2.9, "note": "Lãi cuối kỳ"},
            {"term": "12 tháng (1 năm)", "rate_percent_per_year": 4.6, "note": "Kỳ hạn phổ biến nhất"},
            {"term": "24 tháng (2 năm)", "rate_percent_per_year": 4.7, "note": "Lãi suất cao nhất cho khách hàng cá nhân"},
            {"term": "36 - 60 tháng", "rate_percent_per_year": 4.7, "note": "Tiền gửi dài hạn"}
        ]
        return {
            "status": "success",
            "currency": "VND",
            "effective_date": "2026-10-25",
            "bank": "Ngân hàng Thương mại",
            "interest_rates": rates,
            "disclaimer": "Lãi suất có thể thay đổi theo từng thời kỳ và áp dụng ưu đãi khi gửi online."
        }
    else:
        return {
            "status": "success",
            "currency": "USD",
            "effective_date": "2026-10-25",
            "bank": "Ngân hàng Thương mại",
            "interest_rates": [
                {"term": "Mọi kỳ hạn", "rate_percent_per_year": 0.0, "note": "Áp dụng theo quy định của NHNN đối với ngoại tệ USD"}
            ],
            "disclaimer": "Lãi suất tiền gửi USD hiện là 0%/năm theo quy định kiểm soát ngoại hối của Ngân hàng Nhà nước."
        }


def get_foreign_exchange_rates(currency: str = "USD") -> dict:
    """Look up current foreign exchange rates."""
    curr = currency.strip().upper()

    rate_table = {
        "USD": {"name": "Đô la Mỹ", "buy_cash": 25120.0, "buy_transfer": 25150.0, "sell": 25480.0},
        "EUR": {"name": "Đồng Euro", "buy_cash": 27150.0, "buy_transfer": 27260.0, "sell": 28620.0},
        "JPY": {"name": "Yên Nhật", "buy_cash": 162.8, "buy_transfer": 164.4, "sell": 172.3},
        "GBP": {"name": "Bảng Anh", "buy_cash": 32350.0, "buy_transfer": 32670.0, "sell": 33730.0},
        "SGD": {"name": "Đô la Singapore", "buy_cash": 18890.0, "buy_transfer": 19080.0, "sell": 19690.0},
        "AUD": {"name": "Đô la Úc", "buy_cash": 16540.0, "buy_transfer": 16700.0, "sell": 17240.0},
        "CAD": {"name": "Đô la Canada", "buy_cash": 18120.0, "buy_transfer": 18300.0, "sell": 18890.0},
        "CNY": {"name": "Nhân dân tệ", "buy_cash": 3480.0, "buy_transfer": 3510.0, "sell": 3630.0}
    }

    if curr in rate_table:
        selected_rate = {curr: rate_table[curr]}
    else:
        selected_rate = rate_table

    return {
        "status": "success",
        "effective_date": "2026-10-25",
        "bank": "Ngân hàng Thương mại",
        "rates": selected_rate,
        "note": "Tỷ giá mua chuyển khoản và bán được cập nhật thời gian thực từ hệ thống."
    }


def calculate_savings_profit(amount_vnd: float, term_months: int) -> dict:
    """
    Calculate estimated savings profit and total maturity balance.
    Formula: Interest = Deposit amount * Rate (%/year) * (Months / 12)
    """
    if amount_vnd <= 0:
        return {
            "status": "ERROR",
            "message": "Số tiền gửi phải lớn hơn 0 VND."
        }

    if term_months < 1:
        rate = 0.1
    elif term_months < 3:
        rate = 1.6
    elif term_months < 6:
        rate = 1.9
    elif term_months < 12:
        rate = 2.9
    elif term_months < 24:
        rate = 4.6
    else:
        rate = 4.7

    interest_earned = amount_vnd * (rate / 100.0) * (term_months / 12.0)
    total_received = amount_vnd + interest_earned

    return {
        "status": "SUCCESS",
        "principal_amount_vnd": round(amount_vnd, 0),
        "term_months": term_months,
        "applicable_rate_percent": rate,
        "interest_earned_vnd": round(interest_earned, 0),
        "total_at_maturity_vnd": round(total_received, 0),
        "formatted_summary": (
            f"Với số tiền gửi {amount_vnd:,.0f} VND kỳ hạn {term_months} tháng "
            f"(lãi suất {rate}%/năm), tiền lãi dự kiến nhận được là {interest_earned:,.0f} VND. "
            f"Tổng số tiền nhận được khi đáo hạn là {total_received:,.0f} VND."
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
            "message": "Số tiền vay gốc phải lớn hơn 0 VNĐ."
        }

    if term_months <= 0:
        return {
            "status": "ERROR",
            "message": "Kỳ hạn vay (tháng) phải lớn hơn 0."
        }

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
        if month_idx <= 3:
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
        "calculation_method": "Dư nợ giảm dần (Declining Balance)",
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
    - Single transaction > 10,000,000 VND
    - Daily cumulative amount > 20,000,000 VND
    - First activation or login on new mobile device.
    """
    requires_biometric = False
    reasons = []

    if is_first_time_new_device is True:
        requires_biometric = True
        reasons.append("Kích hoạt ứng dụng lần đầu hoặc đăng nhập trên thiết bị di động mới.")

    if single_tx_amount_vnd > 10000000.0:
        requires_biometric = True
        reasons.append("Giá trị giao dịch chuyển tiền đơn lẻ vượt ngưỡng 10.000.000 VNĐ/lần (Quyết định 2345/QĐ-NHNN).")

    if daily_cumulative_amount_vnd > 20000000.0:
        requires_biometric = True
        reasons.append("Tổng giá trị giao dịch chuyển tiền cộng dồn trong ngày vượt ngưỡng 20.000.000 VNĐ/ngày (Quyết định 2345/QĐ-NHNN).")

    can_proceed_immediately = True
    action_required = "Giao dịch có thể thực hiện ngay bằng mã xác thực Smart OTP / SMS OTP thông thường."

    if requires_biometric is True:
        if has_verified_chip_cccd is True:
            can_proceed_immediately = True
            action_required = "BẮT BUỘC quét khuôn mặt sinh trắc học (Facepay) kết hợp Smart OTP để hoàn tất giao dịch."
        else:
            can_proceed_immediately = False
            action_required = "CHẶN GIAO DỊCH TẠM THỜI: Khách hàng bắt buộc phải cập nhật Căn cước công dân gắn chip trước khi thực hiện giao dịch này."

    return {
        "status": "SUCCESS",
        "regulation_reference": "Quyết định 2345/QĐ-NHNN & Thông tư 50/2024/TT-NHNN",
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
    ts = int(time.time())
    ticket_id = f"TK-{ts}"

    clean_priority = priority.strip().upper()
    if clean_priority not in ["URGENT", "HIGH", "NORMAL", "LOW"]:
        clean_priority = "NORMAL"

    return {
        "status": "created",
        "ticket_id": ticket_id,
        "customer_name": customer_name.strip(),
        "issue_type": issue_type.strip(),
        "description": description.strip(),
        "priority": clean_priority,
        "emergency_hotline": "1900 54 54 13 (Phục vụ 24/7)",
        "message": (
            f"Hệ thống đã tạo thành công phiếu yêu cầu hỗ trợ mã [{ticket_id}]. "
            "Chuyên viên chăm sóc khách hàng sẽ liên hệ xác minh trong vòng 15-30 phút. "
            "Trong trường hợp khẩn cấp (mất thẻ, nghi ngờ lộ thông tin), vui lòng gọi ngay Tổng đài để khóa dịch vụ tức thì."
        )
    }


def _clean_token_number_str(token: str) -> str:
    cleaned = token.strip()
    while len(cleaned) > 0 and cleaned[-1] in "?!:,;()[]{}*":
        cleaned = cleaned[:-1]
    while len(cleaned) > 0 and cleaned[0] in "?!:,;()[]{}*":
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
            if next_clean in ["%", "phần trăm", "phan tram"]:
                num = _parse_float_or_int(clean_tok)
                if num is not None:
                    annual_rate_percent = num
                    continue

        # 2. Parse term (months / years)
        if i + 1 < len(words):
            next_clean = _clean_token_number_str(words[i + 1])
            if next_clean in ["năm", "nam"]:
                num = _parse_float_or_int(clean_tok)
                if num is not None:
                    term_months = int(num * 12)
                    continue
            elif next_clean in ["tháng", "thang"]:
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
                if next_clean in ["tỷ", "ty", "tỉ"]:
                    multiplier = 1000000000.0
                    has_suffix = True
                elif next_clean in ["triệu", "trieu", "tr"]:
                    multiplier = 1000000.0
                    has_suffix = True
                elif next_clean in ["nghìn", "ngàn", "k"]:
                    multiplier = 1000.0
                    has_suffix = True

            if has_suffix is True:
                total_val = num * multiplier
                start_context = max(0, i - 4)
                context_prev = " ".join(words[start_context:i])
                amounts_found.append((total_val, context_prev))
            else:
                if num >= 10000.0:
                    start_context = max(0, i - 4)
                    context_prev = " ".join(words[start_context:i])
                    amounts_found.append((num, context_prev))

    # Allocate single vs daily cumulative amounts
    for val, ctx in amounts_found:
        if "tổng" in ctx or "cộng dồn" in ctx or "trong ngày" in ctx or "đã chuyển" in ctx:
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
    device_kws = ["thiết bị mới", "thiet bi moi", "lần đầu", "lan dau", "đổi máy", "doi may", "đăng nhập lần đầu"]
    for dkw in device_kws:
        if dkw in cleaned_q:
            is_first_time_new_device = True
            break

    # Detect chip CCCD verification status
    has_verified_chip_cccd = True
    no_cccd_markers = ["chưa có cccd", "chưa gắn chip", "chưa xác thực", "chưa cập nhật cccd", "chưa đối soát", "chưa cập nhật"]
    for ncm in no_cccd_markers:
        if ncm in cleaned_q:
            has_verified_chip_cccd = False
            break

    # Biometric requirement evaluation under Decision 2345
    requires_biometric = False
    if is_first_time_new_device is True:
        requires_biometric = True
    if single_tx_amount_vnd > 10000000.0:
        requires_biometric = True
    if daily_cumulative_amount_vnd > 20000000.0:
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
    """Track complaint counts and suggest human agent escalation if >= 2."""
    updated_count = int(complain_count) + 1
    escalate_to_human = False
    if updated_count >= 2:
        escalate_to_human = True

    if escalate_to_human:
        message = (
            "Hệ thống ghi nhận Quý khách đang gặp vấn đề chưa hài lòng. "
            "Để hỗ trợ Quý khách giải quyết nhanh chóng và chu đáo nhất, "
            "hệ thống đề xuất kết nối trực tiếp với Tư vấn viên chuyên trách."
        )
    else:
        message = "Hệ thống rất tiếc vì sự bất tiện của Quý khách và đang nỗ lực hỗ trợ tốt nhất."

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
    now_vn = datetime.datetime.now()
    current_hour = now_vn.hour

    is_working_hour = False
    if current_hour >= 7 and current_hour < 23:
        is_working_hour = True

    queue_name = "LIVE_AGENT_INDIVIDUAL_QUEUE"
    if customer_segment.lower() in ["corporate", "doanh_nghiep", "doanh nghiệp"]:
        queue_name = "LIVE_AGENT_CORPORATE_QUEUE"

    if is_working_hour:
        handoff_status = "queued"
        customer_notice = (
            "Dạ, yêu cầu của Quý khách đã được chuyển tới hàng đợi Chuyên viên Tư vấn. "
            "Chuyên viên sẽ tiếp nhận và phản hồi Quý khách trong ít phút. "
            "Quý khách vui lòng giữ kết nối màn hình trò chuyện."
        )
    else:
        handoff_status = "offline_hours"
        customer_notice = (
            "Hiện tại đang ngoài khung giờ hỗ trợ trực tuyến của Chuyên viên (7h00 - 23h00 hàng ngày). "
            "Để được xử lý khẩn cấp (như khóa thẻ, tra soát giao dịch), Quý khách vui lòng liên hệ ngay "
            "Tổng đài Trung tâm Hỗ trợ Khách hàng 24/7 qua số hotline: 1900 54 54 13."
        )

    return {
        "status": "success",
        "handoff_status": handoff_status,
        "is_working_hour": is_working_hour,
        "business_process": business_process,
        "customer_segment": customer_segment,
        "target_queue": queue_name,
        "hotline": "1900 54 54 13",
        "customer_notice": customer_notice
    }


def rate_agent_csat(
    rating_score: int,
    user_message: str = "",
    channel: str = "web"
) -> dict:
    """Record customer CSAT rating from 1 to 5 stars."""
    score = int(rating_score)
    if score < 1:
        score = 1
    if score > 5:
        score = 5

    feedback_map = {
        1: "Ngân hàng chân thành xin lỗi vì trải nghiệm chưa được như mong đợi của Quý khách. Chúng tôi xin ghi nhận ý kiến để cải thiện chất lượng phục vụ ngay lập tức.",
        2: "Ngân hàng rất tiếc khi trải nghiệm hôm nay của Quý khách chưa thực sự hài lòng. Đội ngũ phát triển sẽ rà soát và nâng cao năng lực hỗ trợ.",
        3: "Ngân hàng cảm ơn phản hồi của Quý khách. Chúng tôi sẽ không ngừng nỗ lực để mang lại trải nghiệm tốt hơn trong các lần giao dịch tới.",
        4: "Ngân hàng rất vui khi đã hỗ trợ được Quý khách. Chúc Quý khách một ngày làm việc hiệu quả và nhiều niềm vui!",
        5: "Ngân hàng xin chân thành cảm ơn Quý khách đã tin tưởng và đánh giá 5 sao. Rất vinh hạnh được đồng hành cùng Quý khách!"
    }

    response_text = feedback_map.get(score, "Cảm ơn Quý khách đã đóng góp ý kiến cho Ngân hàng.")

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
    return {
        "status": "success",
        "strategy": "deflection",
        "directive": (
            "Tuyệt đối không so sánh tiêu cực với ngân hàng đối thủ. "
            "Tập trung làm rõ các chính sách vượt trội của Ngân hàng: "
            "Mạng lưới thanh toán rộng khắp, miễn phí chuyển tiền ngân hàng số, "
            "chuẩn mực an toàn thông tin quốc tế PCI-DSS Level 1 cao nhất."
        ),
        "competitor_mentioned": competitor_name,
        "original_query": query
    }
