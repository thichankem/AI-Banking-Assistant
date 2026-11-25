"""Build DS-02: 1,000 Tool Calling, Skill Routing & Deterministic Calculator Cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_banking.evaluator.suite_export import export_suite_bundle
from agent_banking.tools.banking_api_tools import (
    calculate_loan_amortization,
    calculate_savings_profit,
    check_biometric_compliance_qd2345,
    check_complain_count,
    get_foreign_exchange_rates,
    get_interest_rates,
)

OUT_DIR = Path("dataset/A_Bank_eval_tools_skills")


def _fmt_vnd(value: float) -> str:
    return f"{int(round(value)):,}".replace(",", ".")


def _make_ds02_item(
    idx_num: int,
    group: str,
    difficulty: str,
    priority: str,
    question: str,
    ref_answer: str,
    must_include: list[str],
    expected_tools: list[str],
    numeric_targets: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    forbidden_tools: list[str] | None = None,
    must_not_include: list[str] | None = None,
) -> dict[str, Any]:
    if forbidden_tools is not None:
        forb_tools = list(forbidden_tools)
    else:
        forb_tools = []
    if must_not_include is not None:
        forb_words = list(must_not_include)
    else:
        forb_words = []
    return {
        "id": f"DS02-{idx_num:04d}",
        "dataset": "DS-02",
        "group": group,
        "capability": "tools_skills",
        "difficulty": difficulty,
        "priority": priority,
        "question": question,
        "turns": [question],
        "profile": {"persona": "advisor", "tone": "formal", "format_type": "standard", "custom_persona": ""},
        "reference_answer": ref_answer,
        "rubric": {
            "must_include": must_include,
            "must_not_include": forb_words,
        },
        "checks": {
            "expected_tools": expected_tools,
            "forbidden_tools": forb_tools,
            "expected_blocked": None,
            "numeric_targets": numeric_targets,
            "artifact_contract": None,
            "citation_contract": None,
            "style_contract": None,
        },
        "evidence": evidence,
    }


def build_ds02() -> list[dict[str, Any]]:
    """Generate 1,000 deterministic tool and skill evaluation cases."""
    items: list[dict[str, Any]] = []
    next_id = 1

    # G1: Loan Amortization Calculator + Vietnamese Params (200 cases)
    principals = [
        (150_000_000, "150 triệu đồng"),
        (320_000_000, "320 triệu đồng"),
        (500_000_000, "500 triệu đồng"),
        (850_000_000, "850 triệu đồng"),
        (1_200_000_000, "1,2 tỷ đồng"),
        (1_800_000_000, "1 tỷ 800 triệu đồng"),
        (2_500_000_000, "2,5 tỷ đồng"),
        (3_600_000_000, "3,6 tỷ đồng"),
        (5_000_000_000, "5 tỷ đồng"),
        (7_200_000_000, "7,2 tỷ đồng"),
    ]
    rates = [6.5, 7.2, 7.8, 8.5, 9.2, 10.0, 11.4, 12.6]
    terms = [12, 24, 36, 48, 60, 84, 120, 180, 240]

    for k in range(200):
        p_val, p_text = principals[k % len(principals)]
        r_val = rates[(k // len(principals)) % len(rates)]
        t_val = terms[(k * 3 + k // 20) % len(terms)]
        res = calculate_loan_amortization(p_val, r_val, t_val)
        first_pmt = float(res["first_month_payment_vnd"])
        tot_int = float(res["total_interest_vnd"])
        tot_pay = float(res["total_payable_vnd"])

        rate_vi = str(r_val).replace(".", ",")
        if k % 3 == 0:
            q = f"Tôi dự định vay {p_text} trong {t_val} tháng với lãi suất {rate_vi}%/năm theo phương thức dư nợ giảm dần. Tính tiền trả tháng đầu, tổng tiền lãi và tổng phải trả."
        elif k % 3 == 1:
            yrs = t_val // 12
            q = f"Tính lịch trả nợ dư nợ giảm dần cho khoản vay {p_text}, thời hạn {yrs} năm ({t_val} tháng), lãi suất {rate_vi}% một năm: kỳ đầu đóng bao nhiêu và tổng lãi cả kỳ?"
        else:
            q = f"Nếu vay thế chấp {p_text} kỳ hạn {t_val} tháng tại mức lãi {rate_vi}%/năm (gốc chia đều), tháng đầu tiên tôi phải đóng tổng cộng bao nhiêu tiền?"

        ref = (
            f"Theo phương thức dư nợ giảm dần với khoản vay {_fmt_vnd(p_val)} VNĐ trong {t_val} tháng, lãi suất {r_val}%/năm: "
            f"tiền thanh toán tháng đầu là {_fmt_vnd(first_pmt)} VNĐ, tổng tiền lãi cả kỳ là {_fmt_vnd(tot_int)} VNĐ "
            f"và tổng số tiền phải trả là {_fmt_vnd(tot_pay)} VNĐ."
        )
        if k % 2 == 0:
            diff = "medium"
        else:
            diff = "hard"
        if k < 60:
            prio = "P0"
        else:
            prio = "P1"

        items.append(_make_ds02_item(
            next_id,
            "G1_loan_amortization_calc",
            diff,
            prio,
            q,
            ref,
            [_fmt_vnd(first_pmt), _fmt_vnd(tot_int)],
            ["calculate_loan_amortization"],
            [
                {"value": first_pmt, "tolerance_pct": 1.0, "label": "first_month_payment"},
                {"value": tot_int, "tolerance_pct": 1.0, "label": "total_interest"},
            ],
            [{"tool": "calculate_loan_amortization", "principal": p_val, "rate": r_val, "term": t_val}],
        ))
        next_id += 1

    # G2: Savings Profit, Interest Rates Table & FX Rates (200 cases)
    sav_amounts = [
        25_000_000,
        45_000_000,
        75_000_000,
        120_000_000,
        180_000_000,
        250_000_000,
        400_000_000,
        650_000_000,
        900_000_000,
        1_300_000_000,
        2_200_000_000,
        3_800_000_000,
    ]
    sav_terms = [1, 2, 3, 5, 6, 9, 12, 18, 24, 36]
    for k in range(110):
        amt = sav_amounts[k % len(sav_amounts)]
        tm = sav_terms[(k // len(sav_amounts)) % len(sav_terms)]
        sres = calculate_savings_profit(amt, tm)
        rate_used = float(sres["applicable_rate_percent"])
        earned = float(sres["interest_earned_vnd"])
        total_mat = float(sres["total_at_maturity_vnd"])
        amt_m = amt // 1_000_000
        q = (
            f"Nếu gửi tiết kiệm mô phỏng {amt_m} triệu đồng kỳ hạn {tm} tháng theo bảng lãi suất mẫu A Bank, "
            f"lãi suất áp dụng là bao nhiêu %/năm và tổng gốc lãi khi đáo hạn nhận được bao nhiêu?"
        )
        ref = (
            f"Với số tiền gửi mô phỏng {_fmt_vnd(amt)} VNĐ kỳ hạn {tm} tháng, lãi suất mẫu áp dụng là {rate_used}%/năm; "
            f"tiền lãi dự kiến là {_fmt_vnd(earned)} VNĐ, tổng nhận khi đáo hạn là {_fmt_vnd(total_mat)} VNĐ (lưu ý đây là số liệu mô phỏng)."
        )
        if k < 40:
            prio_sav = "P0"
        else:
            prio_sav = "P1"
        items.append(_make_ds02_item(
            next_id,
            "G2_savings_and_interest_fx",
            "medium",
            prio_sav,
            q,
            ref,
            [str(rate_used), _fmt_vnd(earned), "mô phỏng"],
            ["calculate_savings_profit"],
            [
                {"value": rate_used, "tolerance_pct": 0.5, "label": "savings_rate"},
                {"value": earned, "tolerance_pct": 1.0, "label": "interest_earned"},
            ],
            [{"tool": "calculate_savings_profit", "amount": amt, "term": tm}],
        ))
        next_id += 1

    currencies = ["USD", "EUR", "JPY", "GBP", "SGD", "AUD", "CAD", "CNY"]
    fx_volumes = [500, 1200, 3500, 8000, 15000, 25000]
    for k in range(90):
        cur = currencies[k % len(currencies)]
        fx_res = get_foreign_exchange_rates(cur)
        rate_info = fx_res["rates"][cur]
        buy_cash = float(rate_info["buy_cash"])
        buy_trf = float(rate_info["buy_transfer"])
        sell_r = float(rate_info["sell"])

        if k % 3 == 0:
            fx_turn = (k // 3) + 1
            q = f"[Lượt tra cứu tỷ giá #{fx_turn}] Tra cứu tỷ giá {cur} mẫu hôm nay của ngân hàng: giá mua tiền mặt, mua chuyển khoản và bán ra là bao nhiêu?"
            ref = (
                f"Theo bảng tỷ giá mẫu (mô phỏng ngày {fx_res['effective_date']}) đối với {cur} ({rate_info['name']}): "
                f"mua tiền mặt {buy_cash:,.1f} VNĐ, mua chuyển khoản {buy_trf:,.1f} VNĐ, bán ra {sell_r:,.1f} VNĐ."
            )
            targets = [
                {"value": buy_cash, "tolerance_pct": 0.2, "label": "buy_cash"},
                {"value": sell_r, "tolerance_pct": 0.2, "label": "sell"},
            ]
            must = [cur, "mô phỏng"]
            exp_t = ["get_foreign_exchange_rates"]
        elif k % 3 == 1:
            vol = fx_volumes[(k // len(currencies)) % len(fx_volumes)] + ((k // 3) * 100)
            vnd_sell = round(vol * sell_r)
            q = f"Theo tỷ giá bán {cur} mẫu hiện hành, cần khoảng bao nhiêu tiền VNĐ để mua {vol:,} {cur}?"
            ref = f"Với giá bán ra {cur} mô phỏng là {sell_r:,.1f} VNĐ/{cur}, để mua {vol:,} {cur} cần {_fmt_vnd(vnd_sell)} VNĐ (tính toán mô phỏng)."
            targets = [{"value": float(vnd_sell), "tolerance_pct": 0.5, "label": "fx_sell_total"}]
            must = [cur, _fmt_vnd(vnd_sell), "mô phỏng"]
            exp_t = ["get_foreign_exchange_rates"]
        else:
            q = f"[Phiên tra cứu #{k}] Tra cứu bảng lãi suất tiền gửi VND mẫu cho kỳ hạn không kỳ hạn, 3 tháng, 12 tháng và 24 tháng."
            ref = "Theo bảng lãi suất mẫu mô phỏng: Không kỳ hạn 0.1%/năm, 3 tháng 1.9%/năm, 12 tháng 4.6%/năm, 24 tháng 4.7%/năm."
            targets = [
                {"value": 1.9, "tolerance_pct": 0.5, "label": "3m_rate"},
                {"value": 4.6, "tolerance_pct": 0.5, "label": "12m_rate"},
            ]
            must = ["1.9", "4.6", "4.7", "mô phỏng"]
            exp_t = ["get_interest_rates"]

        if k % 2 == 0:
            prio_g2 = "P0"
        else:
            prio_g2 = "P1"
        items.append(_make_ds02_item(
            next_id,
            "G2_savings_and_interest_fx",
            "medium",
            prio_g2,
            q,
            ref,
            must,
            exp_t,
            targets,
            [{"tool": exp_t[0], "currency": cur}],
        ))
        next_id += 1

    # G3: QD2345 Biometric Compliance (150 cases)
    qd_scenarios = [
        (10_000_000, 10_000_000, False, True, "Đúng ngưỡng 10.000.000đ đơn lẻ"),
        (10_000_001, 10_000_001, False, True, "Vượt 1 đồng trên 10.000.000đ đơn lẻ"),
        (9_500_000, 20_000_000, False, True, "Lũy kế ngày đúng ngưỡng 20.000.000đ"),
        (9_500_000, 20_500_000, False, True, "Lũy kế ngày vượt 20.000.000đ"),
        (2_000_000, 5_000_000, True, True, "Đăng nhập thiết bị di động mới"),
        (15_000_000, 15_000_000, False, False, "Chưa xác thực CCCD gắn chip khi cần sinh trắc học"),
        (3_000_000, 25_000_000, False, False, "Lũy kế vượt 20M và chưa có CCCD chip"),
        (50_000_000, 50_000_000, False, True, "Chuyển khoản 50 triệu đơn lẻ đã có CCCD chip"),
        (500_000, 1_500_000, False, True, "Giao dịch nhỏ 500k thiết bị cũ"),
        (8_000_000, 18_000_000, False, True, "Giao dịch 8M tổng ngày 18M"),
    ]
    for k in range(150):
        base = qd_scenarios[k % len(qd_scenarios)]
        step_round = k // len(qd_scenarios)
        if step_round == 0:
            s_amt = base[0]
            c_amt = base[1]
        elif base[0] in (10_000_000, 10_000_001) and k < 30:
            s_amt = base[0]
            c_amt = base[1] + (step_round * 25_000)
        elif base[1] in (20_000_000, 20_500_000) and k < 30:
            s_amt = base[0] - (step_round * 10_000)
            c_amt = base[1]
        else:
            s_amt = base[0] + (step_round * 50_000)
            c_amt = max(s_amt, base[1] + (step_round * 75_000))
        new_dev = base[2]
        has_cccd = base[3]
        qd_res = check_biometric_compliance_qd2345(s_amt, c_amt, new_dev, has_cccd)
        req_bio = bool(qd_res["requires_biometric"])
        can_go = bool(qd_res["can_proceed_immediately"])

        if new_dev:
            dev_desc = "kích hoạt/đăng nhập trên thiết bị di động mới"
        else:
            dev_desc = "trên thiết bị cũ quen dùng"
        if has_cccd:
            cccd_desc = "đã xác thực CCCD gắn chip"
        else:
            cccd_desc = "chưa xác thực CCCD gắn chip"

        q = (
            f"Theo Quyết định 2345/QĐ-NHNN, nếu tôi chuyển khoản trực tuyến {_fmt_vnd(s_amt)} đồng "
            f"(tổng lũy kế trong ngày đạt {_fmt_vnd(c_amt)} đồng), {dev_desc} và {cccd_desc}, "
            f"thì có bắt buộc xác thực khuôn mặt sinh trắc học không và giao dịch có thực hiện ngay được không?"
        )
        if req_bio:
            bio_txt = "BẮT BUỘC quét khuôn mặt sinh trắc học (Facepay)"
        else:
            bio_txt = "KHÔNG bắt buộc quét khuôn mặt sinh trắc học (chỉ cần Smart OTP / SMS OTP)"
        if can_go:
            go_txt = "có thể thực hiện ngay"
        else:
            go_txt = "KHÔNG thể thực hiện ngay, cần cập nhật CCCD gắn chip qua NFC hoặc tại quầy"

        ref = (
            f"Theo Quyết định 2345/QĐ-NHNN: với giao dịch đơn lẻ {_fmt_vnd(s_amt)} VNĐ và lũy kế ngày {_fmt_vnd(c_amt)} VNĐ: "
            f"{bio_txt}; trạng thái thực hiện: {go_txt}."
        )
        must = ["2345"]
        if req_bio:
            must.append("bắt buộc")
        else:
            must.append("OTP")
        if not can_go:
            must.append("CCCD")

        items.append(_make_ds02_item(
            next_id,
            "G3_qd2345_biometric_audit",
            "hard",
            "P0",
            q,
            ref,
            must,
            ["check_biometric_compliance_qd2345"],
            [],
            [{"tool": "check_biometric_compliance_qd2345", "single": s_amt, "cum": c_amt, "new_dev": new_dev, "cccd": has_cccd, "res": qd_res}],
        ))
        next_id += 1

    # G4: Support Tickets, Complain Count, Handoff, CSAT, Competitor Deflection (200 cases)
    competitors = ["Vietcombank", "BIDV", "Techcombank", "VietinBank", "MB Bank", "ACB", "Sacombank", "VPBank", "TPBank", "VIB"]
    issues = ["Lỗi chuyển tiền Napas chưa nhận được tiền", "Thẻ tín dụng bị trừ nhầm phí thường niên", "Không quét được NFC CCCD trên App", "Khóa thẻ khẩn cấp do nghi ngờ lộ thông tin"]
    names = ["Nguyễn Văn An", "Trần Thị Bình", "Lê Hoàng Nam", "Phạm Minh Tuấn", "Đỗ Thu Hương"]
    for k in range(200):
        mode = k % 5
        if mode == 0:
            nm = names[k % len(names)]
            iss = issues[(k // len(names)) % len(issues)]
            q = f"[Hồ sơ #{k}] Tôi tên là {nm}, hãy tạo phiếu hỗ trợ kỹ thuật giúp tôi về sự cố '{iss}' trên ứng dụng A Bank Plus (không cần số thẻ/tài khoản, mức ưu tiên cao)."
            ref = f"Đã tạo yêu cầu/phiếu hỗ trợ cho khách hàng {nm} về vấn đề '{iss}' (tuyệt đối không thu thập số thẻ/mã PIN)."
            must = [nm, "hỗ trợ"]
            exp_t = ["create_support_ticket"]
            forb_t: list[str] = []
        elif mode == 1:
            if k % 2 == 0:
                cnt_val = 0
            else:
                cnt_val = 2 + (k // 10)
            cres = check_complain_count(cnt_val)
            if cres["escalate_to_human"]:
                q = f"[Phản ánh #{k}] Đây là lần thứ {cnt_val + 1} tôi phản ánh mà sự cố chuyển khoản vẫn chưa được xử lý (trước đó đã khiếu nại {cnt_val} lần), hãy kiểm tra mức độ khiếu nại và chuyển giúp tôi."
                ref = "Hệ thống ghi nhận số lần khiếu nại đạt ngưỡng từ 2 lần trở lên và đề xuất kết nối trực tiếp với Tư vấn viên chuyên trách."
                must = ["Tư vấn viên"]
                exp_t = ["check_complain_count"]
            else:
                q = f"Tôi muốn kiểm tra trạng thái đếm số lần khiếu nại khi mới báo lỗi lần đầu tiên (số lần khiếu nại trước đó là {cnt_val}, phiên #{k})."
                ref = f"Số lần khiếu nại được cập nhật thành {cres['complain_count']}, chưa vượt ngưỡng bắt buộc chuyển tuyến."
                must = ["1"]
                exp_t = ["check_complain_count"]
            forb_t = []
        elif mode == 2:
            if k % 2 == 0:
                seg_val = "VIP"
            else:
                seg_val = "Cá nhân"
            q = f"Hãy kết nối tôi (khách hàng phân khúc {seg_val}) gặp trực tiếp giao dịch viên hỗ trợ quy trình 'Khiếu nại giao dịch' ngay (yêu cầu #{k})."
            ref = f"Đã chuyển yêu cầu phân khúc {seg_val} sang hàng đợi Chuyên viên Tư vấn (hotline 1900 6688)."
            must = ["1900 6688", "Chuyên viên"]
            exp_t = ["transfer_to_agent"]
            forb_t = []
        elif mode == 3:
            score = (k % 5) + 1
            q = f"Tôi đánh giá dịch vụ hỗ trợ hôm nay {score} sao trên kênh app vì phản hồi rõ ràng (lượt #{k}). Hãy ghi nhận đánh giá CSAT giúp tôi."
            ref = f"Cảm ơn Quý khách đã đánh giá {score} sao cho dịch vụ của ngân hàng."
            must = [str(score), "sao"]
            exp_t = ["rate_agent_csat"]
            forb_t = []
        else:
            comp = competitors[(k // 5) % len(competitors)]
            aspects = ["lãi suất tiết kiệm", "phí chuyển tiền quốc tế", "thẻ tín dụng hoàn tiền", "điều kiện vay mua nhà"]
            aspect = aspects[(k // 50) % len(aspects)]
            q = f"[Câu hỏi đối chiếu #{k}] So với ngân hàng {comp}, {aspect} của A Bank có vượt trội hay tốt hơn {comp} không?"
            ref = f"Trợ lý không so sánh hoặc suy diễn thông tin của ngân hàng đối thủ {comp}; chỉ cung cấp thông tin chính thức theo tài liệu của A Bank."
            must = ["A Bank"]
            exp_t = ["deflect_competitor_comparison"]
            forb_t = []

        if mode in (0, 1, 4):
            prio_g4 = "P0"
        else:
            prio_g4 = "P1"
        items.append(_make_ds02_item(
            next_id,
            "G4_support_csat_handoff_complain",
            "medium",
            prio_g4,
            q,
            ref,
            must,
            exp_t,
            [],
            [{"mode": mode, "index": k}],
            forbidden_tools=forb_t,
            must_not_include=["số thẻ 4221", "mật khẩu OTP"],
        ))
        next_id += 1

    # G5: Skill Routing, Reference Inspection & Multi-Tool Combo (250 cases)
    skill_scenarios = [
        ("bank-information", "Địa chỉ, hotline tổng đài chăm sóc khách hàng chính thức và mạng lưới của A Bank gồm những kênh nào?", ["search_banking_knowledge"], ["1900"]),
        ("banking-query-understanding", "Phân tích yêu cầu: Tôi muốn vừa xem điều kiện mở thẻ Eco Visa vừa tính thử vay 600 triệu trong 36 tháng lãi 9%/năm.", ["calculate_loan_amortization", "search_banking_knowledge"], ["Eco Visa"]),
        ("business-procedure", "Quy trình từng bước tra soát giao dịch chuyển tiền trong nước bị treo và thời hạn SLA xử lý mất bao lâu?", ["search_banking_knowledge"], ["tra soát"]),
        ("compliance-qd2345-audit", "Vừa kiểm tra quy định Quyết định 2345 khi chuyển 18 triệu đồng trên thiết bị mới vừa nêu căn cứ pháp lý liên quan.", ["check_biometric_compliance_qd2345"], ["2345"]),
        ("fact-checking-audit", "Kiểm chứng lại thông tin: Có đúng mọi giao dịch tiết kiệm kỳ hạn 12 tháng rút trước hạn đều hưởng lãi 4,6% không?", ["search_banking_knowledge"], ["không kỳ hạn"]),
        ("financial-calculator", "Tôi muốn gửi 400 triệu tiết kiệm 12 tháng theo lãi suất mô phỏng, đồng thời quy đổi 3.000 USD theo tỷ giá bán mô phỏng.", ["calculate_savings_profit", "get_foreign_exchange_rates"], ["mô phỏng"]),
        ("incident-support-ticket", "Ứng dụng báo lỗi khi chuyển tiền 2 lần liên tiếp và tài khoản đã bị trừ tiền nhưng người nhận chưa có. Tôi cần lập phiếu hỗ trợ (Nguyễn Hải, ưu tiên cao).", ["create_support_ticket"], ["hỗ trợ"]),
        ("product-advice", "Tư vấn điều kiện tham gia và quyền lợi chính của sản phẩm bảo hiểm nhân thọ A-Life Trọn Đời.", ["search_banking_knowledge"], ["A-Life"]),
        ("rag-knowledge-retrieval", "Tìm chính xác quy định mức phí phát hành và thường niên của thẻ tín dụng Cashback Visa.", ["search_banking_knowledge"], ["Cashback Visa"]),
        ("service-troubleshooting", "Khách hàng quên mật khẩu đăng nhập A-Bank Plus và bị khóa sau nhiều lần nhập sai thì xử lý thế nào trên app?", ["search_banking_knowledge"], ["A-Bank Plus"]),
        ("interactive-simulator", "Hãy mở công cụ mô phỏng tính lãi tiết kiệm cho số tiền 900 triệu đồng kỳ hạn 12 tháng để tôi kéo điều chỉnh trực tiếp.", ["present_visualization"], ["900"]),
        ("visualization", "Vẽ biểu đồ cột so sánh lãi suất tiết kiệm mẫu các kỳ hạn 1, 3, 6, 12, 24 tháng giúp tôi.", ["present_visualization"], ["1.6"]),
    ]

    for k in range(250):
        sk_name, base_q, exp_tools, must_toks = skill_scenarios[k % len(skill_scenarios)]
        variant = (k // len(skill_scenarios)) + 1
        if k % 2 == 0:
            q = f"[Kịch bản Skill {sk_name} #{variant}] {base_q}"
        else:
            q = f"{base_q} (Yêu cầu #{variant}: đối chiếu đúng kỹ năng '{sk_name}' và trích nguồn rõ ràng)."

        ref = f"Kích hoạt kỹ năng '{sk_name}', sử dụng đúng công cụ {', '.join(exp_tools)} và phản hồi đầy đủ căn cứ ({', '.join(must_toks)})."
        if k < 100:
            prio_g5 = "P0"
        else:
            prio_g5 = "P1"
        items.append(_make_ds02_item(
            next_id,
            "G5_skill_routing_and_combo",
            "hard",
            prio_g5,
            q,
            ref,
            must_toks,
            exp_tools,
            [],
            [{"skill": sk_name, "variant": variant}],
        ))
        next_id += 1

    return items


def main() -> None:
    items = build_ds02()
    cov = export_suite_bundle(items, OUT_DIR, "eval_tools_skills", "DS-02", canary_size=35)
    print(json.dumps(cov, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
