"""Build DS-03: 1,000 Multi-Turn Statefulness, Memory Compaction & HITL Scenarios."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_banking.evaluator.suite_export import export_suite_bundle

SCRATCH_DIR = Path(
    r"C:\Users\ADMIN\.gemini\antigravity\brain\37df9fad-e9a2-46be-ab83-7d59cdac4108\scratch"
)
OUT_DIR = Path("dataset/A_Bank_eval_multiturn_hitl")


def _make_ds03_item(
    idx_num: int,
    group: str,
    difficulty: str,
    priority: str,
    turns: list[str],
    ref_answer: str,
    must_include: list[str],
    evidence: list[dict[str, Any]],
    hitl_replies: dict[str, Any] | None = None,
    expected_tools: list[str] | None = None,
    numeric_targets: list[dict[str, Any]] | None = None,
    max_words: int | None = None,
) -> dict[str, Any]:
    if expected_tools is not None:
        tools = list(expected_tools)
    else:
        tools = []
    if numeric_targets is not None:
        nums = list(numeric_targets)
    else:
        nums = []
    if max_words is not None:
        style_c: dict[str, Any] | None = {
            "persona": "advisor",
            "tone": "formal",
            "format_type": "standard",
            "max_words": max_words,
            "require_table": False,
            "require_bullets": False,
            "require_disclaimer": False,
        }
    else:
        style_c = None

    item: dict[str, Any] = {
        "id": f"DS03-{idx_num:04d}",
        "dataset": "DS-03",
        "group": group,
        "capability": "multiturn_hitl",
        "difficulty": difficulty,
        "priority": priority,
        "question": turns[0],
        "turns": turns,
        "profile": {"persona": "advisor", "tone": "formal", "format_type": "standard", "custom_persona": ""},
        "reference_answer": ref_answer,
        "rubric": {"must_include": must_include, "must_not_include": []},
        "checks": {
            "expected_tools": tools,
            "forbidden_tools": [],
            "expected_blocked": None,
            "numeric_targets": nums,
            "artifact_contract": None,
            "citation_contract": None,
            "style_contract": style_c,
        },
        "evidence": evidence,
    }
    if hitl_replies is not None:
        item["hitl_replies"] = hitl_replies
    return item


def build_ds03() -> list[dict[str, Any]]:
    """Generate 1,000 multi-turn coreference, HITL, memory, and policy scope cases."""
    cross_facts = json.loads((SCRATCH_DIR / "cross_docs_facts.json").read_text(encoding="utf-8"))
    nv_docs = [d["source"] for d in cross_facts.get("1. Nghiệp vụ ngân hàng", [])]
    card_docs = [d["source"] for d in cross_facts.get("2. Chương trình thẻ", [])]
    bh_docs = [d["source"] for d in cross_facts.get("5. Sản phẩm bảo hiểm và tín dụng", [])]

    items: list[dict[str, Any]] = []
    next_id = 1

    # G1: Coreference & Self-Correction (250 scenarios)
    coref_topics = [
        ("thẻ tín dụng Eco Visa", "THE 14", "Vậy thẻ đó có chính sách miễn phí thường niên hoặc ưu đãi hoàn tiền gì không?", "chính sách"),
        ("thẻ tín dụng Cashback Visa", "THE 03", "Với thẻ này, hạn mức rút tiền mặt và biểu phí áp dụng thế nào?", "Cashback Visa"),
        ("thẻ tín dụng sinh viên Edu Visa", "THE 13", "Điều kiện thu nhập hoặc chứng minh đối tượng cho loại thẻ trên là gì?", "Edu Visa"),
        ("thẻ tín dụng doanh nghiệp Corporate Visa", "THE 07", "Doanh nghiệp mở thẻ vừa nêu cần cung cấp hồ sơ pháp lý nào?", "Corporate Visa"),
        ("sản phẩm bảo hiểm A-Life Trọn Đời", "BH 01", "Sản phẩm bảo hiểm đó có quy định thời gian cân nhắc và quyền lợi tử vong ra sao?", "A-Life"),
        ("gói vay mua nhà và sửa chữa bất động sản", "NV 06", "Thủ tục thẩm định tài sản bảo đảm cho khoản vay ấy gồm những bước nào?", "mua nhà"),
        ("gói vay mua ô tô tiêu dùng", "NV 07", "Tỷ lệ cho vay tối đa trên giá trị xe của gói vay trên là bao nhiêu?", "ô tô"),
        ("quy trình tra soát giao dịch chuyển tiền", "NV 15", "Thời hạn tối đa tiếp nhận yêu cầu tra soát theo quy trình đó là bao lâu?", "tra soát"),
    ]
    sav_principals = [200, 450, 700, 950, 1500]
    for k in range(250):
        if k % 2 == 0:
            prod, code, t2, tok = coref_topics[k % len(coref_topics)]
            var = (k // len(coref_topics)) + 1
            t1 = f"[Lượt ngữ cảnh #{var}] Hãy giới thiệu khái quát về {prod} của ngân hàng A Bank."
            t3 = f"Nếu khách hàng hỏi tóm tắt điểm quan trọng nhất về {prod} trong câu trả lời vừa rồi thì gồm ý nào?"
            turns = [t1, t2, t3]
            ref = f"Hệ thống duy trì đúng đối tượng '{prod}' ({code}) qua các lượt đồng tham chiếu và trả lời đúng theo tài liệu."
            must = [tok]
            exp_t = ["search_banking_knowledge"]
            nums = []
            diff_g1 = "medium"
            ev_g1 = [{"doc_hint": code}]
        else:
            p1 = sav_principals[k % len(sav_principals)]
            p2 = p1 + 100
            turns = [
                f"[Phiên tự sửa #{k}] Tính giúp tôi lãi gửi tiết kiệm mô phỏng {p1} triệu đồng trong 12 tháng.",
                f"Khoan đã, tôi nhầm! Đổi sang số tiền {p2} triệu đồng và kỳ hạn chỉ 6 tháng (lãi suất mẫu 2,9%/năm) thôi nhé.",
            ]
            expected_int = (p2 * 1_000_000) * 0.029 * (6.0 / 12.0)
            int_str = f"{int(round(expected_int)):,}".replace(",", ".")
            ref = f"Cập nhật theo yêu cầu sửa của khách hàng: số tiền {p2} triệu đồng kỳ hạn 6 tháng (lãi suất 2,9%/năm), tiền lãi mô phỏng là {int_str} VNĐ."
            must = [int_str, "6"]
            exp_t = ["calculate_savings_profit"]
            nums = [{"value": float(round(expected_int)), "tolerance_pct": 1.0, "label": "corrected_savings_interest"}]
            diff_g1 = "hard"
            ev_g1 = [{"doc_hint": "savings_self_correct", "p1": p1, "p2": p2}]

        if k < 80:
            prio_g1 = "P0"
        else:
            prio_g1 = "P1"
        items.append(_make_ds03_item(
            next_id,
            "G1_coreference_and_self_correction",
            diff_g1,
            prio_g1,
            turns,
            ref,
            must,
            ev_g1,
            expected_tools=exp_t,
            numeric_targets=nums,
        ))
        next_id += 1

    # G2: HITL Clarify & Multi-Select (200 scenarios)
    clarify_prompts = [
        ("Tôi muốn hỏi phí chuyển tiền của A Bank là bao nhiêu?", "Chọn kênh/loại giao dịch cụ thể (Chuyển tiền VND hay Quốc tế SWIFT)", "phí"),
        ("Tính giúp tôi tiền lãi khi gửi 500 triệu tại ngân hàng.", "Chọn kỳ hạn mong muốn (6 tháng hoặc 12 tháng)", "500"),
        ("Tôi muốn mở thẻ tín dụng A Bank, cần điều kiện gì?", "Chọn hạng thẻ quan tâm (Cashback, Edu, Eco hay Corporate Visa)", "thẻ tín dụng"),
        ("Quy trình xử lý khi chuyển nhầm hoặc giao dịch lỗi trên app?", "Chọn tình huống cụ thể (tra soát nội địa hay khóa tài khoản)", "tra soát"),
        ("Tôi muốn tìm hiểu gói bảo hiểm của A-Life.", "Chọn sản phẩm cụ thể (A-Life Trọn Đời, Tích Lũy hay Tương Lai)", "A-Life"),
    ]
    for k in range(200):
        base_q, goal_note, tok = clarify_prompts[k % len(clarify_prompts)]
        var = (k // len(clarify_prompts)) + 1
        if k % 2 == 0:
            reply_spec = {"clarify": {"choice_id": "option_1"}, "default": {"choice_id": "option_1"}}
            mode_tag = "single_select"
        else:
            reply_spec = {"clarify": {"choice_ids": ["option_1", "option_2"]}, "default": {"choice_ids": ["option_1", "option_2"]}}
            mode_tag = "multi_select"
        if k < 60:
            prio_g2 = "P0"
        else:
            prio_g2 = "P1"
        q = f"[HITL #{var}-{mode_tag}] {base_q}"
        items.append(_make_ds03_item(
            next_id,
            "G2_hitl_clarify_and_multiselect",
            "medium",
            prio_g2,
            [q],
            f"Nếu câu hỏi chưa đủ tham số, hệ thống hỏi làm rõ ({goal_note}) hoặc trả lời đúng phương án khách hàng chọn sau khi resume.",
            [tok],
            [{"hitl_mode": mode_tag, "variant": var}],
            hitl_replies=reply_spec,
        ))
        next_id += 1

    # G3: HITL Confirm Approve / Reject / Cancel (150 scenarios)
    confirm_cases = [
        ("Hãy lập phiếu hỗ trợ gửi bộ phận kỹ thuật xử lý lỗi không nhận được OTP cho khách hàng Lê Minh Tâm.", "Lê Minh Tâm", "approve"),
        ("Hãy lập phiếu hỗ trợ khẩn cấp khóa kênh Digital vì nghi ngờ lộ thông tin (khách hàng Trần Lan Anh).", "Trần Lan Anh", "cancel"),
        ("Kết nối ngay giúp tôi tới tổng đài viên hàng đợi VIP để khiếu nại giao dịch.", "1900 6688", "approve"),
        ("Hãy gửi phiếu yêu cầu tra soát và khiếu nại cho khách hàng Phạm Quốc Bảo nhưng chờ tôi xác nhận trước.", "Phạm Quốc Bảo", "cancel"),
    ]
    for k in range(150):
        q_tmpl, tok, action = confirm_cases[k % len(confirm_cases)]
        var = (k // len(confirm_cases)) + 1
        q = f"[Confirm-{action} #{var}] {q_tmpl}"
        if action == "approve":
            replies = {"confirm": {"choice_id": "approve"}, "default": {"choice_id": "approve"}}
            ref = f"Khi khách chọn approve xác nhận, hệ thống hoàn tất thao tác và trả kết quả rõ ràng ({tok})."
            must = [tok]
        else:
            replies = {"confirm": {"choice_id": "cancel"}, "default": {"choice_id": "cancel"}}
            ref = "Khi khách chọn cancel / không tiếp tục, hệ thống dừng yêu cầu an toàn và không thực hiện hành động."
            must = ["dừng"]
        items.append(_make_ds03_item(
            next_id,
            "G3_hitl_confirm_approve_reject_cancel",
            "hard",
            "P0",
            [q],
            ref,
            must,
            [{"action": action, "variant": var}],
            hitl_replies=replies,
        ))
        next_id += 1

    # G4: Interaction Budget Exhaustion MAX_INTERACTIONS=3 (100 scenarios)
    vague_turns = [
        "Tôi muốn tính giúp một khoản tiền của tôi.",
        "Số tiền cũng bình thường thôi, cứ xem giúp tôi.",
        "Kỳ hạn tùy ngân hàng chọn giúp.",
        "Cứ chốt phương án phù hợp nhất đi.",
    ]
    for k in range(100):
        t_list = [f"[Ngân sách tương tác #{k + 1}] {vague_turns[0]}"] + vague_turns[1:]
        items.append(_make_ds03_item(
            next_id,
            "G4_hitl_interaction_budget_exhaustion",
            "hard",
            "P0",
            t_list,
            "Sau tối đa 3 lượt làm rõ nhưng người dùng vẫn không cung cấp đủ tham số (số tiền, kỳ hạn, nhu cầu cụ thể), hệ thống dừng hỏi thêm, thông báo thiếu thông tin để tiếp tục và mời khách hàng bổ sung chi tiết.",
            ["thiếu"],
            [{"max_interactions": 3, "variant": k + 1}],
            hitl_replies={"default": {"text": "Chưa xác định được cụ thể, tùy hệ thống tư vấn."}},
        ))
        next_id += 1

    # G5: Long Session Memory Remember/Forget/Compact & Answer Scope (300 scenarios)
    # 5a: Long-turn remember + compaction >12 messages (200 cases, 7 user turns -> 14 msgs total!)
    filler_topics = [
        "Cho biết giờ làm việc hỗ trợ chuyển tiền ngoài giờ theo mẫu.",
        "Thẻ ghi nợ nội địa rút tiền ở ATM A Bank có mất phí không?",
        "Nộp tiền mặt VND tại quầy mất phí thế nào?",
        "Quyết định 2345 yêu cầu sinh trắc học từ mức tiền nào?",
        "Phí mở tài khoản thanh toán VND cá nhân tại quầy là bao nhiêu?",
        "Đổi tiền không đủ tiêu chuẩn lưu thông trên app có làm được không?",
    ]
    user_facts = [
        ("loan_amount_goal", "2 tỷ đồng"),
        ("preferred_city", "Đà Nẵng"),
        ("card_interest", "Eco Visa"),
        ("insurance_plan", "A-Life Trọn Đời"),
        ("savings_goal", "800 triệu đồng"),
    ]
    for k in range(200):
        key_name, fact_val = user_facts[k % len(user_facts)]
        var = (k // len(user_facts)) + 1
        t0 = f"Hãy ghi nhớ thông tin cá nhân của tôi (khóa `{key_name}`): '{fact_val} - hồ sơ #{var}'."
        middle_turns = [filler_topics[(k + j) % len(filler_topics)] for j in range(5)]
        if k % 4 == 0:
            final_turns = [
                t0,
                *middle_turns,
                f"Hãy xóa thông tin đã nhớ ở khóa `{key_name}` giúp tôi.",
                f"Kiểm tra lại xem thông tin về '{key_name}' trước đó còn được lưu không?",
            ]
            ref = f"Hệ thống đã xóa mục nhớ '{key_name}' theo yêu cầu khách hàng và xác nhận không còn lưu giá trị cũ."
            must = ["xóa"]
        else:
            final_turns = [
                t0,
                *middle_turns,
                f"Nhắc lại chính xác thông tin tôi đã nhờ ghi nhớ ở đầu phiên (hồ sơ #{var}) là gì?",
            ]
            ref = f"Dù đã qua nhiều lượt trao đổi kích hoạt nén bộ nhớ, hệ thống vẫn nhắc chính xác thông tin đã ghi nhớ: '{fact_val} - hồ sơ #{var}'."
            must = [fact_val]

        if k < 80:
            prio_g5 = "P0"
        else:
            prio_g5 = "P1"
        items.append(_make_ds03_item(
            next_id,
            "G5_long_session_memory_compact_scope",
            "hard",
            prio_g5,
            final_turns,
            ref,
            must,
            [{"memory_key": key_name, "fact": fact_val, "turns": len(final_turns)}],
        ))
        next_id += 1

    # 5b: ANSWER_SCOPE & SUMMARY_MAX_WORDS=160 tests (100 cases)
    all_scope_docs = nv_docs + card_docs + bh_docs
    for k in range(100):
        src_doc = all_scope_docs[k % len(all_scope_docs)]
        doc_title = src_doc.rsplit("/", 1)[-1].removesuffix(".docx").removesuffix(".pdf")
        if k % 2 == 0:
            turns = [
                f"[Kiểm thử giới hạn tóm tắt #{k}] Hãy tóm tắt ngắn gọn các điểm chính trong tài liệu '{doc_title}' (tối đa 5 ý, súc tích).",
            ]
            ref = f"Tóm tắt cô đọng các ý cốt lõi của '{doc_title}' trong giới hạn <= 160 từ và <= 5 gạch đầu dòng."
            max_w = 175
        else:
            turns = [
                f"[Phiên hủy tóm tắt #{k}] Hãy tóm tắt ngắn quy định trong tài liệu '{doc_title}'.",
                "Không tóm tắt nữa, hãy trình bày đầy đủ và chi tiết các điều kiện, hạn mức/quy định chính trong tài liệu đó.",
            ]
            ref = f"Trình bày đầy đủ nội dung chi tiết của '{doc_title}' theo yêu cầu hủy chế độ tóm tắt của khách hàng."
            max_w = None

        items.append(_make_ds03_item(
            next_id,
            "G5_long_session_memory_compact_scope",
            "medium",
            "P1",
            turns,
            ref,
            [doc_title.split()[0]],
            [{"source": src_doc, "summary_mode": (k % 2 == 0)}],
            expected_tools=["search_banking_knowledge"],
            max_words=max_w,
        ))
        next_id += 1

    return items


def main() -> None:
    items = build_ds03()
    cov = export_suite_bundle(items, OUT_DIR, "eval_multiturn_hitl", "DS-03", canary_size=30)
    print(json.dumps(cov, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
