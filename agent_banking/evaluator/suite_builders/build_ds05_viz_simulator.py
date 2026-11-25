"""Build DS-05: 1,000 Visualization, Mermaid Diagram & Interactive Simulator Cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_banking.evaluator.suite_export import export_suite_bundle

OUT_DIR = Path("dataset/A_Bank_eval_viz_simulator")


def _make_ds05_item(
    idx_num: int,
    group: str,
    difficulty: str,
    priority: str,
    turns: list[str],
    ref_answer: str,
    must_include: list[str],
    artifact_contract: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": f"DS05-{idx_num:04d}",
        "dataset": "DS-05",
        "group": group,
        "capability": "viz_simulator",
        "difficulty": difficulty,
        "priority": priority,
        "question": turns[0],
        "turns": turns,
        "profile": {"persona": "advisor", "tone": "formal", "format_type": "standard", "custom_persona": ""},
        "reference_answer": ref_answer,
        "rubric": {"must_include": must_include, "must_not_include": []},
        "checks": {
            "expected_tools": ["present_visualization"],
            "forbidden_tools": [],
            "expected_blocked": None,
            "numeric_targets": [],
            "artifact_contract": artifact_contract,
            "citation_contract": None,
            "style_contract": None,
        },
        "evidence": evidence,
    }


def build_ds05() -> list[dict[str, Any]]:
    """Generate 1,000 ECharts, Mermaid, Simulator, and multi-turn update_of cases."""
    items: list[dict[str, Any]] = []
    next_id = 1

    # G1: ECharts All 7 Chart Types (300 cases)
    echarts_configs = [
        ("bar", "So sánh Tổng tài sản (382.863 tỷ -> 508.330 tỷ -> 605.585 tỷ) giai đoạn 2023-2025 theo BC 01/BC 02", ["508", "605"], "BC 01 & BC 02"),
        ("stacked_bar", "Cơ cấu Tổng thu nhập hoạt động năm 2024 và 2025 gồm Thu nhập lãi thuần (15.394 -> 16.034 tỷ) và Thu nhập ngoài lãi (4.539 -> 5.847 tỷ)", ["16", "5"], "BC 02 p.14"),
        ("line", "Diễn biến chỉ số ROE (19% -> 25,10% -> 25,23%) và CAR (12,24% -> 13,33% -> 11,87%) qua các năm 2023-2025", ["25", "11"], "BC 01 & BC 02"),
        ("area", "Xu hướng tăng trưởng Tiền gửi tổ chức kinh tế và cá nhân (237.392 -> 283.172 -> 337.583 tỷ đồng) từ 2023 đến 2025", ["283", "337"], "BC 01 & BC 02"),
        ("pie", "Tỷ trọng 1.260 điểm mạng lưới ML 01: ATM (685), Phòng GD (481), Chi nhánh (85), CDM (9)", ["685", "481"], "ML 01"),
        ("donut", "Cơ cấu Nguồn thu nhập năm 2025: Thu nhập lãi thuần 16.034 tỷ đồng so với Thu nhập ngoài lãi 5.847 tỷ đồng", ["16", "5"], "BC 02 p.14"),
        ("scatter", "Tương quan giữa các kỳ hạn gửi tiết kiệm (1, 3, 6, 12, 24 tháng) và mức lãi suất mẫu (1.6%, 1.9%, 2.9%, 4.6%, 4.7%)", ["1.6", "4.7"], "banking_data.yaml"),
    ]
    focus_slices = [
        "Hiển thị rõ nhãn đơn vị trên trục",
        "Đặt tiêu đề song ngữ rõ ràng",
        "So sánh trực quan từng giai đoạn",
        "Chuẩn hóa đơn vị tỷ đồng hoặc %",
        "Sắp xếp dữ liệu theo thứ tự logic",
        "Nêu ngắn gọn điểm nổi bật dưới biểu đồ",
        "Phân tách màu sắc rõ giữa các nhóm",
    ]
    for k in range(300):
        ctype, desc, must, src_hint = echarts_configs[k % len(echarts_configs)]
        var = (k // len(echarts_configs)) + 1
        extra_req = focus_slices[(k // len(echarts_configs)) % len(focus_slices)]
        q = (
            f"[Yêu cầu biểu đồ #{var}] Hãy vẽ biểu đồ ECharts dạng '{ctype}' thể hiện: {desc}. "
            f"Lưu ý: {extra_req} (lượt #{k + 1})."
        )
        ref = f"Sử dụng tool present_visualization dựng biểu đồ ECharts kind='echarts', chart_type='{ctype}' với số liệu chuẩn từ {src_hint}."
        if k % 2 == 0:
            diff_g1 = "medium"
        else:
            diff_g1 = "hard"
        if k < 90:
            prio_g1 = "P0"
        else:
            prio_g1 = "P1"
        items.append(_make_ds05_item(
            next_id,
            "G1_echarts_all_7_types",
            diff_g1,
            prio_g1,
            [q],
            ref,
            must,
            {"kind": "echarts", "chart_type": ctype, "update_expected": False},
            [{"source": src_hint, "chart_type": ctype, "variant": var}],
        ))
        next_id += 1

    # G2: Mermaid All 6 Diagram Types — Strictly 100% Mermaid, Zero Draw.io (200 cases)
    mermaid_configs = [
        ("flowchart", "Quy trình xử lý yêu cầu tra soát giao dịch chuyển tiền nội địa và thời hạn SLA theo VH 01/VH 03", ["tra soát"], "VH 01 & VH 03"),
        ("sequence", "Luồng xác thực giao dịch trực tuyến > 10 triệu đồng trên A-Bank Plus qua Facepay và Smart OTP theo Quyết định 2345", ["2345"], "HDSD 03 & PL"),
        ("class", "Sơ đồ phân cấp các dòng thẻ tín dụng A-Bank gồm Cashback Visa, Eco Visa, Edu Visa và Corporate Visa", ["Visa"], "THE 03, 07, 13, 14"),
        ("er", "Mô hình thực thể liên kết (ER) giữa Khách hàng cá nhân, Tài khoản thanh toán, Thẻ ghi nợ và Lịch sử giao dịch", ["Tài khoản"], "NV 01 & NV 02"),
        ("mindmap", "Sơ đồ tư duy hệ sinh thái quyền lợi và điều khoản loại trừ bảo hiểm nhân thọ A-Life Trọn Đời", ["A-Life"], "BH 01"),
        ("architecture", "Kiến trúc tổng thể trợ lý AI Banking gồm Guardrail, Hybrid Qdrant + BM25, Cloud Reranker và Tool Sandbox", ["Qdrant"], "System Docs"),
    ]
    for k in range(200):
        dtype, desc, must, src_hint = mermaid_configs[k % len(mermaid_configs)]
        var = (k // len(mermaid_configs)) + 1
        q = (
            f"[Sơ đồ Mermaid #{var}] Trực quan hóa bằng sơ đồ Mermaid dạng '{dtype}' cho nội dung: {desc} "
            f"(lượt #{k + 1}, tuyệt đối không dùng drawio)."
        )
        ref = f"Sử dụng present_visualization với kind='mermaid' và diagram_type='{dtype}' tuân thủ chuẩn 100% Mermaid cho chủ đề {src_hint}."
        if k < 80:
            prio_g2 = "P0"
        else:
            prio_g2 = "P1"
        items.append(_make_ds05_item(
            next_id,
            "G2_mermaid_all_6_types",
            "hard",
            prio_g2,
            [q],
            ref,
            must,
            {"kind": "mermaid", "diagram_type": dtype, "update_expected": False},
            [{"source": src_hint, "diagram_type": dtype, "variant": var}],
        ))
        next_id += 1

    # G3: Interactive Simulator Calculators (300 cases)
    sim_ops = [
        ("pmt", "Tạo bộ mô phỏng tính tiền trả góp cố định hàng tháng (PMT) cho khoản vay {p} triệu đồng trong {n} tháng, lãi suất {r}%/năm.", "pmt"),
        ("declining_interest", "Tạo công cụ mô phỏng trả nợ theo dư nợ giảm dần cho khoản vay {p} triệu trong {n} tháng, lãi suất {r}%/năm.", "declining_interest"),
        ("power", "Mở bộ mô phỏng tăng trưởng lãi kép cho số vốn ban đầu {p} triệu đồng tích lũy {n} tháng với lãi suất {r}%/năm.", "power"),
        ("multiply", "Tạo bộ tính tương tác quy đổi phí giao dịch theo tỷ lệ phần trăm cho số tiền gốc {p} triệu đồng (lượt #{k}).", "multiply"),
        ("max", "Tạo bộ mô phỏng tính phí chuyển tiền có áp mức sàn tối thiểu (max giữa phí % và sàn tối thiểu) cho khoản {p} triệu.", "max"),
        ("min", "Tạo bộ mô phỏng tính phí có áp mức trần tối đa (min giữa phí tính và trần phí) cho giá trị {p} triệu.", "min"),
        ("add", "Tạo bộ tính tổng chi phí gốc + phí phát hành + phụ phí hồ sơ cho khoản {p} triệu đồng.", "add"),
        ("subtract", "Tạo bộ mô phỏng số dư thực nhận sau khi khấu trừ khoản phí dịch vụ trên số tiền {p} triệu đồng.", "subtract"),
        ("divide", "Tạo bộ mô phỏng chia bình quân chi phí hàng tháng trong {n} tháng cho ngân sách {p} triệu đồng.", "divide"),
    ]
    sim_ps = [200, 350, 500, 850, 1200, 2000, 3500]
    sim_ns = [12, 24, 36, 48, 60, 120]
    sim_rs = [6.5, 7.8, 8.5, 9.2, 10.5]
    for k in range(300):
        op_code, tmpl, must_op = sim_ops[k % len(sim_ops)]
        p_m = sim_ps[(k // len(sim_ops)) % len(sim_ps)]
        n_m = sim_ns[(k // (len(sim_ops) * len(sim_ps))) % len(sim_ns)] + (k % 3) * 6
        r_v = sim_rs[k % len(sim_rs)]
        q = f"[Mô phỏng tương tác #{k + 1}] " + tmpl.format(p=p_m, n=n_m, r=r_v, k=k + 1)
        ref = f"Khởi tạo trình mô phỏng tương tác Interactive Simulator (operation='{op_code}') với các thanh trượt đầu vào mặc định khớp tham số khách hàng."
        if k < 90:
            prio_g3 = "P0"
        else:
            prio_g3 = "P1"
        items.append(_make_ds05_item(
            next_id,
            "G3_interactive_simulator",
            "hard",
            prio_g3,
            [q],
            ref,
            [str(p_m)],
            {"kind": "simulator", "operation": must_op, "update_expected": False},
            [{"operation": op_code, "principal_m": p_m, "months": n_m, "rate": r_v}],
        ))
        next_id += 1

    # G4: Multi-Turn Artifact Revision with update_of (200 cases)
    for k in range(200):
        amt_1 = 400 + (k % 10) * 100
        amt_2 = amt_1 + 300
        if k % 2 == 0:
            turns = [
                f"[Biểu đồ động #{k + 1}] Vẽ biểu đồ cột (bar) so sánh lãi suất tiết kiệm mẫu A-Bank kỳ hạn 1, 3, 6, 12, 24 tháng.",
                f"Hãy giữ dữ liệu cũ nhưng cập nhật lại (update) biểu đồ vừa vẽ thành dạng biểu đồ đường (line) cho dễ nhìn xu hướng.",
            ]
            contract = {"kind": "echarts", "chart_type": "line", "update_expected": True}
            ref = "Lượt 1 vẽ biểu đồ bar; lượt 2 cập nhật artifact vừa tạo sang chart_type='line' với update_of trỏ về artifact cũ."
            must = ["line"]
        else:
            turns = [
                f"[Mô phỏng động #{k + 1}] Tạo bộ mô phỏng tính trả góp dư nợ giảm dần cho khoản vay {amt_1} triệu trong 36 tháng lãi 8,5%/năm.",
                f"Hãy cập nhật lại mặc định trong bộ mô phỏng vừa tạo lên mức vay {amt_2} triệu đồng và kỳ hạn 60 tháng.",
            ]
            contract = {"kind": "simulator", "operation": "declining_interest", "update_expected": True}
            ref = f"Lượt 2 cập nhật giá trị mặc định của bộ mô phỏng vừa tạo thành {amt_2} triệu đồng và 60 tháng."
            must = [str(amt_2)]

        if k < 80:
            prio_g4 = "P0"
        else:
            prio_g4 = "P1"
        items.append(_make_ds05_item(
            next_id,
            "G4_multiturn_update_of",
            "hard",
            prio_g4,
            turns,
            ref,
            must,
            contract,
            [{"turn_update": True, "variant": k + 1}],
        ))
        next_id += 1

    return items


def main() -> None:
    items = build_ds05()
    cov = export_suite_bundle(items, OUT_DIR, "eval_viz_simulator", "DS-05", canary_size=35)
    print(json.dumps(cov, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
