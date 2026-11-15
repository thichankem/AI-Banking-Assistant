"""
Script to format A_Bank_golden_master_785.xlsx to contain ONLY ONE sheet:
'Master 785 Golden (Chưa chấm)'

Columns kept:
1. STT
2. ID Câu hỏi
3. Mã VB
4. Tiêu chí Kiểm thử
5. Độ khó
6. Câu hỏi Tình huống (Question)
7. Đáp án Chuẩn (Reference Answer)
8. Tiêu chí Bắt buộc (Must Include)
9. Đường dẫn Nguồn (Source Path)
10. Câu trả lời của Trợ lý AI Banking (Agent Answer)

Columns removed as requested by user:
- Nhóm nghiệp vụ
- Dạng bài toán
- Lỗi cần tránh (Must Avoid)
- Mục tiêu kiểm chứng / kiểm thử (Table/Figure)
- Điều khoản tham chiếu
- Bằng chứng trích dẫn

MANDATORY CODING RULES:
1. 0% REGEX LAW
2. ZERO TERNARY LAW (Explicit if/else)
3. NO TYPING LIBRARY
4. NO __INIT__.PY
"""

import os
import json
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

def format_excel():
    base_dir = r"d:\Desktop\GitHub\AI Banking Asistant"

    groups_info = [
        os.path.join(base_dir, "dataset", "A_Bank_golden_the", "a_bank_the_golden.json"),
        os.path.join(base_dir, "dataset", "A_Bank_golden_compliance", "a_bank_compliance_golden.json"),
        os.path.join(base_dir, "dataset", "A_Bank_golden_app", "a_bank_app_golden.json"),
        os.path.join(base_dir, "dataset", "A_Bank_golden_specialized", "a_bank_specialized_golden.json")
    ]

    all_cases = []
    for fpath in groups_info:
        with open(fpath, "r", encoding="utf-8") as f:
            cases = json.load(f)
            for c in cases:
                all_cases.append(c)

    wb = openpyxl.Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    ws = wb.create_sheet(title="Master 785 Golden (Chưa chấm)")
    ws.views.sheetView[0].showGridLines = True
    ws.freeze_panes = "A2"

    navy_fill = PatternFill(start_color="0F2240", end_color="0F2240", fill_type="solid")
    gold_fill = PatternFill(start_color="B45309", end_color="B45309", fill_type="solid")
    header_font = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")
    data_font = Font(name="Segoe UI", size=9, bold=False, color="1E293B")
    bold_font = Font(name="Segoe UI", size=9, bold=True, color="0F2240")

    thin_border = Border(
        left=Side(style="thin", color="CBD5E1"),
        right=Side(style="thin", color="CBD5E1"),
        top=Side(style="thin", color="CBD5E1"),
        bottom=Side(style="thin", color="CBD5E1")
    )

    align_left_wrap = Alignment(horizontal="left", vertical="top", wrap_text=True)
    align_center = Alignment(horizontal="center", vertical="top", wrap_text=True)

    headers = [
        "STT",
        "ID Câu hỏi",
        "Mã VB",
        "Tiêu chí Kiểm thử",
        "Độ khó",
        "Câu hỏi Tình huống (Question)",
        "Đáp án Chuẩn (Reference Answer)",
        "Tiêu chí Bắt buộc (Must Include)",
        "Đường dẫn Nguồn (Source Path)",
        "Câu trả lời của Trợ lý AI Banking (Agent Answer)"
    ]

    for col_idx, h_text in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_idx, value=h_text)
        if col_idx == 10:
            cell.fill = gold_fill
        else:
            cell.fill = navy_fill
        cell.font = header_font
        cell.alignment = align_center
        cell.border = thin_border

    # Check if existing cache of predictions exists
    answers_cache = {}
    cache_path = os.path.join(base_dir, "dataset", "agent_answers_cache.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f_c:
                answers_cache = json.load(f_c)
        except Exception:
            pass

    for idx, case in enumerate(all_cases, 1):
        r_num = idx + 1
        case_id = case.get("id", "")
        topic_id = case.get("topic_id", "")
        test_crit = case.get("test_criteria", "")
        diff = case.get("difficulty", "")
        question = case.get("question", "")
        ref_ans = case.get("reference_answer", "")

        rubric = case.get("rubric", {})
        must_inc = rubric.get("must_include", [])
        must_inc_str = ""
        for item in must_inc:
            must_inc_str += " • " + item + "\n"
        must_inc_str = must_inc_str.strip()

        evidence_list = case.get("evidence", [])
        src_list = []
        for ev in evidence_list:
            sp = ev.get("source_path", "")
            if sp:
                if sp not in src_list:
                    src_list.append(sp)
        src_str = "\n".join(src_list)

        agent_ans = ""
        if case_id in answers_cache:
            agent_ans = answers_cache[case_id]
        else:
            agent_ans = "[Chờ xử lý qua Agent Banking...]"

        row_vals = [
            idx,
            case_id,
            topic_id,
            test_crit,
            diff,
            question,
            ref_ans,
            must_inc_str,
            src_str,
            agent_ans
        ]

        for col_idx, val in enumerate(row_vals, 1):
            cell = ws.cell(row=r_num, column=col_idx, value=val)
            cell.font = data_font
            cell.border = thin_border
            if col_idx in [1, 2, 3, 5]:
                cell.alignment = align_center
            else:
                cell.alignment = align_left_wrap

    col_widths = {
        "A": 6,
        "B": 12,
        "C": 10,
        "D": 26,
        "E": 10,
        "F": 45,
        "G": 45,
        "H": 34,
        "I": 35,
        "J": 50
    }
    for col_letter, w in col_widths.items():
        ws.column_dimensions[col_letter].width = w

    out_file = os.path.join(base_dir, "dataset", "A_Bank_golden_master_785.xlsx")
    wb.save(out_file)
    print("Formatted Excel Workbook saved to: " + out_file)
    print("Total rows:", len(all_cases) + 1)

if __name__ == "__main__":
    format_excel()
