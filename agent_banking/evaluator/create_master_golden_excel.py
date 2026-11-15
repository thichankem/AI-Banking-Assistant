"""
Script to generate Master Golden Excel Workbook for A-Bank 785 cases.
Includes:
- Sheet 1: Dashboard & Multi-Criteria Evaluation Framework
- Sheet 2: Master 785 Golden Dataset (Unscored / Original)
- Sheet 3: Multi-Criteria Evaluation Sheet (With built-in formulas and score rubrics)
- Sheets 4-7: Individual folders (Thẻ, Tuân thủ, App, Nghiệp vụ chuyên sâu)

DESIGN PRINCIPLES:
1. 0% REGEX LAW: No regex imports or functions.
2. ZERO TERNARY: Explicit if/else statements only.
3. NO TYPING LIBRARY: Native types only.
4. NO __INIT__.PY: Direct module imports.
"""

import os
import json
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

def build_master_workbook():
    base_dir = r"d:\Desktop\GitHub\AI Banking Asistant"

    groups_info = [
        ("Folder 2: Nghiệp vụ Thẻ (200 câu)", os.path.join(base_dir, "dataset", "A_Bank_golden_the", "a_bank_the_golden.json")),
        ("Folder 3: Pháp lý Tuân thủ & KYC (200 câu)", os.path.join(base_dir, "dataset", "A_Bank_golden_compliance", "a_bank_compliance_golden.json")),
        ("Folder 4: App Ngân hàng số A-Bank Plus (195 câu)", os.path.join(base_dir, "dataset", "A_Bank_golden_app", "a_bank_app_golden.json")),
        ("Folders 5-6-7: Bảo hiểm, BCTC, SLA (190 câu)", os.path.join(base_dir, "dataset", "A_Bank_golden_specialized", "a_bank_specialized_golden.json"))
    ]

    all_785 = []
    group_stats = {}

    for gname, fpath in groups_info:
        with open(fpath, "r", encoding="utf-8") as f:
            cases = json.load(f)
            for c in cases:
                c["group_name"] = gname
                all_785.append(c)
            group_stats[gname] = len(cases)

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    # Styles
    navy_fill = PatternFill(start_color="0F2240", end_color="0F2240", fill_type="solid")
    gold_fill = PatternFill(start_color="B45309", end_color="B45309", fill_type="solid")
    blue_fill = PatternFill(start_color="1E40AF", end_color="1E40AF", fill_type="solid")
    header_font = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")
    data_font = Font(name="Segoe UI", size=9, bold=False, color="1E293B")
    bold_font = Font(name="Segoe UI", size=9, bold=True, color="0F2240")
    title_font = Font(name="Segoe UI", size=13, bold=True, color="0F2240")

    thin_border = Border(
        left=Side(style="thin", color="CBD5E1"),
        right=Side(style="thin", color="CBD5E1"),
        top=Side(style="thin", color="CBD5E1"),
        bottom=Side(style="thin", color="CBD5E1")
    )

    align_left_wrap = Alignment(horizontal="left", vertical="top", wrap_text=True)
    align_center = Alignment(horizontal="center", vertical="top", wrap_text=True)

    # -------------------------------------------------------------
    # SHEET 1: DASHBOARD & EVALUATION FRAMEWORK
    # -------------------------------------------------------------
    ws_dash = wb.create_sheet(title="Dashboard & Tiêu chí Chấm")
    ws_dash.views.sheetView[0].showGridLines = True

    ws_dash.cell(row=2, column=2, value="BẢNG ĐIỀU KHIỂN & KHUNG TIÊU CHÍ CHẤM ĐIỂM BỘ 785 CÂU GOLDEN QUESTIONS").font = title_font
    ws_dash.cell(row=3, column=2, value="Dự án: AI Banking Assistant - Ngân hàng Thương mại Cổ phần A-Bank | Tiêu chuẩn thẩm định 2 tầng").font = Font(name="Segoe UI", size=10, italic=True, color="64748B")

    # Table 1: Quy mô dữ liệu 4 folder
    ws_dash.cell(row=5, column=2, value="1. QUY MÔ PHÂN BỔ 785 CÂU THEO NHÓM NGHIỆP VỤ").font = bold_font
    tbl1_headers = ["STT", "Nhóm Nghiệp vụ", "Quy mô (Số câu)", "Tỷ lệ %", "Mã định danh"]
    for c_idx, h in enumerate(tbl1_headers, start=2):
        cell = ws_dash.cell(row=6, column=c_idx, value=h)
        cell.fill = navy_fill
        cell.font = header_font
        cell.alignment = align_center

    r_idx = 7
    stt = 1
    for gname, count in group_stats.items():
        ws_dash.cell(row=r_idx, column=2, value=stt).alignment = align_center
        ws_dash.cell(row=r_idx, column=3, value=gname).alignment = align_left_wrap
        ws_dash.cell(row=r_idx, column=4, value=count).alignment = align_center
        rate_str = str(round((count / 785.0) * 100.0, 1)) + "%"
        ws_dash.cell(row=r_idx, column=5, value=rate_str).alignment = align_center
        
        file_code = "the"
        if "Pháp lý" in gname:
            file_code = "compliance"
        else:
            if "App" in gname:
                file_code = "app"
            else:
                if "Bảo hiểm" in gname:
                    file_code = "specialized"
        ws_dash.cell(row=r_idx, column=6, value="a_bank_" + file_code + "_golden").alignment = align_center
        
        for c in range(2, 7):
            ws_dash.cell(row=r_idx, column=c).border = thin_border
            ws_dash.cell(row=r_idx, column=c).font = data_font
        r_idx += 1
        stt += 1

    # Total row
    ws_dash.cell(row=r_idx, column=2, value="Tổng").alignment = align_center
    ws_dash.cell(row=r_idx, column=3, value="Toàn bộ 4 nhóm nghiệp vụ (gần 800 câu)").alignment = align_left_wrap
    ws_dash.cell(row=r_idx, column=4, value=785).alignment = align_center
    ws_dash.cell(row=r_idx, column=5, value="100.0%").alignment = align_center
    ws_dash.cell(row=r_idx, column=6, value="785 test cases").alignment = align_center
    for c in range(2, 7):
        ws_dash.cell(row=r_idx, column=c).fill = PatternFill(start_color="E2E8F0", end_color="E2E8F0", fill_type="solid")
        ws_dash.cell(row=r_idx, column=c).font = bold_font
        ws_dash.cell(row=r_idx, column=c).border = thin_border

    # Table 2: 4 Tiêu chí kiểm thử
    r_idx += 3
    ws_dash.cell(row=r_idx, column=2, value="2. KHUNG 4 TIÊU CHÍ KIỂM THỬ NGHIỆP VỤ (TEST CRITERIA)").font = bold_font
    r_idx += 1
    tbl2_headers = ["STT", "Tiêu chí Kiểm thử", "Số lượng câu", "Tỷ lệ %", "Mô tả Nghiệp vụ Mục tiêu"]
    for c_idx, h in enumerate(tbl2_headers, start=2):
        cell = ws_dash.cell(row=r_idx, column=c_idx, value=h)
        cell.fill = navy_fill
        cell.font = header_font
        cell.alignment = align_center

    crit_counts = {}
    for c in all_785:
        k = c.get("test_criteria", "Khác")
        crit_counts[k] = crit_counts.get(k, 0) + 1

    crit_desc = {
        "Table Reasoning & Matrix Lookup": "Truy vấn ma trận biểu phí, hạn mức thẻ/giao dịch, bảng số liệu tài chính BCTC",
        "Figure, Flowchart & Visual Diagram Reasoning": "Suy luận quy trình bồi thường, lưu đồ phê duyệt, luồng thao tác màn hình App",
        "Multi-Condition & Eligibility": "Đối soát tổ hợp điều kiện phát hành thẻ, mở tài khoản, quyền lợi BH, tiêu chuẩn mở",
        "Safety, False Premise & Edge Cases": "Từ chối an toàn, nhận diện tiền đề sai, giới hạn quyền riêng tư KYC/AML, ngoại lệ"
    }

    r_idx += 1
    stt = 1
    for crit, count in crit_counts.items():
        ws_dash.cell(row=r_idx, column=2, value=stt).alignment = align_center
        ws_dash.cell(row=r_idx, column=3, value=crit).alignment = align_left_wrap
        ws_dash.cell(row=r_idx, column=4, value=count).alignment = align_center
        crit_rate_str = str(round((count / 785.0) * 100.0, 1)) + "%"
        ws_dash.cell(row=r_idx, column=5, value=crit_rate_str).alignment = align_center
        
        desc_val = ""
        if crit in crit_desc:
            desc_val = crit_desc[crit]
        ws_dash.cell(row=r_idx, column=6, value=desc_val).alignment = align_left_wrap
        
        for c in range(2, 7):
            ws_dash.cell(row=r_idx, column=c).border = thin_border
            ws_dash.cell(row=r_idx, column=c).font = data_font
        r_idx += 1
        stt += 1

    # Table 3: Cơ chế Chấm Điểm Đa Tiêu Chí
    r_idx += 3
    ws_dash.cell(row=r_idx, column=2, value="3. CƠ CHẾ CHẤM ĐIỂM ĐA TIÊU CHÍ (2-TIER EVALUATION FORMULA)").font = bold_font
    r_idx += 1
    tbl3_headers = ["Tầng đánh giá", "Tiêu chí thành phần", "Trọng số", "Nguyên tắc chấm & Chế tài", "Công cụ thực thi"]
    for c_idx, h in enumerate(tbl3_headers, start=2):
        cell = ws_dash.cell(row=r_idx, column=c_idx, value=h)
        cell.fill = gold_fill
        cell.font = header_font
        cell.alignment = align_center

    eval_rules = [
        ("Tier-1: Rule Verifier", "Must-Include (Từ khóa & Ý bắt buộc)", "70% Rule Score", "Khớp chính xác các mốc hạn mức, con số, thuật ngữ nghiệp vụ", "golden_evaluator.py (Rule Engine)"),
        ("Tier-1: Rule Verifier", "Grounding & Citation (Trích dẫn nguồn)", "30% Rule Score", "Kiểm tra độ phủ nguồn tài liệu, số điều khoản, tên bảng/hình", "golden_evaluator.py (Rule Engine)"),
        ("Tier-1: Rule Verifier", "Must-Avoid (Lỗi cấm & Giả định sai)", "Phạt 70% Điểm", "Nếu xuất hiện thông tin cấm hoặc sai số nguy hiểm: Phạt nhân hệ số 0.3", "Safety Filter"),
        ("Tier-2: LLM Judge", "Semantic Correctness & Professionalism", "60% Final Score", "Chấm độ chuẩn xác ngữ nghĩa, văn phong ngân hàng, tính đầy đủ", "DeepSeek-V4-Flash (FPT Cloud API)"),
        ("Composite Score", "Tổng điểm Đa tiêu chí Toàn diện", "100.0 Điểm", "Final Score = (Rule Score * 0.4) + (LLM Judge * 0.6)", "Ngưỡng Đạt: Score >= 80 & 0 lỗi cấm")
    ]

    r_idx += 1
    for t_tier, t_crit, t_w, t_rule, t_tool in eval_rules:
        ws_dash.cell(row=r_idx, column=2, value=t_tier).alignment = align_left_wrap
        ws_dash.cell(row=r_idx, column=3, value=t_crit).alignment = align_left_wrap
        ws_dash.cell(row=r_idx, column=4, value=t_w).alignment = align_center
        ws_dash.cell(row=r_idx, column=5, value=t_rule).alignment = align_left_wrap
        ws_dash.cell(row=r_idx, column=6, value=t_tool).alignment = align_left_wrap
        for c in range(2, 7):
            ws_dash.cell(row=r_idx, column=c).border = thin_border
            ws_dash.cell(row=r_idx, column=c).font = data_font
        r_idx += 1

    ws_dash.column_dimensions["B"].width = 24
    ws_dash.column_dimensions["C"].width = 38
    ws_dash.column_dimensions["D"].width = 18
    ws_dash.column_dimensions["E"].width = 16
    ws_dash.column_dimensions["F"].width = 46

    # -------------------------------------------------------------
    # SHEET 2: MASTER 785 GOLDEN QUESTIONS (CHƯA CHẤM)
    # -------------------------------------------------------------
    ws_master = wb.create_sheet(title="Master 785 Golden (Chưa chấm)")
    ws_master.views.sheetView[0].showGridLines = True
    ws_master.freeze_panes = "A2"

    master_headers = [
        "STT", "ID Câu hỏi", "Nhóm Nghiệp vụ", "Mã VB", "Tiêu chí Kiểm thử",
        "Dạng Bài toán", "Độ khó", "Câu hỏi Tình huống (Question)",
        "Đáp án Chuẩn (Reference Answer)", "Tiêu chí Bắt buộc (Must Include)",
        "Lỗi cần Tránh (Must Avoid)", "Mục tiêu Kiểm chứng (Table/Figure)",
        "Điều khoản Tham chiếu", "Bằng chứng Trích dẫn (Verbatim Quote)",
        "Đường dẫn Nguồn (Source Path)"
    ]

    for col_num, h_text in enumerate(master_headers, 1):
        cell = ws_master.cell(row=1, column=col_num, value=h_text)
        cell.fill = navy_fill
        cell.font = header_font
        cell.alignment = align_center
        cell.border = thin_border

    for idx, case in enumerate(all_785, 1):
        rubric = case.get("rubric", {})
        must_inc = rubric.get("must_include", [])
        must_av = rubric.get("must_avoid", [])
        
        must_inc_str = ""
        for item in must_inc:
            must_inc_str += " • " + item + "\n"
        must_inc_str = must_inc_str.strip()
        
        must_av_str = ""
        for item in must_av:
            must_av_str += " • " + item + "\n"
        must_av_str = must_av_str.strip()
        
        evidence_list = case.get("evidence", [])
        tgt_list = []
        sec_list = []
        quote_list = []
        src_list = []
        for ev in evidence_list:
            if ev.get("target_artifact"):
                tgt_list.append(ev.get("target_artifact"))
            if ev.get("section"):
                sec_list.append(ev.get("section"))
            if ev.get("quote"):
                quote_list.append(ev.get("quote"))
            if ev.get("source_path"):
                src_list.append(ev.get("source_path"))
                
        tgt_str = "\n".join(tgt_list)
        sec_str = "\n".join(sec_list)
        quote_str = "\n".join(quote_list)
        src_str = "\n".join(src_list)
        
        row_data = [
            idx,
            case.get("id", ""),
            case.get("group_name", ""),
            case.get("topic_id", ""),
            case.get("test_criteria", ""),
            case.get("case_type", ""),
            case.get("difficulty", ""),
            case.get("question", ""),
            case.get("reference_answer", ""),
            must_inc_str,
            must_av_str,
            tgt_str,
            sec_str,
            quote_str,
            src_str
        ]
        
        for col_num, val in enumerate(row_data, 1):
            cell = ws_master.cell(row=idx + 1, column=col_num, value=val)
            cell.font = data_font
            cell.border = thin_border
            if col_num in [1, 2, 4, 6, 7]:
                cell.alignment = align_center
            else:
                cell.alignment = align_left_wrap

    master_widths = {
        "A": 6, "B": 12, "C": 26, "D": 10, "E": 26,
        "F": 16, "G": 10, "H": 45, "I": 45, "J": 32,
        "K": 32, "L": 30, "M": 28, "N": 45, "O": 36
    }
    for col_l, w in master_widths.items():
        ws_master.column_dimensions[col_l].width = w

    # -------------------------------------------------------------
    # SHEET 3: PHIẾU CHẤM ĐIỂM ĐA TIÊU CHÍ (EVALUATION TEMPLATE)
    # -------------------------------------------------------------
    ws_eval = wb.create_sheet(title="Phiếu Chấm Đa Tiêu Chí")
    ws_eval.views.sheetView[0].showGridLines = True
    ws_eval.freeze_panes = "A2"

    eval_headers = [
        "STT", "ID Câu hỏi", "Nhóm Nghiệp vụ", "Tiêu chí Kiểm thử",
        "Câu hỏi Tình huống (Question)", "Đáp án Chuẩn Tham chiếu",
        "Tiêu chí Bắt buộc (Must Include)", "Lỗi Cấm (Must Avoid)",
        "Câu trả lời Thực tế của AI (Prediction)",
        "Điểm Must-Include (0-100)",
        "Vi phạm Lỗi cấm (0=Không, 1=Có)",
        "Điểm Grounding Nguồn (0-100)",
        "Điểm LLM Judge (0-100)",
        "Tổng Điểm Composite (/100)",
        "Kết quả (PASS / FAIL)",
        "Nhận xét & Ghi chú của Giám khảo"
    ]

    for col_num, h_text in enumerate(eval_headers, 1):
        cell = ws_eval.cell(row=1, column=col_num, value=h_text)
        if col_num <= 8:
            cell.fill = navy_fill
        else:
            if col_num == 9:
                cell.fill = blue_fill
            else:
                cell.fill = gold_fill
        cell.font = header_font
        cell.alignment = align_center
        cell.border = thin_border

    for idx, case in enumerate(all_785, 1):
        r_num = idx + 1
        rubric = case.get("rubric", {})
        must_inc = rubric.get("must_include", [])
        must_av = rubric.get("must_avoid", [])
        
        must_inc_str = ""
        for item in must_inc:
            must_inc_str += " • " + item + "\n"
        must_inc_str = must_inc_str.strip()
        
        must_av_str = ""
        for item in must_av:
            must_av_str += " • " + item + "\n"
        must_av_str = must_av_str.strip()
        
        ws_eval.cell(row=r_num, column=1, value=idx).alignment = align_center
        ws_eval.cell(row=r_num, column=2, value=case.get("id", "")).alignment = align_center
        ws_eval.cell(row=r_num, column=3, value=case.get("group_name", "")).alignment = align_left_wrap
        ws_eval.cell(row=r_num, column=4, value=case.get("test_criteria", "")).alignment = align_left_wrap
        ws_eval.cell(row=r_num, column=5, value=case.get("question", "")).alignment = align_left_wrap
        ws_eval.cell(row=r_num, column=6, value=case.get("reference_answer", "")).alignment = align_left_wrap
        ws_eval.cell(row=r_num, column=7, value=must_inc_str).alignment = align_left_wrap
        ws_eval.cell(row=r_num, column=8, value=must_av_str).alignment = align_left_wrap
        
        # Column 9: AI Prediction (Template placeholder)
        ws_eval.cell(row=r_num, column=9, value="[Chưa chạy dự đoán]").alignment = align_left_wrap
        
        # Column 10: Score Must Include
        ws_eval.cell(row=r_num, column=10, value="").alignment = align_center
        # Column 11: Safety Violation (0 = No, 1 = Yes)
        ws_eval.cell(row=r_num, column=11, value=0).alignment = align_center
        # Column 12: Grounding Score (0-100)
        ws_eval.cell(row=r_num, column=12, value="").alignment = align_center
        # Column 13: LLM Judge Score (0-100)
        ws_eval.cell(row=r_num, column=13, value="").alignment = align_center
        
        # Column 14: Formula for Composite Final Score
        # Rule Score = J{r_num}*0.7 + L{r_num}*0.3
        # Penalty: IF K{r_num}=1 then * 0.3
        # Composite = Rule Score * 0.4 + M{r_num} * 0.6
        composite_formula = f"=ROUND(IF(ISBLANK(J{r_num}), 0, ((J{r_num}*0.7 + L{r_num}*0.3) * IF(K{r_num}=1, 0.3, 1.0) * 0.4) + (M{r_num} * 0.6)), 2)"
        cell_comp = ws_eval.cell(row=r_num, column=14, value=composite_formula)
        cell_comp.alignment = align_center
        cell_comp.font = bold_font
        
        # Column 15: Formula for PASS/FAIL
        pass_formula = f'=IF(ISBLANK(J{r_num}), "CHƯA CHẤM", IF(AND(N{r_num}>=80, K{r_num}=0), "PASS", "FAIL"))'
        cell_pass = ws_eval.cell(row=r_num, column=15, value=pass_formula)
        cell_pass.alignment = align_center
        cell_pass.font = bold_font
        
        # Column 16: Evaluator Notes
        ws_eval.cell(row=r_num, column=16, value="").alignment = align_left_wrap
        
        for c in range(1, 17):
            ws_eval.cell(row=r_num, column=c).border = thin_border
            if c not in [14, 15]:
                ws_eval.cell(row=r_num, column=c).font = data_font

    eval_widths = {
        "A": 6, "B": 12, "C": 24, "D": 24, "E": 40,
        "F": 40, "G": 30, "H": 30, "I": 40, "J": 16,
        "K": 16, "L": 16, "M": 16, "N": 16, "O": 14, "P": 30
    }
    for col_l, w in eval_widths.items():
        ws_eval.column_dimensions[col_l].width = w

    # -------------------------------------------------------------
    # SHEETS 4-7: PER GROUP DETAIL SHEETS
    # -------------------------------------------------------------
    group_sheet_names = [
        ("Nghiệp vụ Thẻ (200 câu)", "Folder 2: Nghiệp vụ Thẻ (200 câu)"),
        ("Pháp lý Tuân thủ (200 câu)", "Folder 3: Pháp lý Tuân thủ & KYC (200 câu)"),
        ("App Digital (195 câu)", "Folder 4: App Ngân hàng số A-Bank Plus (195 câu)"),
        ("BH, BCTC, SLA (190 câu)", "Folders 5-6-7: Bảo hiểm, BCTC, SLA (190 câu)")
    ]

    for short_title, full_group_name in group_sheet_names:
        ws_grp = wb.create_sheet(title=short_title)
        ws_grp.views.sheetView[0].showGridLines = True
        ws_grp.freeze_panes = "A2"
        
        for col_num, h_text in enumerate(master_headers, 1):
            cell = ws_grp.cell(row=1, column=col_num, value=h_text)
            cell.fill = navy_fill
            cell.font = header_font
            cell.alignment = align_center
            cell.border = thin_border
            
        grp_cases = []
        for c in all_785:
            if c.get("group_name") == full_group_name:
                grp_cases.append(c)
                
        for g_idx, case in enumerate(grp_cases, 1):
            rubric = case.get("rubric", {})
            must_inc = rubric.get("must_include", [])
            must_av = rubric.get("must_avoid", [])
            
            must_inc_str = ""
            for item in must_inc:
                must_inc_str += " • " + item + "\n"
            must_inc_str = must_inc_str.strip()
            
            must_av_str = ""
            for item in must_av:
                must_av_str += " • " + item + "\n"
            must_av_str = must_av_str.strip()
            
            evidence_list = case.get("evidence", [])
            tgt_list = []
            sec_list = []
            quote_list = []
            src_list = []
            for ev in evidence_list:
                if ev.get("target_artifact"):
                    tgt_list.append(ev.get("target_artifact"))
                if ev.get("section"):
                    sec_list.append(ev.get("section"))
                if ev.get("quote"):
                    quote_list.append(ev.get("quote"))
                if ev.get("source_path"):
                    src_list.append(ev.get("source_path"))
                    
            tgt_str = "\n".join(tgt_list)
            sec_str = "\n".join(sec_list)
            quote_str = "\n".join(quote_list)
            src_str = "\n".join(src_list)
            
            row_data = [
                g_idx,
                case.get("id", ""),
                case.get("group_name", ""),
                case.get("topic_id", ""),
                case.get("test_criteria", ""),
                case.get("case_type", ""),
                case.get("difficulty", ""),
                case.get("question", ""),
                case.get("reference_answer", ""),
                must_inc_str,
                must_av_str,
                tgt_str,
                sec_str,
                quote_str,
                src_str
            ]
            
            for col_num, val in enumerate(row_data, 1):
                cell = ws_grp.cell(row=g_idx + 1, column=col_num, value=val)
                cell.font = data_font
                cell.border = thin_border
                if col_num in [1, 2, 4, 6, 7]:
                    cell.alignment = align_center
                else:
                    cell.alignment = align_left_wrap

        for col_l, w in master_widths.items():
            ws_grp.column_dimensions[col_l].width = w

    out_path = os.path.join(base_dir, "dataset", "A_Bank_golden_master_785.xlsx")
    wb.save(out_path)
    print("Master Golden 785 Excel Workbook saved successfully to: " + out_path)

if __name__ == "__main__":
    build_master_workbook()
