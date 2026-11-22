"""
Script to generate Master Golden Excel Workbook for A-Bank 785 cases.
Includes:
- Sheet 1: Dashboard & Multi-Criteria Evaluation Framework
- Sheet 2: Master 785 Golden Dataset (Unscored / Original)
- Sheet 3: Multi-Criteria Evaluation Sheet (With built-in formulas and score rubrics)
- Sheets 4-7: Individual folders

DESIGN PRINCIPLES:
1. 0% REGEX LAW: No regex imports or functions.
2. ZERO TERNARY: Explicit if/else statements only.
4. NO __INIT__.PY: Direct module imports.
"""

import json
from pathlib import Path
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

from agent_banking.evaluator.config.settings import (
    EvaluatorConfig,
    get_evaluator_messages,
    resolve_repo_path,
)

_OUT_CFG = EvaluatorConfig.CACHES_AND_OUTPUTS
_GE_CFG = EvaluatorConfig.GOLDEN_EVALUATOR
_EXCEL_CFG = EvaluatorConfig.EXCEL
_FEEDBACK = get_evaluator_messages("feedback")
_EXCEL_LABELS = get_evaluator_messages("excel")


def build_master_workbook():
    base_dir = Path(__file__).resolve().parents[2]
    groups_info = list(_EXCEL_LABELS["groups_info"])

    all_785 = []
    group_stats = {}

    for grp in groups_info:
        gname = grp["group_name"]
        fpath = resolve_repo_path(grp["relpath"], root=base_dir)
        with open(fpath, "r", encoding="utf-8") as f:
            cases = json.load(f)
            for c in cases:
                c["group_name"] = gname
                all_785.append(c)
            group_stats[gname] = grp

    total_expected = float(EvaluatorConfig.EXPECTED_TOTAL_CASES)
    styles = _EXCEL_CFG["styles"]
    sheets = _EXCEL_CFG["sheets"]

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    # Styles
    navy_fill = PatternFill(start_color=styles["navy_color"], end_color=styles["navy_color"], fill_type="solid")
    gold_fill = PatternFill(start_color=styles["gold_color"], end_color=styles["gold_color"], fill_type="solid")
    blue_fill = PatternFill(start_color=styles["blue_color"], end_color=styles["blue_color"], fill_type="solid")
    total_fill = PatternFill(start_color=styles["total_fill_color"], end_color=styles["total_fill_color"], fill_type="solid")

    font_name = str(styles["font_name"])
    header_font = Font(name=font_name, size=int(styles["header_font_size"]), bold=True, color="FFFFFF")
    data_font = Font(name=font_name, size=int(styles["data_font_size"]), bold=False, color=styles["data_text_color"])
    bold_font = Font(name=font_name, size=int(styles["data_font_size"]), bold=True, color=styles["navy_color"])
    title_font = Font(name=font_name, size=int(styles["title_font_size"]), bold=True, color=styles["navy_color"])
    subtitle_font = Font(name=font_name, size=int(styles["header_font_size"]), italic=True, color=styles["subtitle_color"])

    border_color = str(styles["border_color"])
    thin_border = Border(
        left=Side(style="thin", color=border_color),
        right=Side(style="thin", color=border_color),
        top=Side(style="thin", color=border_color),
        bottom=Side(style="thin", color=border_color)
    )

    align_left_wrap = Alignment(horizontal="left", vertical="top", wrap_text=True)
    align_center = Alignment(horizontal="center", vertical="top", wrap_text=True)

    # -------------------------------------------------------------
    # SHEET 1: DASHBOARD & EVALUATION FRAMEWORK
    # -------------------------------------------------------------
    ws_dash = wb.create_sheet(title=str(sheets["dashboard_sheet_title"]))
    ws_dash.views.sheetView[0].showGridLines = True

    ws_dash.cell(row=2, column=2, value=str(_EXCEL_LABELS["dashboard_title"])).font = title_font
    ws_dash.cell(row=3, column=2, value=str(_EXCEL_LABELS["dashboard_subtitle"])).font = subtitle_font

    # Table 1
    ws_dash.cell(row=5, column=2, value=str(_EXCEL_LABELS["section_1_title"])).font = bold_font
    tbl1_headers = list(_EXCEL_LABELS["tbl1_headers"])
    for c_idx, h in enumerate(tbl1_headers, start=2):
        cell = ws_dash.cell(row=6, column=c_idx, value=h)
        cell.fill = navy_fill
        cell.font = header_font
        cell.alignment = align_center

    r_idx = 7
    stt = 1
    for gname, grp in group_stats.items():
        count = 0
        for item in all_785:
            if item.get("group_name") == gname:
                count += 1
        ws_dash.cell(row=r_idx, column=2, value=stt).alignment = align_center
        ws_dash.cell(row=r_idx, column=3, value=gname).alignment = align_left_wrap
        ws_dash.cell(row=r_idx, column=4, value=count).alignment = align_center
        rate_str = str(round((count / total_expected) * 100.0, 1)) + "%"
        ws_dash.cell(row=r_idx, column=5, value=rate_str).alignment = align_center
        ws_dash.cell(row=r_idx, column=6, value="a_bank_" + str(grp["group_code"]) + "_golden").alignment = align_center

        for c in range(2, 7):
            ws_dash.cell(row=r_idx, column=c).border = thin_border
            ws_dash.cell(row=r_idx, column=c).font = data_font
        r_idx += 1
        stt += 1

    # Total row
    ws_dash.cell(row=r_idx, column=2, value=str(_EXCEL_LABELS["total_label"])).alignment = align_center
    ws_dash.cell(row=r_idx, column=3, value=str(_EXCEL_LABELS["total_group_desc"])).alignment = align_left_wrap
    ws_dash.cell(row=r_idx, column=4, value=int(total_expected)).alignment = align_center
    ws_dash.cell(row=r_idx, column=5, value="100.0%").alignment = align_center
    ws_dash.cell(row=r_idx, column=6, value=str(_EXCEL_LABELS["total_cases_label"])).alignment = align_center
    for c in range(2, 7):
        ws_dash.cell(row=r_idx, column=c).fill = total_fill
        ws_dash.cell(row=r_idx, column=c).font = bold_font
        ws_dash.cell(row=r_idx, column=c).border = thin_border

    # Table 2
    r_idx += 3
    ws_dash.cell(row=r_idx, column=2, value=str(_EXCEL_LABELS["section_2_title"])).font = bold_font
    r_idx += 1
    tbl2_headers = list(_EXCEL_LABELS["tbl2_headers"])
    for c_idx, h in enumerate(tbl2_headers, start=2):
        cell = ws_dash.cell(row=r_idx, column=c_idx, value=h)
        cell.fill = navy_fill
        cell.font = header_font
        cell.alignment = align_center

    default_crit = str(_EXCEL_LABELS["default_criteria_label"])
    crit_counts = {}
    for c in all_785:
        k = c.get("test_criteria", default_crit)
        crit_counts[k] = crit_counts.get(k, 0) + 1

    crit_desc = dict(_EXCEL_LABELS["criteria_descriptions"])

    r_idx += 1
    stt = 1
    for crit, count in crit_counts.items():
        ws_dash.cell(row=r_idx, column=2, value=stt).alignment = align_center
        ws_dash.cell(row=r_idx, column=3, value=crit).alignment = align_left_wrap
        ws_dash.cell(row=r_idx, column=4, value=count).alignment = align_center
        crit_rate_str = str(round((count / total_expected) * 100.0, 1)) + "%"
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

    # Table 3
    r_idx += 3
    ws_dash.cell(row=r_idx, column=2, value=str(_EXCEL_LABELS["section_3_title"])).font = bold_font
    r_idx += 1
    tbl3_headers = list(_EXCEL_LABELS["tbl3_headers"])
    for c_idx, h in enumerate(tbl3_headers, start=2):
        cell = ws_dash.cell(row=r_idx, column=c_idx, value=h)
        cell.fill = gold_fill
        cell.font = header_font
        cell.alignment = align_center

    eval_rules = list(_EXCEL_LABELS["eval_rules"])

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

    for col_l, w in _EXCEL_CFG["column_widths"]["dashboard"].items():
        ws_dash.column_dimensions[col_l].width = w

    # -------------------------------------------------------------
    # SHEET 2: MASTER 785 GOLDEN QUESTIONS
    # -------------------------------------------------------------
    ws_master = wb.create_sheet(title=str(sheets["master_sheet_title"]))
    ws_master.views.sheetView[0].showGridLines = True
    ws_master.freeze_panes = str(styles["freeze_cell"])

    master_headers = list(_EXCEL_LABELS["master_headers"])

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

    master_widths = _EXCEL_CFG["column_widths"]["master_dataset"]
    for col_l, w in master_widths.items():
        ws_master.column_dimensions[col_l].width = w

    # -------------------------------------------------------------
    # SHEET 3: EVALUATION TEMPLATE
    # -------------------------------------------------------------
    ws_eval = wb.create_sheet(title=str(sheets["eval_sheet_title"]))
    ws_eval.views.sheetView[0].showGridLines = True
    ws_eval.freeze_panes = str(styles["freeze_cell"])

    eval_headers = list(_EXCEL_LABELS["eval_headers"])

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

    w_must = round(float(_GE_CFG["must_include_weight"]) / 100.0, 4)
    w_ground = round(float(_GE_CFG["grounding_weight"]) / 100.0, 4)
    safety_mult = float(_GE_CFG["safety_violation_multiplier"])
    w_rule = float(_GE_CFG["rule_score_weight"])
    w_llm = float(_GE_CFG["llm_score_weight"])
    pass_thresh = int(float(_GE_CFG["pass_threshold"]))

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

        ws_eval.cell(row=r_num, column=9, value=str(_FEEDBACK["pending_prediction_placeholder"])).alignment = align_left_wrap
        ws_eval.cell(row=r_num, column=10, value="").alignment = align_center
        ws_eval.cell(row=r_num, column=11, value=0).alignment = align_center
        ws_eval.cell(row=r_num, column=12, value="").alignment = align_center
        ws_eval.cell(row=r_num, column=13, value="").alignment = align_center

        composite_formula = (
            f"=ROUND(IF(ISBLANK(J{r_num}), 0, "
            f"((J{r_num}*{w_must} + L{r_num}*{w_ground}) * IF(K{r_num}=1, {safety_mult}, 1.0) * {w_rule}) "
            f"+ (M{r_num} * {w_llm})), 2)"
        )
        cell_comp = ws_eval.cell(row=r_num, column=14, value=composite_formula)
        cell_comp.alignment = align_center
        cell_comp.font = bold_font

        pending_label = str(_FEEDBACK["pending_judge_verdict"])
        pass_formula = f'=IF(ISBLANK(J{r_num}), "{pending_label}", IF(AND(N{r_num}>={pass_thresh}, K{r_num}=0), "PASS", "FAIL"))'
        cell_pass = ws_eval.cell(row=r_num, column=15, value=pass_formula)
        cell_pass.alignment = align_center
        cell_pass.font = bold_font

        ws_eval.cell(row=r_num, column=16, value="").alignment = align_left_wrap

        for c in range(1, 17):
            ws_eval.cell(row=r_num, column=c).border = thin_border
            if c not in [14, 15]:
                ws_eval.cell(row=r_num, column=c).font = data_font

    for col_l, w in _EXCEL_CFG["column_widths"]["eval_template"].items():
        ws_eval.column_dimensions[col_l].width = w

    # -------------------------------------------------------------
    # SHEETS 4-7: PER GROUP DETAIL SHEETS
    # -------------------------------------------------------------
    for grp in groups_info:
        short_title = grp["short_title"]
        full_group_name = grp["group_name"]
        ws_grp = wb.create_sheet(title=short_title)
        ws_grp.views.sheetView[0].showGridLines = True
        ws_grp.freeze_panes = str(styles["freeze_cell"])

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

    out_path = str(resolve_repo_path(_OUT_CFG["master_excel_relpath"], root=base_dir))
    wb.save(out_path)
    print("Master Golden 785 Excel Workbook saved successfully to: " + out_path)


if __name__ == "__main__":
    build_master_workbook()
