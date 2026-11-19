"""
Script to format A_Bank_golden_master_785.xlsx to contain ONLY ONE sheet:
'Master 785 Golden (Chưa chấm)'

MANDATORY CODING RULES:
1. 0% REGEX LAW
2. ZERO TERNARY LAW (Explicit if/else)
3. NO TYPING LIBRARY
4. NO __INIT__.PY
"""

import os
import json
from pathlib import Path
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

from agent_banking.evaluator.config.settings import (
    EvaluatorConfig,
    get_evaluator_messages,
    resolve_repo_path,
)

_DS_CFG = EvaluatorConfig.DATASETS
_OUT_CFG = EvaluatorConfig.CACHES_AND_OUTPUTS
_EXCEL_CFG = EvaluatorConfig.EXCEL
_FEEDBACK = get_evaluator_messages("feedback")
_EXCEL_LABELS = get_evaluator_messages("excel")


def format_excel():
    base_dir = Path(__file__).resolve().parents[2]
    template = _DS_CFG["group_path_template"]

    all_cases = []
    for group in EvaluatorConfig.DATASET_GROUPS:
        fpath = resolve_repo_path(template.replace("{group}", group), root=base_dir)
        with open(fpath, "r", encoding="utf-8") as f:
            cases = json.load(f)
            for c in cases:
                all_cases.append(c)

    styles = _EXCEL_CFG["styles"]
    sheets = _EXCEL_CFG["sheets"]

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    ws = wb.create_sheet(title=str(sheets["batch_sheet_title"]))
    ws.views.sheetView[0].showGridLines = True
    ws.freeze_panes = str(styles["freeze_cell"])

    navy_fill = PatternFill(start_color=styles["navy_color"], end_color=styles["navy_color"], fill_type="solid")
    gold_fill = PatternFill(start_color=styles["gold_color"], end_color=styles["gold_color"], fill_type="solid")
    font_name = str(styles["font_name"])
    header_font = Font(name=font_name, size=int(styles["header_font_size"]), bold=True, color="FFFFFF")
    data_font = Font(name=font_name, size=int(styles["data_font_size"]), bold=False, color=styles["data_text_color"])

    border_color = str(styles["border_color"])
    thin_border = Border(
        left=Side(style="thin", color=border_color),
        right=Side(style="thin", color=border_color),
        top=Side(style="thin", color=border_color),
        bottom=Side(style="thin", color=border_color)
    )

    align_left_wrap = Alignment(horizontal="left", vertical="top", wrap_text=True)
    align_center = Alignment(horizontal="center", vertical="top", wrap_text=True)

    headers = list(_EXCEL_LABELS["batch_headers"])

    for col_idx, h_text in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_idx, value=h_text)
        if col_idx == 10:
            cell.fill = gold_fill
        else:
            cell.fill = navy_fill
        cell.font = header_font
        cell.alignment = align_center
        cell.border = thin_border

    answers_cache = {}
    cache_path = resolve_repo_path(_OUT_CFG["answers_legacy_cache_relpath"], root=base_dir)
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

        if case_id in answers_cache:
            agent_ans = answers_cache[case_id]
        else:
            agent_ans = str(_FEEDBACK["pending_batch_placeholder"])

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

    col_widths = _EXCEL_CFG["column_widths"]["single_sheet_10_cols"]
    for col_letter, w in col_widths.items():
        ws.column_dimensions[col_letter].width = w

    out_file = str(resolve_repo_path(_OUT_CFG["master_excel_relpath"], root=base_dir))
    wb.save(out_file)
    print("Formatted Excel Workbook saved to: " + out_file)
    print("Total rows:", len(all_cases) + 1)


if __name__ == "__main__":
    format_excel()
