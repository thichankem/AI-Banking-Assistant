"""
Batch Runner: Execute 785 Golden Questions through LangGraph Agent Banking Pipeline.
===================================================================================
Features:
- Incremental execution with JSON state caching (agent_answers_785_cache.json).
- Safe resume capability: seamlessly continues from un-evaluated cases if interrupted.
- Periodically updates Excel workbook with AI Generated Answers.
- 100% compliant with GEMINI.md rules:
    * 0% REGEX LAW (No regex imports or usage)
    * ZERO TERNARY LAW (Explicit if/else only)
    * NO TYPING LIBRARY (Native Python types only)
    * NO __INIT__.PY (Direct module imports)
"""

import os
import sys
import time
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(r"d:\Desktop\GitHub\AI Banking Asistant")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

from agent_banking.agent_pipeline import BankingAgentPipeline

def load_all_785_cases():
    base_dir = r"d:\Desktop\GitHub\AI Banking Asistant"
    group_files = [
        os.path.join(base_dir, "dataset", "A_Bank_golden_the", "a_bank_the_golden.json"),
        os.path.join(base_dir, "dataset", "A_Bank_golden_compliance", "a_bank_compliance_golden.json"),
        os.path.join(base_dir, "dataset", "A_Bank_golden_app", "a_bank_app_golden.json"),
        os.path.join(base_dir, "dataset", "A_Bank_golden_specialized", "a_bank_specialized_golden.json")
    ]

    all_cases = []
    for fpath in group_files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                all_cases.append(item)
    return all_cases

def load_cache(cache_file):
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache_data, cache_file):
    temp_file = cache_file + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)
    if os.path.exists(cache_file):
        os.replace(temp_file, cache_file)
    else:
        os.rename(temp_file, cache_file)

def export_to_excel(all_cases, answers_cache, output_path):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    ws = wb.create_sheet(title="Master 785 Golden (Chưa chấm)")
    ws.views.sheetView[0].showGridLines = True
    ws.freeze_panes = "A2"

    navy_fill = PatternFill(start_color="0F2240", end_color="0F2240", fill_type="solid")
    gold_fill = PatternFill(start_color="B45309", end_color="B45309", fill_type="solid")
    header_font = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")
    data_font = Font(name="Segoe UI", size=9, bold=False, color="1E293B")

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
            agent_ans = answers_cache[case_id].get("answer", "")
        else:
            agent_ans = "[Đang chờ tiến trình AI Banking xử lý...]"

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
        "J": 55
    }
    for col_letter, w in col_widths.items():
        ws.column_dimensions[col_letter].width = w

    try:
        wb.save(output_path)
    except PermissionError:
        # Fallback if primary file is currently open in Excel
        alt_path = output_path.replace(".xlsx", "_agent.xlsx")
        wb.save(alt_path)
        print(f"[Notice] File is locked by Excel. Saved update to alternative path: {alt_path}")

def run_batch():
    base_dir = r"d:\Desktop\GitHub\AI Banking Asistant"
    cache_file = os.path.join(base_dir, "dataset", "agent_answers_785_cache.json")
    excel_path = os.path.join(base_dir, "dataset", "A_Bank_golden_master_785_agent.xlsx")

    all_cases = load_all_785_cases()
    total_cases = len(all_cases)
    print("=" * 70)
    print("AI BANKING ASSISTANT - BATCH AGENT RUNNER")
    print(f"Total Golden Questions: {total_cases}")
    print("=" * 70)

    answers_cache = load_cache(cache_file)
    completed_count = len(answers_cache)
    print(f"Already completed in cache: {completed_count}/{total_cases}")

    # Export initial state to Excel
    export_to_excel(all_cases, answers_cache, excel_path)
    print(f"Initial Excel exported to: {excel_path}")

    # Check if all cases already evaluated
    if completed_count >= total_cases:
        print("All 785 cases have already been processed through Agent Banking!")
        return

    print("\nInitializing BankingAgentPipeline (LangGraph Engine)...")
    pipeline = BankingAgentPipeline()

    start_total_time = time.time()
    processed_in_session = 0

    for idx, case in enumerate(all_cases, 1):
        case_id = case.get("id", f"CASE-{idx:03d}")
        if case_id in answers_cache:
            continue

        question = case.get("question", "")
        print(f"\n[{idx:03d}/{total_cases:03d}] Processing {case_id}...")
        print(f"  Q: {question[:80]}...")

        t0 = time.time()
        try:
            res = pipeline.chat(query=question, session_id=f"eval_{case_id}")
            latency = round(time.time() - t0, 2)
            answer_text = res.get("answer", "")
            citations = res.get("citations", [])
            audit_status = res.get("audit_status", "PASS")

            cit_strings = []
            for c in citations:
                if isinstance(c, dict):
                    cit_strings.append(str(c.get("section", c.get("source", str(c)))))
                else:
                    cit_strings.append(str(c))

            answers_cache[case_id] = {
                "id": case_id,
                "answer": answer_text,
                "latency_sec": latency,
                "citations": cit_strings,
                "audit_status": audit_status,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            print(f"  -> Done in {latency}s | Audit: {audit_status} | Ans: {answer_text[:60]}...")
        except Exception as e:
            latency = round(time.time() - t0, 2)
            print(f"  -> ERROR on {case_id}: {e}")
            answers_cache[case_id] = {
                "id": case_id,
                "answer": f"[Lỗi xử lý pipeline: {e}]",
                "latency_sec": latency,
                "citations": [],
                "audit_status": "ERROR",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

        # Save cache after every single case to ensure no progress is lost
        save_cache(answers_cache, cache_file)
        processed_in_session += 1

        # Periodically refresh Excel every 5 cases
        if processed_in_session % 5 == 0:
            export_to_excel(all_cases, answers_cache, excel_path)
            print(f"  [Auto-Save] Excel updated with {len(answers_cache)}/{total_cases} answers.")

    # Final export
    export_to_excel(all_cases, answers_cache, excel_path)
    total_elapsed = round(time.time() - start_total_time, 2)
    print("\n" + "=" * 70)
    print("ALL 785 CASES PROCESSED SUCCESSFULLY THROUGH AGENT BANKING!")
    print(f"Total time: {total_elapsed}s")
    print(f"Final Excel Workbook: {excel_path}")
    print("=" * 70)

if __name__ == "__main__":
    run_batch()
