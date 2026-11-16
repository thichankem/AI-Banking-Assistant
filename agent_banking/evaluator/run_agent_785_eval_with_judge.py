"""
Batch Runner: Execute 785 Golden Questions through LangGraph Agent Banking
AND Evaluate Each Answer with LLM-as-a-Judge across 3 Core Criteria:
1. Accuracy (0-100): Match against Reference Answer and Must-Include facts
2. Faithfulness (0-100): Groundedness, absence of hallucination
3. Completeness (0-100): Directness, coverage of user query requirements

DESIGN PRINCIPLES:
1. 0% REGEX LAW: No regex imports or functions.
2. ZERO TERNARY: Explicit if/else statements only.
3. NO TYPING LIBRARY: Native types only.
4. NO __INIT__.PY: Direct module imports.
"""

import os
import sys
import time
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(r"d:\Desktop\GitHub\AI Banking Asistant")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

from agent_banking.core.chat import FallbackChatClient
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

def parse_judge_output(content):
    clean_c = content.strip()
    start_idx = clean_c.find("{")
    end_idx = clean_c.rfind("}")
    if start_idx != -1:
        if end_idx != -1:
            json_str = clean_c[start_idx:end_idx+1]
            try:
                data = json.loads(json_str)
                acc = float(data.get("accuracy", 0.0))
                faith = float(data.get("faithfulness", 0.0))
                comp = float(data.get("completeness", 0.0))
                overall = float(data.get("overall_score", 0.0))
                if overall == 0.0:
                    overall = round((acc * 0.4) + (faith * 0.3) + (comp * 0.3), 1)
                verdict = str(data.get("verdict", "FAIL")).upper()
                if "PASS" in verdict:
                    verdict = "PASS"
                else:
                    verdict = "FAIL"
                feedback = str(data.get("feedback", ""))
                return {
                    "accuracy": acc,
                    "faithfulness": faith,
                    "completeness": comp,
                    "overall_score": overall,
                    "verdict": verdict,
                    "feedback": feedback
                }
            except Exception:
                pass
    return {
        "accuracy": 0.0,
        "faithfulness": 0.0,
        "completeness": 0.0,
        "overall_score": 0.0,
        "verdict": "FAIL",
        "feedback": "Lỗi phân tích cú pháp kết quả từ LLM Judge"
    }

def call_llm_judge(client, question, reference_answer, must_include_list, agent_answer):
    must_inc_str = ", ".join(must_include_list)
    prompt = (
        "Bạn là Giám khảo AI Độc lập (LLM-as-a-Judge) chuyên chấm điểm chất lượng câu trả lời của Trợ lý Ngân hàng A-Bank.\n"
        "Hãy chấm điểm câu trả lời thực tế của Trợ lý dựa trên Câu hỏi tình huống, Đáp án chuẩn tham chiếu và các tiêu chí nghiệp vụ.\n\n"
        f"[CÂU HỎI TÌNH HUỐNG]:\n{question}\n\n"
        f"[ĐÁP ÁN CHUẨN THAM CHIẾU]:\n{reference_answer}\n\n"
        f"[CÁC Ý / CON SỐ BẮT BUỘC PHẢI CÓ (MUST INCLUDE)]:\n{must_inc_str}\n\n"
        f"[CÂU TRẢ LỜI CỦA TRỢ LÝ AI ĐƯỢC CHẤM]:\n{agent_answer}\n\n"
        "HÃY ĐÁNH GIÁ THEO 3 TIÊU CHÍ CỐT LÕI (Thang điểm 0 - 100):\n"
        "1. accuracy: Độ chính xác so với đáp án chuẩn và các mốc hạn mức, con số, điều kiện bắt buộc.\n"
        "2. faithfulness: Độ trung thực, bám sát nghiệp vụ ngân hàng, không bịa đặt hoặc suy diễn sai lệch.\n"
        "3. completeness: Độ đầy đủ, giải quyết trực diện và trọn vẹn tình huống của khách hàng.\n\n"
        "TRẢ VỀ DUY NHẤT 1 ĐOẠN JSON KHÔNG CÓ MARKDOWN BỌC NGOÀI THEO CẤU TRÚC:\n"
        "{\n"
        '  "accuracy": <điểm từ 0 đến 100>,\n'
        '  "faithfulness": <điểm từ 0 đến 100>,\n'
        '  "completeness": <điểm từ 0 đến 100>,\n'
        '  "overall_score": <điểm tổng hợp từ 0 đến 100>,\n'
        '  "verdict": "<PASS hoặc FAIL>",\n'
        '  "feedback": "<nhận xét ngắn gọn 1-2 câu>"\n'
        "}\n"
    )
    try:
        resp = client.create(messages=[{"role": "user", "content": prompt}], temperature=0.0)
        content = resp.choices[0].message.content
        return parse_judge_output(content)
    except Exception as e:
        return {
            "accuracy": 0.0,
            "faithfulness": 0.0,
            "completeness": 0.0,
            "overall_score": 0.0,
            "verdict": "FAIL",
            "feedback": f"Lỗi gọi LLM Judge API: {e}"
        }

def export_to_excel(all_cases, answers_cache, scores_cache, output_path):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    ws = wb.create_sheet(title="Master 785 Golden (Chưa chấm)")
    ws.views.sheetView[0].showGridLines = True
    ws.freeze_panes = "A2"

    navy_fill = PatternFill(start_color="0F2240", end_color="0F2240", fill_type="solid")
    blue_fill = PatternFill(start_color="1E40AF", end_color="1E40AF", fill_type="solid")
    gold_fill = PatternFill(start_color="B45309", end_color="B45309", fill_type="solid")
    green_fill = PatternFill(start_color="15803D", end_color="15803D", fill_type="solid")
    purple_fill = PatternFill(start_color="701A75", end_color="701A75", fill_type="solid")

    header_font = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")
    data_font = Font(name="Segoe UI", size=9, bold=False, color="1E293B")
    bold_pass = Font(name="Segoe UI", size=9, bold=True, color="166534")
    bold_fail = Font(name="Segoe UI", size=9, bold=True, color="991B1B")

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
        "Câu trả lời của Trợ lý AI Banking (Agent Answer)",
        "LLM Điểm 1: Độ Chính Xác (Accuracy)",
        "LLM Điểm 2: Độ Trung Thực (Faithfulness)",
        "LLM Điểm 3: Độ Đầy Đủ (Completeness)",
        "Tổng Điểm LLM (/100)",
        "Kết Luận (PASS / FAIL)",
        "Nhận Xét của Giám Khảo AI (LLM Feedback)"
    ]

    for col_idx, h_text in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_idx, value=h_text)
        if col_idx <= 9:
            cell.fill = navy_fill
        else:
            if col_idx == 10:
                cell.fill = blue_fill
            else:
                if col_idx in [11, 12, 13, 14]:
                    cell.fill = gold_fill
                else:
                    if col_idx == 15:
                        cell.fill = green_fill
                    else:
                        cell.fill = purple_fill
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

        # Agent answer
        agent_ans = ""
        if case_id in answers_cache:
            agent_ans = answers_cache[case_id].get("answer", "")
        else:
            agent_ans = "[Đang chờ tiến trình AI Banking xử lý...]"

        # LLM Judge Scores
        score_acc = ""
        score_faith = ""
        score_comp = ""
        score_overall = ""
        verdict = ""
        feedback = ""

        if case_id in scores_cache:
            sc = scores_cache[case_id]
            score_acc = sc.get("accuracy", 0.0)
            score_faith = sc.get("faithfulness", 0.0)
            score_comp = sc.get("completeness", 0.0)
            score_overall = sc.get("overall_score", 0.0)
            verdict = sc.get("verdict", "FAIL")
            feedback = sc.get("feedback", "")
        else:
            if case_id in answers_cache:
                score_acc = "[Chờ chấm]"
                score_faith = "[Chờ chấm]"
                score_comp = "[Chờ chấm]"
                score_overall = "[Chờ chấm]"
                verdict = "CHỜ CHẤM"
                feedback = "Đang xếp hàng gọi LLM Judge"

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
            agent_ans,
            score_acc,
            score_faith,
            score_comp,
            score_overall,
            verdict,
            feedback
        ]

        for col_idx, val in enumerate(row_vals, 1):
            cell = ws.cell(row=r_num, column=col_idx, value=val)
            cell.font = data_font
            cell.border = thin_border
            if col_idx in [1, 2, 3, 5, 11, 12, 13, 14, 15]:
                cell.alignment = align_center
                if col_idx == 15:
                    if str(val) == "PASS":
                        cell.font = bold_pass
                    else:
                        if str(val) == "FAIL":
                            cell.font = bold_fail
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
        "J": 55,
        "K": 18,
        "L": 18,
        "M": 18,
        "N": 16,
        "O": 14,
        "P": 45
    }
    for col_letter, w in col_widths.items():
        ws.column_dimensions[col_letter].width = w

    try:
        wb.save(output_path)
    except PermissionError:
        alt_path = output_path.replace(".xlsx", "_agent.xlsx")
        wb.save(alt_path)

def run_batch_and_judge():
    base_dir = r"d:\Desktop\GitHub\AI Banking Asistant"
    answers_cache_file = os.path.join(base_dir, "dataset", "agent_answers_785_cache.json")
    scores_cache_file = os.path.join(base_dir, "dataset", "agent_eval_scores_785_cache.json")
    excel_path = os.path.join(base_dir, "dataset", "A_Bank_golden_master_785_agent.xlsx")

    all_cases = load_all_785_cases()
    total_cases = len(all_cases)
    print("=" * 70)
    print("AI BANKING ASSISTANT - AGENT PIPELINE & LLM-AS-A-JUDGE RUNNER")
    print(f"Total Golden Questions: {total_cases}")
    print("=" * 70)

    answers_cache = load_cache(answers_cache_file)
    scores_cache = load_cache(scores_cache_file)

    print(f"Loaded existing answers: {len(answers_cache)}/{total_cases}")
    print(f"Loaded existing judge scores: {len(scores_cache)}/{total_cases}")

    # Initialize LLM Judge Client
    print("\n[1/2] Initializing LLM-as-a-Judge Client...")
    judge_client = FallbackChatClient(module="generator")

    # PHASE 1: Judge all existing answers that haven't been scored yet
    cases_by_id = {}
    for c in all_cases:
        cases_by_id[c.get("id")] = c

    unscored_ids = []
    for cid, ans_obj in answers_cache.items():
        if cid not in scores_cache:
            unscored_ids.append(cid)

    if len(unscored_ids) > 0:
        print(f"\n[Judge Backlog] Scoring {len(unscored_ids)} existing answers with LLM Judge...")
        for j_idx, cid in enumerate(unscored_ids, 1):
            case_data = cases_by_id.get(cid, {})
            q = case_data.get("question", "")
            ref_ans = case_data.get("reference_answer", "")
            must_inc = case_data.get("rubric", {}).get("must_include", [])
            ans_text = answers_cache[cid].get("answer", "")

            print(f"  [{j_idx:03d}/{len(unscored_ids):03d}] Judging {cid}...")
            t0 = time.time()
            score_res = call_llm_judge(judge_client, q, ref_ans, must_inc, ans_text)
            elapsed = round(time.time() - t0, 2)
            scores_cache[cid] = score_res
            save_cache(scores_cache, scores_cache_file)
            print(f"    -> Acc: {score_res['accuracy']} | Faith: {score_res['faithfulness']} | Comp: {score_res['completeness']} | Verdict: {score_res['verdict']} ({elapsed}s)")

            if j_idx % 5 == 0:
                export_to_excel(all_cases, answers_cache, scores_cache, excel_path)
                print(f"    [Auto-Save] Excel updated with {len(scores_cache)} scores.")

        export_to_excel(all_cases, answers_cache, scores_cache, excel_path)

    # Export intermediate state
    export_to_excel(all_cases, answers_cache, scores_cache, excel_path)

    # PHASE 2: Process remaining questions through Agent Banking + LLM Judge
    remaining_cases = []
    for c in all_cases:
        cid = c.get("id")
        if cid not in answers_cache:
            remaining_cases.append(c)

    if len(remaining_cases) == 0:
        print("\nAll 785 cases have answers and scores! Exporting final workbook...")
        export_to_excel(all_cases, answers_cache, scores_cache, excel_path)
        return

    print(f"\n[2/2] Initializing BankingAgentPipeline for {len(remaining_cases)} remaining cases...")
    pipeline = BankingAgentPipeline()

    for idx, case in enumerate(remaining_cases, 1):
        cid = case.get("id", "")
        q = case.get("question", "")
        ref_ans = case.get("reference_answer", "")
        must_inc = case.get("rubric", {}).get("must_include", [])

        print(f"\n[Batch {idx:03d}/{len(remaining_cases):03d}] Running {cid} through Agent Banking...")
        t0 = time.time()
        try:
            res = pipeline.chat(query=q, session_id=f"eval_{cid}")
            latency = round(time.time() - t0, 2)
            answer_text = res.get("answer", "")
            citations = res.get("citations", [])
            audit_status = res.get("audit_status", "PASS")

            cit_strings = []
            for c_item in citations:
                if isinstance(c_item, dict):
                    cit_strings.append(str(c_item.get("section", c_item.get("source", str(c_item)))))
                else:
                    cit_strings.append(str(c_item))

            answers_cache[cid] = {
                "id": cid,
                "answer": answer_text,
                "latency_sec": latency,
                "citations": cit_strings,
                "audit_status": audit_status,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            save_cache(answers_cache, answers_cache_file)
            print(f"  -> Agent done in {latency}s | Audit: {audit_status}")

            # Immediately judge with LLM Judge
            print(f"  -> Judging {cid} with LLM Judge...")
            tj0 = time.time()
            score_res = call_llm_judge(judge_client, q, ref_ans, must_inc, answer_text)
            j_elapsed = round(time.time() - tj0, 2)
            scores_cache[cid] = score_res
            save_cache(scores_cache, scores_cache_file)
            print(f"    -> Acc: {score_res['accuracy']} | Faith: {score_res['faithfulness']} | Comp: {score_res['completeness']} | Verdict: {score_res['verdict']} ({j_elapsed}s)")

        except Exception as e:
            latency = round(time.time() - t0, 2)
            print(f"  -> ERROR on {cid}: {e}")
            answers_cache[cid] = {
                "id": cid,
                "answer": f"[Lỗi xử lý pipeline: {e}]",
                "latency_sec": latency,
                "citations": [],
                "audit_status": "ERROR",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            save_cache(answers_cache, answers_cache_file)

        if idx % 3 == 0:
            export_to_excel(all_cases, answers_cache, scores_cache, excel_path)
            print(f"  [Auto-Save] Excel synchronized (Answers: {len(answers_cache)}, Scores: {len(scores_cache)})")

    # Final export
    export_to_excel(all_cases, answers_cache, scores_cache, excel_path)
    print("\n" + "=" * 70)
    print("ALL 785 CASES FULLY PROCESSED AND JUDGED BY LLM!")
    print(f"Final Excel Workbook: {excel_path}")
    print("=" * 70)

if __name__ == "__main__":
    run_batch_and_judge()
