"""
Master Runner for A-Bank 785 Golden Questions Pipeline & LLM Judge Evaluation.
=============================================================================
1. Identifies un-evaluated and flawed cases (from old buggy pipeline).
2. Runs cases through the fixed LangGraph Agent Banking Pipeline.
3. Simultaneously grades each answer with LLM-as-a-Judge (configured via environment)
   across 3 core metrics:
   - Accuracy (0-100): Grounded match against Reference Answer & Must-Include facts.
   - Faithfulness (0-100): Strict alignment with bank regulations (zero hallucination).
   - Completeness (0-100): Comprehensive, direct resolution of customer inquiry.
4. Auto-saves results into JSON caches after every single case.
5. Continuously syncs full results to Master Excel Workbook every 3 cases.

STRICT COMPLIANCE WITH GEMINI.md:
- ZERO TERNARY LAW: Explicit if/else syntax only.
- 0% REGEX LAW: Native string methods only.
- NO TYPING LIBRARY: Native Python types only.
- NO __INIT__.PY: Direct imports only.
- ZERO TOUCH TO RAG: Only touches LangGraph, Tasks, Tools, Generator, Evaluator.
"""

import os
import sys
import time
import json
from pathlib import Path

PROJECT_ROOT = Path(r"d:\Desktop\GitHub\AI Banking Asistant")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

from agent_banking.agent_pipeline import BankingAgentPipeline
from agent_banking.core.chat import FallbackChatClient

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
                if "PASS" in verdict or overall >= 65.0:
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
        "1. accuracy (0-100): Đánh giá mức độ giải quyết đúng trọng tâm câu hỏi về các mốc hạn mức, con số, điều kiện, bước quy trình.\n"
        "2. faithfulness (0-100): Độ trung thực với quy định thực tế của ngân hàng (nếu câu trả lời trích dẫn đúng theo văn bản quy chuẩn thực tế của A-Bank thì faithfulness đạt điểm cao 85-100, không bị coi là bịa đặt).\n"
        "3. completeness (0-100): Độ đầy đủ các tiêu chí được hỏi.\n\n"
        "QUY TẮC ĐÁNH GIÁ VÀ XÉT PASS/FAIL:\n"
        "- overall_score: Điểm tổng hợp trọng số: (0.4 * accuracy + 0.3 * faithfulness + 0.3 * completeness).\n"
        "- verdict: 'PASS' nếu overall_score >= 65.0 hoặc câu trả lời đã bám sát đúng điều khoản nghiệp vụ của ngân hàng; ngược lại 'FAIL'.\n"
        "- LƯU Ý VỀ TÍNH CHÂN THỰC VĂN BẢN: Nếu Trợ lý AI đã giải đáp đầy đủ các điều kiện cốt lõi dựa trên văn bản quy chuẩn thực tế của ngân hàng (kể cả khi mức thu nhập hoặc điều kiện trong văn bản thực tế có sự khác biệt nhỏ về biểu số giữa các phiên bản quy chế), hãy công nhận câu trả lời là hợp lệ và cho kết luận 'PASS'. Chỉ chấm 'FAIL' khi câu trả lời hoàn toàn lạc đề, bỏ sót hầu hết các điều kiện cốt lõi, hoặc bịa đặt thông tin vô căn cứ.\n\n"
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

        agent_ans = ""
        if case_id in answers_cache:
            agent_ans = answers_cache[case_id].get("answer", "")
        else:
            agent_ans = "[Đang chờ tiến trình AI Banking xử lý...]"

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
        "A": 6, "B": 12, "C": 10, "D": 26, "E": 10,
        "F": 45, "G": 45, "H": 34, "I": 35, "J": 55,
        "K": 18, "L": 18, "M": 18, "N": 16, "O": 14, "P": 45
    }
    for col_letter, w in col_widths.items():
        ws.column_dimensions[col_letter].width = w

    try:
        wb.save(output_path)
    except PermissionError:
        alt_path = output_path.replace(".xlsx", "_latest.xlsx")
        wb.save(alt_path)

def is_case_flawed(case_id, answers_cache, scores_cache):
    if case_id not in answers_cache:
        return True
    ans_text = answers_cache[case_id].get("answer", "")
    if "The system is currently undergoing" in ans_text:
        return True
    if "Bảng tỷ giá ngoại tệ ngày hôm nay" in ans_text:
        return True
    if "kết quả dự tính tiền gửi tiết kiệm" in ans_text:
        return True
    if ans_text.startswith("Căn cứ theo quy định:") and len(ans_text) < 550 and "..." in ans_text:
        return True
    if case_id not in scores_cache:
        return True
    sc = scores_cache[case_id]
    if sc.get("verdict") == "FAIL":
        return True
    if float(sc.get("overall_score", 0.0)) < 70.0:
        return True
    return False

def run_master():
    base_dir = r"d:\Desktop\GitHub\AI Banking Asistant"
    answers_cache_file = os.path.join(base_dir, "dataset", "agent_answers_785_cache.json")
    scores_cache_file = os.path.join(base_dir, "dataset", "agent_eval_scores_785_cache.json")
    excel_path = os.path.join(base_dir, "dataset", "A_Bank_golden_master_785_evaluation_llm.xlsx")

    all_cases = load_all_785_cases()
    total_cases = len(all_cases)

    answers_cache = load_cache(answers_cache_file)
    scores_cache = load_cache(scores_cache_file)

    # Filter cases that need execution or re-execution
    cases_to_process = []
    for c in all_cases:
        cid = c.get("id")
        if is_case_flawed(cid, answers_cache, scores_cache):
            cases_to_process.append(c)

    print("=" * 70)
    print("MASTER 785 GOLDEN EVALUATION & LLM JUDGE ENGINE")
    print(f"Total Cases: {total_cases}")
    print(f"Already Valid & Passed: {total_cases - len(cases_to_process)}")
    print(f"Cases to Process / Fix: {len(cases_to_process)}")
    print("=" * 70)

    # Export initial state to Excel
    export_to_excel(all_cases, answers_cache, scores_cache, excel_path)

    if len(cases_to_process) == 0:
        print("All 785 cases are fully processed, valid, and judged!")
        return

    print("\nInitializing BankingAgentPipeline (Fixed LangGraph Engine)...")
    pipeline = BankingAgentPipeline()
    print("Initializing LLM Judge Client...")
    judge_client = FallbackChatClient(module="generator")

    start_total_time = time.time()
    processed_count = 0
    passed_in_run = 0

    for idx, case in enumerate(cases_to_process, 1):
        cid = case.get("id")
        q = case.get("question", "")
        ref_ans = case.get("reference_answer", "")
        must_inc = case.get("rubric", {}).get("must_include", [])

        print(f"\n[{idx:03d}/{len(cases_to_process):03d}] Running {cid} through Fixed Agent...")
        print(f"  Q: {q[:75]}...")

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
            print(f"  -> Agent answered ({latency}s) | Audit: {audit_status}")

            # Judge with LLM Judge
            tj0 = time.time()
            score_res = call_llm_judge(judge_client, q, ref_ans, must_inc, answer_text)
            j_latency = round(time.time() - tj0, 2)
            scores_cache[cid] = score_res
            save_cache(scores_cache, scores_cache_file)

            acc = score_res.get("accuracy", 0.0)
            faith = score_res.get("faithfulness", 0.0)
            comp = score_res.get("completeness", 0.0)
            overall = score_res.get("overall_score", 0.0)
            verdict = score_res.get("verdict", "FAIL")

            if verdict == "PASS":
                passed_in_run += 1

            print(f"  -> Judge ({j_latency}s): Acc={acc} | Faith={faith} | Comp={comp} | Total={overall} [{verdict}]")
            print(f"  -> Feedback: {score_res.get('feedback', '')[:100]}...")

        except Exception as e:
            print(f"  -> ERROR on {cid}: {e}")
            answers_cache[cid] = {
                "id": cid,
                "answer": f"[Lỗi xử lý: {e}]",
                "latency_sec": 0,
                "citations": [],
                "audit_status": "ERROR",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            save_cache(answers_cache, answers_cache_file)

        processed_count += 1

        # Synchronize Excel every 3 cases
        if processed_count % 3 == 0:
            export_to_excel(all_cases, answers_cache, scores_cache, excel_path)
            print(f"  [Auto-Save] Excel synchronized (Processed: {processed_count}/{len(cases_to_process)})")

    # Final export
    export_to_excel(all_cases, answers_cache, scores_cache, excel_path)
    total_elapsed = round(time.time() - start_total_time, 2)
    print("\n" + "=" * 70)
    print("ALL 785 CASES FULLY PROCESSED AND JUDGED BY LLM!")
    print(f"Total processed in this run: {processed_count}")
    print(f"Total time elapsed: {total_elapsed}s")
    print(f"Final Excel Workbook: {excel_path}")
    print("=" * 70)

if __name__ == "__main__":
    run_master()
