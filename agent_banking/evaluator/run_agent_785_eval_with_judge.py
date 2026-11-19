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
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

from agent_banking.core.chat import FallbackChatClient
from agent_banking.agent_pipeline import BankingAgentPipeline
from agent_banking.evaluator.config.settings import (
    EvaluatorConfig,
    get_evaluator_messages,
    get_evaluator_prompt,
    render_prompt,
    resolve_repo_path,
)

_DS_CFG = EvaluatorConfig.DATASETS
_OUT_CFG = EvaluatorConfig.CACHES_AND_OUTPUTS
_J785 = EvaluatorConfig.JUDGE_785
_EXCEL_CFG = EvaluatorConfig.EXCEL
_FEEDBACK = get_evaluator_messages("feedback")
_EXCEL_LABELS = get_evaluator_messages("excel")
_JUDGE_PROMPT = get_evaluator_prompt("AGENT_785_JUDGE_PROMPT", "agent_785_judge.md")


def load_all_785_cases():
    template = _DS_CFG["group_path_template"]
    all_cases = []
    for group in EvaluatorConfig.DATASET_GROUPS:
        fpath = resolve_repo_path(template.replace("{group}", group), root=PROJECT_ROOT)
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
            json_str = clean_c[start_idx:end_idx + 1]
            try:
                data = json.loads(json_str)
                acc = float(data.get("accuracy", 0.0))
                faith = float(data.get("faithfulness", 0.0))
                comp = float(data.get("completeness", 0.0))
                overall = float(data.get("overall_score", 0.0))
                if overall == 0.0:
                    overall = round(
                        (acc * float(_J785["accuracy_weight"]))
                        + (faith * float(_J785["faithfulness_weight"]))
                        + (comp * float(_J785["completeness_weight"])),
                        1,
                    )
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
        "feedback": _FEEDBACK["judge_parse_error"]
    }


def call_llm_judge(client, question, reference_answer, must_include_list, agent_answer):
    must_inc_str = ", ".join(must_include_list)
    prompt = render_prompt(
        _JUDGE_PROMPT,
        {
            "question": question,
            "reference_answer": reference_answer,
            "must_include": must_inc_str,
            "agent_answer": agent_answer,
        },
    )
    try:
        resp = client.create(
            messages=[{"role": "user", "content": prompt}],
            temperature=float(_J785["judge_temperature"]),
        )
        content = resp.choices[0].message.content
        return parse_judge_output(content)
    except Exception as e:
        return {
            "accuracy": 0.0,
            "faithfulness": 0.0,
            "completeness": 0.0,
            "overall_score": 0.0,
            "verdict": "FAIL",
            "feedback": _FEEDBACK["judge_api_error_prefix"] + str(e)
        }


def export_to_excel(all_cases, answers_cache, scores_cache, output_path):
    styles = _EXCEL_CFG["styles"]
    sheets = _EXCEL_CFG["sheets"]

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    ws = wb.create_sheet(title=str(sheets["batch_sheet_title"]))
    ws.views.sheetView[0].showGridLines = True
    ws.freeze_panes = str(styles["freeze_cell"])

    navy_fill = PatternFill(start_color=styles["navy_color"], end_color=styles["navy_color"], fill_type="solid")
    blue_fill = PatternFill(start_color=styles["blue_color"], end_color=styles["blue_color"], fill_type="solid")
    gold_fill = PatternFill(start_color=styles["gold_color"], end_color=styles["gold_color"], fill_type="solid")
    green_fill = PatternFill(start_color=styles["green_color"], end_color=styles["green_color"], fill_type="solid")
    purple_fill = PatternFill(start_color=styles["purple_color"], end_color=styles["purple_color"], fill_type="solid")

    font_name = str(styles["font_name"])
    header_font = Font(name=font_name, size=int(styles["header_font_size"]), bold=True, color="FFFFFF")
    data_font = Font(name=font_name, size=int(styles["data_font_size"]), bold=False, color=styles["data_text_color"])
    bold_pass = Font(name=font_name, size=int(styles["data_font_size"]), bold=True, color=styles["pass_text_color"])
    bold_fail = Font(name=font_name, size=int(styles["data_font_size"]), bold=True, color=styles["fail_text_color"])

    border_color = str(styles["border_color"])
    thin_border = Border(
        left=Side(style="thin", color=border_color),
        right=Side(style="thin", color=border_color),
        top=Side(style="thin", color=border_color),
        bottom=Side(style="thin", color=border_color)
    )

    align_left_wrap = Alignment(horizontal="left", vertical="top", wrap_text=True)
    align_center = Alignment(horizontal="center", vertical="top", wrap_text=True)

    headers = list(_EXCEL_LABELS["batch_headers"]) + list(_EXCEL_LABELS["judge_extra_headers"])

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
        if case_id in answers_cache:
            agent_ans = answers_cache[case_id].get("answer", "")
        else:
            agent_ans = str(_FEEDBACK["pending_agent_placeholder"])

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
                score_acc = str(_FEEDBACK["pending_judge_cell"])
                score_faith = str(_FEEDBACK["pending_judge_cell"])
                score_comp = str(_FEEDBACK["pending_judge_cell"])
                score_overall = str(_FEEDBACK["pending_judge_cell"])
                verdict = str(_FEEDBACK["pending_judge_verdict"])
                feedback = str(_FEEDBACK["pending_judge_feedback"])

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

    col_widths = _EXCEL_CFG["column_widths"]["judge_16_cols"]
    for col_letter, w in col_widths.items():
        ws.column_dimensions[col_letter].width = w

    try:
        wb.save(output_path)
    except PermissionError:
        alt_path = output_path.replace(".xlsx", str(_EXCEL_CFG["fallback_suffixes"]["agent_xlsx"]))
        wb.save(alt_path)


def run_batch_and_judge():
    answers_cache_file = str(resolve_repo_path(_OUT_CFG["answers_785_cache_relpath"], root=PROJECT_ROOT))
    scores_cache_file = str(resolve_repo_path(_OUT_CFG["scores_785_cache_relpath"], root=PROJECT_ROOT))
    excel_path = str(resolve_repo_path(_OUT_CFG["master_agent_excel_relpath"], root=PROJECT_ROOT))

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
    judge_client = FallbackChatClient(module=str(_J785["judge_client_module"]))

    # PHASE 1: Judge all existing answers that haven't been scored yet
    cases_by_id = {}
    for c in all_cases:
        cases_by_id[c.get("id")] = c

    unscored_ids = []
    for cid, ans_obj in answers_cache.items():
        if cid not in scores_cache:
            unscored_ids.append(cid)

    judge_autosave = int(_J785["autosave_interval_judge"])
    master_autosave = int(_J785["autosave_interval_master"])

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

            if j_idx % judge_autosave == 0:
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
                "answer": f"{_FEEDBACK['pipeline_error_prefix']}{e}]",
                "latency_sec": latency,
                "citations": [],
                "audit_status": "ERROR",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            save_cache(answers_cache, answers_cache_file)

        if idx % master_autosave == 0:
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
