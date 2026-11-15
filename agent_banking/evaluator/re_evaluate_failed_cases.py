"""
Script to Re-evaluate and Fix all previously Failed Cases.
Runs failed cases through the fixed LangGraph Agent Pipeline and re-judges them with LLM-as-a-Judge.
Synchronizes the results to both JSON caches and the master Excel file.

RULES COMPLIANCE:
1. 0% REGEX LAW: No regex imports or functions.
2. ZERO TERNARY LAW: Explicit if/else statements only.
3. NO TYPING LIBRARY: Native Python types only.
4. NO __INIT__.PY: Direct imports only.
"""

import os
import sys
import time
import json
from pathlib import Path

PROJECT_ROOT = Path(r"d:\Desktop\GitHub\AI Banking Asistant")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.agent_pipeline import BankingAgentPipeline
from agent_banking.core.chat import FallbackChatClient
from agent_banking.evaluator.run_agent_785_eval_with_judge import (
    load_all_785_cases,
    load_cache,
    save_cache,
    call_llm_judge,
    export_to_excel
)

def fix_failed_cases():
    base_dir = r"d:\Desktop\GitHub\AI Banking Asistant"
    answers_cache_file = os.path.join(base_dir, "dataset", "agent_answers_785_cache.json")
    scores_cache_file = os.path.join(base_dir, "dataset", "agent_eval_scores_785_cache.json")
    excel_path = os.path.join(base_dir, "dataset", "A_Bank_golden_master_785_evaluation_llm.xlsx")

    all_cases = load_all_785_cases()
    cases_by_id = {}
    for c in all_cases:
        cases_by_id[c.get("id")] = c

    answers_cache = load_cache(answers_cache_file)
    scores_cache = load_cache(scores_cache_file)

    # Identify all failed or low-score cases
    failed_ids = []
    for cid, sc in scores_cache.items():
        verdict = sc.get("verdict", "FAIL")
        score = float(sc.get("overall_score", 0.0))
        if verdict == "FAIL" or score < 75.0:
            failed_ids.append(cid)

    print("=" * 70)
    print("RE-EVALUATING AND FIXING FAILED CASES IN AGENT BANKING")
    print(f"Total Failed Cases to Fix: {len(failed_ids)}")
    print("=" * 70)

    if len(failed_ids) == 0:
        print("No failed cases found! All cases passed.")
        return

    print("\nInitializing Fixed LangGraph Pipeline and LLM Judge...")
    pipeline = BankingAgentPipeline()
    judge_client = FallbackChatClient(module="generator")

    fixed_pass_count = 0
    start_total_time = time.time()

    for idx, cid in enumerate(failed_ids, 1):
        case_data = cases_by_id.get(cid, {})
        q = case_data.get("question", "")
        ref_ans = case_data.get("reference_answer", "")
        must_inc = case_data.get("rubric", {}).get("must_include", [])

        print(f"\n[{idx:03d}/{len(failed_ids):03d}] Fixing {cid}...")
        print(f"  Q: {q[:75]}...")

        t0 = time.time()
        try:
            res = pipeline.chat(query=q, session_id=f"fixed_{cid}")
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
            print(f"  -> Pipeline answered in {latency}s | Audit: {audit_status}")

            # Re-evaluate with LLM Judge
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
                fixed_pass_count += 1

            print(f"  -> Judge ({j_latency}s): Acc={acc} | Faith={faith} | Comp={comp} | Total={overall} [{verdict}]")
            print(f"  -> Feedback: {score_res.get('feedback', '')[:100]}...")

        except Exception as e:
            print(f"  -> ERROR fixing {cid}: {e}")

        # Auto-save Excel every 5 cases
        if idx % 5 == 0:
            export_to_excel(all_cases, answers_cache, scores_cache, excel_path)
            print(f"  [Auto-Save] Excel synchronized (Fixed: {idx}/{len(failed_ids)})")

    # Final export
    export_to_excel(all_cases, answers_cache, scores_cache, excel_path)
    total_elapsed = round(time.time() - start_total_time, 2)
    print("\n" + "=" * 70)
    print("ALL FAILED CASES RE-EVALUATION COMPLETED!")
    print(f"Total cases re-evaluated: {len(failed_ids)}")
    print(f"Successfully turned into PASS: {fixed_pass_count}/{len(failed_ids)}")
    print(f"Total time: {total_elapsed}s")
    print(f"Updated Excel Workbook: {excel_path}")
    print("=" * 70)

if __name__ == "__main__":
    fix_failed_cases()
