"""Real end-to-end runner for NV-200 operational benchmark against indexed Qdrant knowledge."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill


def _load_nv_cases(dataset_path: Path) -> list[dict[str, Any]]:
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        if "cases" in raw:
            return list(raw["cases"])
        if "items" in raw:
            return list(raw["items"])
    raise ValueError("Unsupported NV dataset format")


def _export_nv_excel(
    cases: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    excel_path: Path,
) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "NV 200 Real Evaluation"
    headers = [
        "STT",
        "ID",
        "Topic",
        "Document Title",
        "Test Criteria",
        "Difficulty",
        "Question",
        "Reference Answer",
        "Must Include",
        "Agent Answer",
        "Citations",
        "Audit Status",
        "Accuracy",
        "Faithfulness",
        "Completeness",
        "Overall Score",
        "Verdict",
        "Feedback",
        "Latency (s)",
    ]
    ws.append(headers)
    fill = PatternFill(start_color="0F2240", end_color="0F2240", fill_type="solid")
    font = Font(color="FFFFFF", bold=True, size=10)
    for col_idx in range(1, len(headers) + 1):
        c = ws.cell(row=1, column=col_idx)
        c.fill = fill
        c.font = font
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for idx, case in enumerate(cases, start=1):
        cid = str(case["id"])
        row = results.get(cid, {})
        judge = row.get("judge")
        if not isinstance(judge, dict):
            judge = {}
        must_inc = case.get("rubric", {}).get("must_include", [])
        if isinstance(must_inc, list):
            must_text = "\n".join(str(x) for x in must_inc)
        else:
            must_text = str(must_inc)
        cits = row.get("citations", [])
        ws.append([
            idx,
            cid,
            str(case.get("topic_id", "")),
            str(case.get("document_title", "")),
            str(case.get("test_criteria", "")),
            str(case.get("difficulty", "")),
            str(case.get("question", "")),
            str(case.get("reference_answer", "")),
            must_text,
            str(row.get("answer", "")),
            json.dumps(cits, ensure_ascii=False),
            str(row.get("audit_status", "")),
            judge.get("accuracy", ""),
            judge.get("faithfulness", ""),
            judge.get("completeness", ""),
            judge.get("overall_score", ""),
            str(row.get("verdict", "PENDING")),
            str(judge.get("feedback", "")),
            row.get("latency_sec", ""),
        ])
    ws.freeze_panes = "A2"
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(excel_path)


def run_nv_real(args: argparse.Namespace) -> dict[str, Any]:
    """Execute NV-200 queries live through BankingAgentPipeline and Cloud LLM Judge."""
    os.environ["ABANK_QDRANT_STORAGE_PATH"] = str(args.index)

    from agent_banking.agent_pipeline import BankingAgentPipeline
    from agent_banking.core.chat import FallbackChatClient
    from agent_banking.core.settings import Config
    from agent_banking.evaluator.master_785_runner import call_llm_judge
    from agent_banking.pipeline import BankingRAGPipeline

    cfg = Config()
    cfg.QDRANT_STORAGE_PATH = str(args.index)
    cfg.ABANK_QDRANT_STORAGE_PATH = str(args.index)
    rag = BankingRAGPipeline(config=cfg)
    pipe = BankingAgentPipeline(config=cfg, rag_pipeline=rag, use_memory=True)
    judge_client = FallbackChatClient(module="agent")

    cases = _load_nv_cases(args.dataset)
    if args.limit is not None and args.limit > 0:
        cases = cases[: args.limit]

    args.output.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output / "nv_real_results.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if ckpt_path.exists():
        for line in ckpt_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if args.retry_errors and row.get("verdict") == "ERROR":
                continue
            completed[str(row["id"])] = row

    pending = [c for c in cases if str(c["id"]) not in completed]
    lock = threading.Lock()

    def _run_one(case: dict[str, Any]) -> dict[str, Any]:
        cid = str(case["id"])
        q = str(case.get("question", ""))
        ref_ans = str(case.get("reference_answer", ""))
        must_inc = case.get("rubric", {}).get("must_include", [])
        t0 = time.perf_counter()
        try:
            res = pipe.chat(query=q, session_id=f"nv_real_{cid}")
            ans = str(res.get("answer", ""))
            judge = call_llm_judge(judge_client, q, ref_ans, must_inc, ans)
            verdict = str(judge.get("verdict", "FAIL"))
            return {
                "id": cid,
                "topic_id": case.get("topic_id", ""),
                "test_criteria": case.get("test_criteria", ""),
                "difficulty": case.get("difficulty", ""),
                "answer": ans,
                "citations": res.get("citations", []),
                "sources": res.get("sources", []),
                "audit_status": res.get("audit_status", ""),
                "tool_calls": [c.get("tool") for c in res.get("tool_calls", []) if isinstance(c, dict)],
                "judge": judge,
                "verdict": verdict,
                "latency_sec": round(time.perf_counter() - t0, 2),
            }
        except Exception as exc:
            return {
                "id": cid,
                "topic_id": case.get("topic_id", ""),
                "test_criteria": case.get("test_criteria", ""),
                "difficulty": case.get("difficulty", ""),
                "answer": "",
                "citations": [],
                "sources": [],
                "audit_status": "ERROR",
                "tool_calls": [],
                "judge": {"overall_score": 0.0, "verdict": "ERROR", "feedback": str(exc)},
                "verdict": "ERROR",
                "latency_sec": round(time.perf_counter() - t0, 2),
            }

    try:
        with ckpt_path.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_run_one, c) for c in pending]
            for fut in as_completed(futures):
                row = fut.result()
                with lock:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    completed[str(row["id"])] = row
                    print(f"[NV-REAL] {len(completed)}/{len(cases)} {row['id']} {row['verdict']}", flush=True)
    finally:
        try:
            pipe.graph.checkpointer.conn.close()
        except Exception:
            pass
        rag.close()

    total = len(cases)
    passed = 0
    failed = 0
    errors = 0
    score_sum = 0.0
    by_criteria: dict[str, dict[str, int]] = {}
    for c in cases:
        row = completed.get(str(c["id"]), {})
        v = str(row.get("verdict", "PENDING"))
        if v == "PASS":
            passed += 1
        elif v == "ERROR":
            errors += 1
        else:
            failed += 1
        score_sum += float((row.get("judge") or {}).get("overall_score", 0.0))
        crit = str(c.get("test_criteria", "other"))
        bucket = by_criteria.setdefault(crit, {"total": 0, "PASS": 0, "FAIL": 0, "ERROR": 0})
        bucket["total"] += 1
        if v in bucket:
            bucket[v] += 1
        else:
            bucket["FAIL"] += 1

    avg_score = 0.0
    if total > 0:
        avg_score = round(score_sum / total, 2)
    pass_rate = 0.0
    if total > 0:
        pass_rate = round(passed * 100.0 / total, 2)

    summary = {
        "benchmark": "NV-200 Real End-to-End",
        "index": str(args.index),
        "total_cases": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "pass_rate_pct": pass_rate,
        "avg_score": avg_score,
        "by_criteria": by_criteria,
    }
    (args.output / "nv_real_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _export_nv_excel(cases, completed, args.output / "nv_200_real_report.xlsx")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Live NV-200 real benchmark runner")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset/A_Bank_golden_nv/a_bank_nv_golden_200.json"),
    )
    parser.add_argument("--index", type=Path, default=Path("agent_banking/vector_db/qdrant_full_141"))
    parser.add_argument("--output", type=Path, default=Path("agent_banking/outputs/nv_200_real"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    summary = run_nv_real(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
