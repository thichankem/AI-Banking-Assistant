"""Real end-to-end runner for Excel-350 benchmark (BP-01..BP-10 and ML-01) ignoring cached columns."""

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


def _parse_must_include(raw: Any) -> list[str]:
    if raw is None:
        return []
    out: list[str] = []
    for line in str(raw).splitlines():
        cleaned = line.strip()
        if cleaned.startswith("•") or cleaned.startswith("-") or cleaned.startswith("*"):
            cleaned = cleaned[1:].strip()
        if cleaned:
            out.append(cleaned)
    return out


def _load_excel_350_cases(excel_path: Path) -> list[dict[str, Any]]:
    wb = openpyxl.load_workbook(excel_path, read_only=True)
    ws = wb[wb.sheetnames[0]]
    cases: list[dict[str, Any]] = []
    for idx, row in enumerate(ws.iter_rows(values_only=True), start=1):
        if idx == 1:
            continue
        if not row or len(row) < 9:
            continue
        cid = str(row[1] or "").strip()
        if not cid:
            continue
        cases.append({
            "stt": row[0],
            "id": cid,
            "doc_code": str(row[2] or "").strip(),
            "test_criteria": str(row[3] or "").strip(),
            "difficulty": str(row[4] or "").strip(),
            "question": str(row[5] or "").strip(),
            "reference_answer": str(row[6] or "").strip(),
            "must_include": _parse_must_include(row[7]),
            "source_path": str(row[8] or "").strip(),
        })
    wb.close()
    return cases


def _export_excel_report(
    cases: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    out_xlsx: Path,
) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Excel 350 Real Live"
    headers = [
        "STT",
        "ID",
        "Doc Code",
        "Test Criteria",
        "Difficulty",
        "Question",
        "Reference Answer",
        "Must Include",
        "Expected Source",
        "Agent Answer",
        "Citations",
        "Source Match",
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
        cid = case["id"]
        row = results.get(cid, {})
        judge = row.get("judge")
        if not isinstance(judge, dict):
            judge = {}
        ws.append([
            idx,
            cid,
            case["doc_code"],
            case["test_criteria"],
            case["difficulty"],
            case["question"],
            case["reference_answer"],
            "\n".join(case["must_include"]),
            case["source_path"],
            str(row.get("answer", "")),
            json.dumps(row.get("sources", []), ensure_ascii=False),
            str(row.get("source_matched", False)),
            judge.get("accuracy", ""),
            judge.get("faithfulness", ""),
            judge.get("completeness", ""),
            judge.get("overall_score", ""),
            str(row.get("verdict", "PENDING")),
            str(judge.get("feedback", "")),
            row.get("latency_sec", ""),
        ])
    ws.freeze_panes = "A2"
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_xlsx)


def _check_source_match(expected_source: str, actual_sources: list[str]) -> bool:
    exp_name = expected_source.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if not exp_name:
        return True
    exp_code = exp_name[:5].replace(" ", "")
    for src in actual_sources:
        s_low = str(src).lower()
        if exp_name in s_low:
            return True
        if exp_code and exp_code in s_low.replace(" ", ""):
            return True
    return False


def run_excel_350_real(args: argparse.Namespace) -> dict[str, Any]:
    """Execute Excel-350 benchmark strictly live through BankingAgentPipeline."""
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

    cases = _load_excel_350_cases(args.dataset)
    if args.limit is not None and args.limit > 0:
        cases = cases[: args.limit]

    args.output.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output / "excel_350_real_results.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if ckpt_path.exists():
        for line in ckpt_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if args.retry_errors and row.get("verdict") == "ERROR":
                continue
            completed[str(row["id"])] = row

    pending = [c for c in cases if c["id"] not in completed]
    lock = threading.Lock()

    def _run_one(case: dict[str, Any]) -> dict[str, Any]:
        cid = case["id"]
        q = case["question"]
        t0 = time.perf_counter()
        try:
            res = pipe.chat(query=q, session_id=f"excel350_real_{cid}")
            ans = str(res.get("answer", ""))
            sources = [str(s) for s in res.get("sources", [])]
            src_matched = _check_source_match(case["source_path"], sources)
            judge = call_llm_judge(judge_client, q, case["reference_answer"], case["must_include"], ans)
            verdict = str(judge.get("verdict", "FAIL"))
            return {
                "id": cid,
                "doc_code": case["doc_code"],
                "test_criteria": case["test_criteria"],
                "difficulty": case["difficulty"],
                "answer": ans,
                "citations": res.get("citations", []),
                "sources": sources,
                "source_matched": src_matched,
                "audit_status": res.get("audit_status", ""),
                "tool_calls": [c.get("tool") for c in res.get("tool_calls", []) if isinstance(c, dict)],
                "judge": judge,
                "verdict": verdict,
                "latency_sec": round(time.perf_counter() - t0, 2),
            }
        except Exception as exc:
            return {
                "id": cid,
                "doc_code": case["doc_code"],
                "test_criteria": case["test_criteria"],
                "difficulty": case["difficulty"],
                "answer": "",
                "citations": [],
                "sources": [],
                "source_matched": False,
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
                    print(f"[EXCEL350-REAL] {len(completed)}/{len(cases)} {row['id']} {row['verdict']} src={row['source_matched']}", flush=True)
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
    src_hits = 0
    score_sum = 0.0
    by_doc: dict[str, dict[str, int]] = {}
    for c in cases:
        row = completed.get(c["id"], {})
        v = str(row.get("verdict", "PENDING"))
        if v == "PASS":
            passed += 1
        elif v == "ERROR":
            errors += 1
        else:
            failed += 1
        if row.get("source_matched"):
            src_hits += 1
        score_sum += float((row.get("judge") or {}).get("overall_score", 0.0))
        bucket = by_doc.setdefault(c["doc_code"], {"total": 0, "PASS": 0, "FAIL": 0, "ERROR": 0, "src_hit": 0})
        bucket["total"] += 1
        if v in bucket:
            bucket[v] += 1
        else:
            bucket["FAIL"] += 1
        if row.get("source_matched"):
            bucket["src_hit"] += 1

    pass_rate = 0.0
    src_rate = 0.0
    avg_score = 0.0
    if total > 0:
        pass_rate = round(passed * 100.0 / total, 2)
        src_rate = round(src_hits * 100.0 / total, 2)
        avg_score = round(score_sum / total, 2)

    summary = {
        "benchmark": "Excel-350 Real Live",
        "index": str(args.index),
        "total_cases": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "pass_rate_pct": pass_rate,
        "source_citation_match_pct": src_rate,
        "avg_score": avg_score,
        "by_doc_code": by_doc,
    }
    (args.output / "excel_350_real_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _export_excel_report(cases, completed, args.output / "excel_350_real_report.xlsx")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Excel-350 real benchmark runner")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset/A_Bank_golden_excel_350_evaluation_llm.xlsx"),
    )
    parser.add_argument("--index", type=Path, default=Path("agent_banking/vector_db/qdrant_full_141"))
    parser.add_argument("--output", type=Path, default=Path("agent_banking/outputs/excel_350_real"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    summary = run_excel_350_real(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
