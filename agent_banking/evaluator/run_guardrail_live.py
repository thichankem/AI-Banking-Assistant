"""Live Guardrails runner with SemanticBankingGuardrail and Decision Model enabled."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import threading
import time
from typing import Any


def _group_prefix(case_id: str) -> str:
    if "-" in case_id:
        return case_id.split("-", 1)[0]
    return case_id[:4]


def run_guardrail_live(args: argparse.Namespace) -> dict[str, Any]:
    """Run GuardrailManager with semantic + decision-model live checks (and optional full agent)."""
    os.environ["ABANK_QDRANT_STORAGE_PATH"] = str(args.index)

    from agent_banking.agent_pipeline import BankingAgentPipeline
    from agent_banking.core.settings import Config
    from agent_banking.guardrails.rag_guardrails import BankingGuardrailManager
    from agent_banking.pipeline import BankingRAGPipeline

    Config.ENABLE_SEMANTIC_GUARDRAIL = True
    pipe = None
    rag = None
    if args.full_agent:
        cfg = Config()
        cfg.ENABLE_SEMANTIC_GUARDRAIL = True
        cfg.QDRANT_STORAGE_PATH = str(args.index)
        cfg.ABANK_QDRANT_STORAGE_PATH = str(args.index)
        rag = BankingRAGPipeline(config=cfg)
        pipe = BankingAgentPipeline(config=cfg, rag_pipeline=rag, use_memory=True)
    else:
        rag = BankingRAGPipeline(use_memory=True)
    manager = BankingGuardrailManager(embedder=rag.embedder, use_decision_model=True)

    cases = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.limit is not None and args.limit > 0:
        cases = cases[: args.limit]

    args.output.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output / "guardrail_live_results.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if ckpt_path.exists():
        for line in ckpt_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if args.retry_errors and row.get("status") == "ERROR":
                continue
            completed[str(row["id"])] = row

    pending = [c for c in cases if str(c["id"]) not in completed]
    lock = threading.Lock()

    def _run_one(case: dict[str, Any]) -> dict[str, Any]:
        cid = str(case["id"])
        q = str(case.get("cau_hoi", case.get("question", "")))
        tieu_chi = str(case.get("tieu_chi", case.get("group", "")))
        t0 = time.perf_counter()
        try:
            g_res = manager.guard_input(q, use_decision_model=True)
            is_allowed = bool(getattr(g_res, "is_safe", True))
            reason = str(getattr(g_res, "trigger_reason", "") or "")
            risk_cat = str(getattr(g_res, "trigger_reason", "") or "")
            agent_blocked = not is_allowed
            agent_answer = ""
            if pipe is not None and is_allowed:
                p_res = pipe.chat(query=q, session_id=f"gr_live_{cid}")
                agent_answer = str(p_res.get("answer", ""))
                if p_res.get("status") == "blocked" or p_res.get("audit_status") == "BLOCKED":
                    agent_blocked = True
            return {
                "id": cid,
                "group": _group_prefix(cid),
                "tieu_chi": tieu_chi,
                "guard_input_blocked": not is_allowed,
                "effective_blocked": agent_blocked,
                "reason": reason,
                "risk_category": risk_cat,
                "agent_answer_snippet": agent_answer[:220],
                "status": "OK",
                "latency_sec": round(time.perf_counter() - t0, 3),
            }
        except Exception as exc:
            return {
                "id": cid,
                "group": _group_prefix(cid),
                "tieu_chi": tieu_chi,
                "guard_input_blocked": False,
                "effective_blocked": False,
                "reason": str(exc),
                "risk_category": "ERROR",
                "agent_answer_snippet": "",
                "status": "ERROR",
                "latency_sec": round(time.perf_counter() - t0, 3),
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
                    print(f"[GR-LIVE] {len(completed)}/{len(cases)} {row['id']} blocked={row['effective_blocked']}", flush=True)
    finally:
        if rag is not None:
            rag.close()

    total = len(cases)
    blocked_input = 0
    blocked_eff = 0
    errors = 0
    by_group: dict[str, dict[str, Any]] = {}
    for c in cases:
        cid = str(c["id"])
        row = completed.get(cid, {})
        grp = _group_prefix(cid)
        bucket = by_group.setdefault(
            grp,
            {
                "tieu_chi": str(c.get("tieu_chi", c.get("group", ""))),
                "total": 0,
                "guard_input_blocked": 0,
                "effective_blocked": 0,
                "errors": 0,
            },
        )
        bucket["total"] += 1
        if row.get("status") == "ERROR":
            errors += 1
            bucket["errors"] += 1
        if row.get("guard_input_blocked"):
            blocked_input += 1
            bucket["guard_input_blocked"] += 1
        if row.get("effective_blocked"):
            blocked_eff += 1
            bucket["effective_blocked"] += 1

    block_rate = 0.0
    if total > 0:
        block_rate = round(blocked_eff * 100.0 / total, 2)

    summary = {
        "benchmark": "Guardrails-2000 Live (Semantic + Decision Model)",
        "full_agent": bool(args.full_agent),
        "total_cases": total,
        "guard_input_blocked": blocked_input,
        "effective_blocked": blocked_eff,
        "block_rate_pct": block_rate,
        "errors": errors,
        "by_group": by_group,
    }
    (args.output / "guardrail_live_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Guardrail runner with semantic & decision model")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset/guardrails_2000/a_bank_guardrails_2000.json"),
    )
    parser.add_argument("--index", type=Path, default=Path("agent_banking/vector_db/qdrant_full_141"))
    parser.add_argument("--output", type=Path, default=Path("agent_banking/outputs/guardrail_live"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--full-agent", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    summary = run_guardrail_live(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
