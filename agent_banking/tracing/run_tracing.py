"""
Automated Tracing Collection Runner (Auto Tracing Node with Deduplication).
===========================================================================
Features:
1. Reads trace inputs from tracing/input/sample_trace_inputs.json.
2. Computes SHA-256 hash of each trace request.
3. Checks StateTracker: Reuses previously traced requests (SKIP).
4. Records spans and metrics for new trace queries.
5. Emits results to: sample_trace_outputs.json.

MANDATORY INVARIANTS:
1. NO __INIT__.PY.
2. NO TYPING LIBRARY.
3. ZERO TERNARY: Explicit if/else statements.
4. 0% REGEX LAW.
"""

from __future__ import annotations
import sys
import json
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.core.settings import Config
from agent_banking.tracing.tracer import BankingTracer
from agent_banking.state_tracker import StateTracker


def run_auto_tracing(
    input_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    force: bool = False
) -> list[dict]:
    """
    Tu dong xu ly tracing tu input sang output kem co che chong trung lap.
    """
    cfg = Config()
    in_dir = PROJECT_ROOT / "agent_banking" / "tracing" / "input"
    in_dir.mkdir(parents=True, exist_ok=True)

    if output_dir is not None:
        out_path = Path(output_dir)
    else:
        out_path = PROJECT_ROOT / "agent_banking" / "tracing" / "output"
    out_path.mkdir(parents=True, exist_ok=True)

    state_file = out_path / ".state_tracker.json"
    tracker = StateTracker(state_file)
    if force:
        tracker.clear()

    # 1. Tim nguon input
    target_in_file = None
    if input_file is not None:
        target_in_file = Path(input_file)
    else:
        candidate = in_dir / "sample_trace_inputs.json"
        if candidate.exists():
            target_in_file = candidate

    if target_in_file is None or not target_in_file.exists():
        print(f"⚠️ Khong tim thay tap tin trace input trong: {in_dir}")
        return []

    print(f"\n{'='*75}")
    print(f"▶ [MODULE 9: TRACING] Input: {target_in_file.name} -> Output: {out_path}")
    print(f"{'='*75}")

    with open(target_in_file, "r", encoding="utf-8") as fp:
        raw_traces = json.load(fp)

    if not raw_traces:
        print("⚠️ Danh sach trace input rong.")
        return []

    # 2. Doc ket qua cu de tai su dung
    prev_results = {}
    prev_file = out_path / "sample_trace_outputs.json"
    if prev_file.exists() and not force:
        try:
            with open(prev_file, "r", encoding="utf-8") as fp:
                prev_list = json.load(fp)
                for item in prev_list:
                    tid = item.get("trace_id", "")
                    if tid:
                        prev_results[tid] = item
            if len(prev_results) >= len(raw_traces) and len(raw_traces) > 0:
                t_hash_map = {}
                for item in raw_traces:
                    tid = item.get("trace_id", "")
                    q = item.get("query", "")
                    sess = item.get("session_id", "default")
                    t_hash_map[tid] = tracker.compute_text_hash(f"{tid}:{q}:{sess}")
                if tracker.are_all_items_processed(t_hash_map):
                    print(f"  ⏭️ [SKIP ALL] Tat ca {len(raw_traces)} traces da duoc ghi log truoc do. Bo qua toan bo trong 0.0s!")
                    print(f"✅ [Tracing Hoan Tat] Da xuat {len(prev_list)} trace records vao: {prev_file} trong 0.0s")
                    return prev_list
        except Exception:
            prev_results = {}

    tracer = BankingTracer(config=cfg)
    trace_outputs: list[dict] = []
    skipped_count = 0
    new_count = 0
    start_t = time.time()

    for item in raw_traces:
        tid = item.get("trace_id", "tr_" + str(len(trace_outputs) + 1))
        q = item.get("query", "")
        sess = item.get("session_id", "default")
        meta = item.get("metadata", {})

        trace_hash = tracker.compute_text_hash(f"{tid}:{q}:{sess}")
        is_done = tracker.is_processed(tid, trace_hash)

        if is_done and tid in prev_results:
            cached_res = dict(prev_results[tid])
            trace_outputs.append(cached_res)
            skipped_count += 1
            print(f"  [{tid}] [SKIP] '{q[:35]}...' (Da ghi trace)")
        else:
            t0 = time.time()
            with tracer.trace_query(query=q) as trace:
                trace.record_query_parser(
                    input_query=q,
                    parsed=type("DummyParsed", (), {"normalized_query": q, "keywords": q.split()})(),
                    duration_ms=0.5
                )
                trace.record_embedding(num_texts=1, dimension=1024, duration_ms=1.2)
                trace.finish(answer="Trace record simulated successfully", is_blocked=False)

            dt_ms = round((time.time() - t0) * 1000, 2)
            res = {
                "trace_id": tid,
                "session_id": sess,
                "query": q,
                "status": "RECORDED",
                "phoenix_enabled": cfg.ENABLE_PHOENIX,
                "latency_ms": dt_ms,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            trace_outputs.append(res)
            tracker.record_processed(tid, trace_hash, {"status": "RECORDED"})
            new_count += 1
            print(f"  [{tid}] [NEW]  '{q[:35]}...' -> Ghi trace thanh cong ({dt_ms}ms)")

    tracker.save()
    total_elapsed = round(time.time() - start_t, 2)

    # 3. Xuat ket qua
    out_file = out_path / "sample_trace_outputs.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(trace_outputs, fp, ensure_ascii=False, indent=2)

    print(f"✅ [Tracing Hoan Tat] Da xuat {len(trace_outputs)} trace records vao: {out_file} (Moi: {new_count}, Bo qua: {skipped_count}) trong {total_elapsed}s")
    return trace_outputs


if __name__ == "__main__":
    run_auto_tracing()
