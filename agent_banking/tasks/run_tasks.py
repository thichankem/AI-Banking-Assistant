"""
Automated Business Tasks Processing Node with Deduplication.
==========================================================================
Features:
1. Reads requests from tasks/input/sample_task_inputs.json.
2. Computes SHA-256 hash of each query and context.
3. Checks StateTracker: Reuses previously processed tasks (SKIP).
4. Dispatches new requests to BankingTaskManager.
5. Saves results to sample_task_outputs.json.

MANDATORY RULES:
1. NO __INIT__.PY.
2. NO TYPING LIBRARY.
3. ZERO TERNARY: Explicit if/else only.
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
from agent_banking.tasks.task_manager import BankingTaskManager
from agent_banking.state_tracker import StateTracker


def run_auto_tasks(
    input_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    force: bool = False
) -> list[dict]:
    """
    Automatically process tasks from input to output with deduplication.
    """
    cfg = Config()
    in_dir = PROJECT_ROOT / "agent_banking" / "tasks" / "input"
    in_dir.mkdir(parents=True, exist_ok=True)

    if output_dir is not None:
        out_path = Path(output_dir)
    else:
        out_path = PROJECT_ROOT / "agent_banking" / "tasks" / "output"
    out_path.mkdir(parents=True, exist_ok=True)

    state_file = out_path / ".state_tracker.json"
    tracker = StateTracker(state_file)
    if force:
        tracker.clear()

    # 1. Resolve input source
    target_in_file = None
    if input_file is not None:
        target_in_file = Path(input_file)
    else:
        candidate = in_dir / "sample_task_inputs.json"
        if candidate.exists():
            target_in_file = candidate

    if target_in_file is None or not target_in_file.exists():
        print(f"⚠️ Task inputs file not found in: {in_dir}")
        return []

    print(f"\n{'='*75}")
    print(f"▶ [MODULE 10: TASKS] Input: {target_in_file.name} -> Output: {out_path}")
    print(f"{'='*75}")

    with open(target_in_file, "r", encoding="utf-8") as fp:
        raw_items = json.load(fp)

    if not raw_items:
        print("⚠️ Task inputs list is empty.")
        return []

    # 2. Read existing results for deduplication
    prev_results = {}
    prev_file = out_path / "sample_task_outputs.json"
    if prev_file.exists() and not force:
        try:
            with open(prev_file, "r", encoding="utf-8") as fp:
                prev_list = json.load(fp)
                for item in prev_list:
                    q = item.get("user_query", item.get("query", ""))
                    if q:
                        prev_results[q] = item
            if len(prev_results) >= len(raw_items) and len(raw_items) > 0:
                q_hash_map = {}
                for item in raw_items:
                    if isinstance(item, str):
                        uq = item
                        ctx = {}
                    else:
                        uq = item.get("user_query", item.get("query", ""))
                        ctx = item.get("context", {})
                    ctx_str = json.dumps(ctx, sort_keys=True, ensure_ascii=False)
                    q_hash_map[uq] = tracker.compute_text_hash(f"{uq}:{ctx_str}")
                if tracker.are_all_items_processed(q_hash_map):
                    print(f"  ⏭️ [SKIP ALL] All {len(raw_items)} tasks previously completed. Skipped in 0.0s.")
                    print(f"✅ [Tasks Complete] Exported {len(prev_list)} task results to: {prev_file} in 0.0s")
                    return prev_list
        except Exception:
            prev_results = {}

    task_manager = BankingTaskManager(config=cfg)
    task_outputs: list[dict] = []
    skipped_count = 0
    new_count = 0
    start_t = time.time()

    for idx, item in enumerate(raw_items, start=1):
        if isinstance(item, str):
            uq = item
            ctx = {}
            exp_task = ""
        else:
            uq = item.get("user_query", item.get("query", ""))
            ctx = item.get("context", {})
            exp_task = item.get("expected_task_id", "")

        ctx_str = json.dumps(ctx, sort_keys=True, ensure_ascii=False)
        q_hash = tracker.compute_text_hash(f"{uq}:{ctx_str}")
        is_done = tracker.is_processed(uq, q_hash)

        if is_done and uq in prev_results:
            cached_res = dict(prev_results[uq])
            task_outputs.append(cached_res)
            skipped_count += 1
            print(f"  [{idx:02d}] [SKIP] '{uq[:35]}...' (Already processed)")
        else:
            t0 = time.time()
            res_dict = task_manager.process_query(uq, context=ctx)
            dt_ms = round((time.time() - t0) * 1000, 2)

            res = {
                "user_query": uq,
                "task_id": res_dict.get("task_id", ""),
                "task_name": res_dict.get("task_name", ""),
                "status": res_dict.get("status", "SUCCESS"),
                "answer": res_dict.get("answer", ""),
                "citations": res_dict.get("citations", []),
                "latency_ms": dt_ms,
                "expected_task_id": exp_task
            }
            task_outputs.append(res)
            tracker.record_processed(uq, q_hash, {"task_id": res_dict.get("task_id", "")})
            new_count += 1
            print(f"  [{idx:02d}] [NEW]  '{uq[:35]}...' -> Task: {res_dict.get('task_id', '')} ({dt_ms}ms)")

    tracker.save()
    total_elapsed = round(time.time() - start_t, 2)

    # 3. Export output
    out_file = out_path / "sample_task_outputs.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(task_outputs, fp, ensure_ascii=False, indent=2)

    print(f"✅ [Tasks Complete] Exported {len(task_outputs)} task results to: {out_file} (New: {new_count}, Skipped: {skipped_count}) in {total_elapsed}s")
    return task_outputs


if __name__ == "__main__":
    run_auto_tasks()
