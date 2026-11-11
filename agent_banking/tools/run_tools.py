"""
Automated Tool Execution Node with Deduplication.
===================================================================
Features:
1. Reads tool call requests from tools/input/sample_tool_inputs.json.
2. Computes SHA-256 hash of each tool_name and arguments.
3. Checks StateTracker: Reuses previously completed tool calls (SKIP).
4. Invokes BankingToolRegistry for new tool calls.
5. Exports results to sample_tool_outputs.json.

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

from agent_banking.tools.tool_registry import BankingToolRegistry
from agent_banking.state_tracker import StateTracker


def run_auto_tools(
    input_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    force: bool = False
) -> list[dict]:
    """
    Automatically execute tools from input to output with deduplication.
    """
    in_dir = PROJECT_ROOT / "agent_banking" / "tools" / "input"
    in_dir.mkdir(parents=True, exist_ok=True)

    if output_dir is not None:
        out_path = Path(output_dir)
    else:
        out_path = PROJECT_ROOT / "agent_banking" / "tools" / "output"
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
        candidate = in_dir / "sample_tool_inputs.json"
        if candidate.exists():
            target_in_file = candidate

    if target_in_file is None or not target_in_file.exists():
        print(f"⚠️ Tool inputs file not found in: {in_dir}")
        return []

    print(f"\n{'='*75}")
    print(f"▶ [MODULE 12: TOOLS] Input: {target_in_file.name} -> Output: {out_path}")
    print(f"{'='*75}")

    with open(target_in_file, "r", encoding="utf-8") as fp:
        raw_items = json.load(fp)

    if not raw_items:
        print("⚠️ Tool inputs list is empty.")
        return []

    # 2. Read existing results for deduplication
    prev_results = {}
    prev_file = out_path / "sample_tool_outputs.json"
    if prev_file.exists() and not force:
        try:
            with open(prev_file, "r", encoding="utf-8") as fp:
                prev_list = json.load(fp)
                for item in prev_list:
                    t_name = item.get("tool_name", "")
                    if t_name:
                        prev_results[t_name] = item
            if len(prev_results) >= len(raw_items) and len(raw_items) > 0:
                t_hash_map = {}
                for item in raw_items:
                    tool_name = item.get("tool_name", "")
                    args = item.get("arguments", item.get("params", {}))
                    args_str = json.dumps(args, sort_keys=True, ensure_ascii=False)
                    t_hash_map[tool_name] = tracker.compute_text_hash(f"{tool_name}:{args_str}")
                if tracker.are_all_items_processed(t_hash_map):
                    print(f"  ⏭️ [SKIP ALL] All {len(raw_items)} tools previously executed. Skipped in 0.0s.")
                    print(f"✅ [Tools Complete] Exported {len(prev_list)} tool outputs to: {prev_file} in 0.0s")
                    return prev_list
        except Exception:
            prev_results = {}

    registry = BankingToolRegistry()
    tool_outputs: list[dict] = []
    skipped_count = 0
    new_count = 0
    start_t = time.time()

    for idx, item in enumerate(raw_items, start=1):
        tool_name = item.get("tool_name", "")
        args = item.get("arguments", item.get("params", {}))
        args_str = json.dumps(args, sort_keys=True, ensure_ascii=False)
        req_hash = tracker.compute_text_hash(f"{tool_name}:{args_str}")

        is_done = tracker.is_processed(tool_name, req_hash)

        if is_done and tool_name in prev_results:
            cached_res = dict(prev_results[tool_name])
            tool_outputs.append(cached_res)
            skipped_count += 1
            print(f"  [{idx:02d}] [SKIP] Tool '{tool_name}' (Already executed)")
        else:
            t0 = time.time()
            res_dict = registry.execute_tool(tool_name, args)
            dt_ms = round((time.time() - t0) * 1000, 2)

            res = {
                "tool_name": tool_name,
                "arguments": args,
                "result": res_dict,
                "status": res_dict.get("status", "SUCCESS"),
                "latency_ms": dt_ms
            }
            tool_outputs.append(res)
            tracker.record_processed(tool_name, req_hash, {"status": res["status"]})
            new_count += 1
            print(f"  [{idx:02d}] [NEW]  Tool '{tool_name}' -> {res['status']} ({dt_ms}ms)")

    tracker.save()
    total_elapsed = round(time.time() - start_t, 2)

    # 3. Export output
    out_file = out_path / "sample_tool_outputs.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(tool_outputs, fp, ensure_ascii=False, indent=2)

    print(f"✅ [Tools Complete] Exported {len(tool_outputs)} tool outputs to: {out_file} (New: {new_count}, Skipped: {skipped_count}) in {total_elapsed}s")
    return tool_outputs


if __name__ == "__main__":
    run_auto_tools()
