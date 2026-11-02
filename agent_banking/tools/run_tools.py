"""
Module Tu Dong Thuc Thi Tools (Auto Tools Node with Deduplication).
===================================================================
Dac diem:
1. Doc yeu cau goi cong cu tu tools/input/sample_tool_inputs.json.
2. Tinh toan SHA-256 hash cua tung tool_name va arguments.
3. Kiem tra StateTracker: Tool call nao da chay thi tai su dung (SKIP).
4. Chi kich hoat BankingToolRegistry thuc thi cac lenh goi moi.
5. Xuat ket qua: sample_tool_outputs.json.

NGUYEN TAC BAT BUOC:
1. KHONG DUNG __INIT__.PY.
2. KHONG DUNG THU VIEN TYPING.
3. ZERO TERNARY: if/else tuong minh.
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
    Tu dong thuc thi tools tu input sang output kem co che chong trung lap.
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

    # 1. Tim nguon input
    target_in_file = None
    if input_file is not None:
        target_in_file = Path(input_file)
    else:
        candidate = in_dir / "sample_tool_inputs.json"
        if candidate.exists():
            target_in_file = candidate

    if target_in_file is None or not target_in_file.exists():
        print(f"⚠️ Khong tim thay tap tin tool inputs trong: {in_dir}")
        return []

    print(f"\n{'='*75}")
    print(f"▶ [MODULE 12: TOOLS] Input: {target_in_file.name} -> Output: {out_path}")
    print(f"{'='*75}")

    with open(target_in_file, "r", encoding="utf-8") as fp:
        raw_items = json.load(fp)

    if not raw_items:
        print("⚠️ Danh sach tool inputs rong.")
        return []

    # 2. Doc ket qua cu de tai su dung
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
                    print(f"  ⏭️ [SKIP ALL] Tat ca {len(raw_items)} tools da duoc thuc thi truoc do. Bo qua toan bo trong 0.0s!")
                    print(f"✅ [Tools Hoan Tat] Da xuat {len(prev_list)} tool outputs vao: {prev_file} trong 0.0s")
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
            print(f"  [{idx:02d}] [SKIP] Tool '{tool_name}' (Da thuc thi)")
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

    # 3. Xuat ket qua
    out_file = out_path / "sample_tool_outputs.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(tool_outputs, fp, ensure_ascii=False, indent=2)

    print(f"✅ [Tools Hoan Tat] Da xuat {len(tool_outputs)} tool outputs vao: {out_file} (Moi: {new_count}, Bo qua: {skipped_count}) trong {total_elapsed}s")
    return tool_outputs


if __name__ == "__main__":
    run_auto_tools()
