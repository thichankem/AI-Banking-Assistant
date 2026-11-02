"""
Module Tu Dong Phong Ve Banking Guardrails (Auto Guardrails Node with Deduplication).
=====================================================================================
Dac diem:
1. Doc cau hoi tu guardrails/input/sample_queries.json (hoac queries.json).
2. Tinh toan SHA-256 hash cua tung cau hoi.
3. Kiem tra StateTracker: Cau hoi nao da kiem dinh thi tai su dung (SKIP).
4. Chi thuc hien kiem tra Input Safety Guardrail cho queries moi.
5. Xuat ket qua: sample_guardrail_outputs.json.

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

from agent_banking.config import Config
from agent_banking.guardrails.rag_guardrails import BankingGuardrailManager
from agent_banking.state_tracker import StateTracker


def run_auto_guardrails(
    input_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    force: bool = False
) -> list[dict]:
    """
    Tu dong kiem dinh an toan guardrails tu input sang output kem co che chong trung lap.
    """
    cfg = Config()
    in_dir = PROJECT_ROOT / "agent_banking" / "guardrails" / "input"
    in_dir.mkdir(parents=True, exist_ok=True)

    if output_dir is not None:
        out_path = Path(output_dir)
    else:
        out_path = PROJECT_ROOT / "agent_banking" / "guardrails" / "output"
    out_path.mkdir(parents=True, exist_ok=True)

    state_file = out_path / ".state_tracker.json"
    tracker = StateTracker(state_file)
    if force:
        tracker.clear()

    # 1. Tim nguon queries
    target_in_file = None
    if input_file is not None:
        target_in_file = Path(input_file)
    else:
        candidates = [
            in_dir / "sample_queries.json",
            in_dir / "queries.json"
        ]
        for c in candidates:
            if c.exists():
                target_in_file = c
                break

    if target_in_file is None or not target_in_file.exists():
        print(f"⚠️ Khong tim thay tap tin input queries trong: {in_dir}")
        return []

    print(f"\n{'='*75}")
    print(f"▶ [MODULE 8: GUARDRAILS] Input: {target_in_file.name} -> Output: {out_path}")
    print(f"{'='*75}")

    with open(target_in_file, "r", encoding="utf-8") as fp:
        raw_items = json.load(fp)

    if not raw_items:
        print("⚠️ Danh sach queries rong.")
        return []

    # 2. Doc ket qua cu de tai su dung
    prev_results = {}
    prev_file = out_path / "sample_guardrail_outputs.json"
    if prev_file.exists() and not force:
        try:
            with open(prev_file, "r", encoding="utf-8") as fp:
                prev_list = json.load(fp)
                for item in prev_list:
                    q = item.get("query", "")
                    if q:
                        prev_results[q] = item
            if len(prev_results) >= len(raw_items) and len(raw_items) > 0:
                q_hash_map = {}
                for item in raw_items:
                    if isinstance(item, str):
                        qt = item
                    else:
                        qt = item.get("query", "")
                    q_hash_map[qt] = tracker.compute_text_hash(qt)
                if tracker.are_all_items_processed(q_hash_map):
                    print(f"  ⏭️ [SKIP ALL] Tat ca {len(raw_items)} queries da duoc kiem dinh Guardrails truoc do. Bo qua toan bo trong 0.0s!")
                    print(f"✅ [Guardrails Hoan Tat] Da xuat {len(prev_list)} ket qua vao: {prev_file} trong 0.0s")
                    return prev_list
        except Exception:
            prev_results = {}

    manager = BankingGuardrailManager()
    guardrail_outputs: list[dict] = []
    skipped_count = 0
    new_count = 0
    start_t = time.time()

    for item in raw_items:
        if isinstance(item, str):
            q_text = item
            q_id = "q_" + str(len(guardrail_outputs) + 1)
            desc = ""
        else:
            q_text = item.get("query", "")
            q_id = item.get("id", "q_" + str(len(guardrail_outputs) + 1))
            desc = item.get("description", "")

        q_hash = tracker.compute_text_hash(q_text)
        is_done = tracker.is_processed(q_text, q_hash)

        if is_done and q_text in prev_results:
            cached_res = dict(prev_results[q_text])
            cached_res["id"] = q_id
            guardrail_outputs.append(cached_res)
            skipped_count += 1
            print(f"  [{q_id}] [SKIP] '{q_text[:35]}...' (Da kiem dinh)")
        else:
            t0 = time.time()
            guard_res = manager.guard_input(q_text)
            dt_ms = round((time.time() - t0) * 1000, 2)

            res = {
                "id": q_id,
                "query": q_text,
                "is_safe": guard_res.is_safe,
                "trigger_reason": guard_res.trigger_reason,
                "refusal_message": guard_res.refusal_message,
                "sanitized_query": guard_res.sanitized_query,
                "latency_ms": dt_ms,
                "description": desc
            }
            guardrail_outputs.append(res)
            tracker.record_processed(q_text, q_hash, {"is_safe": guard_res.is_safe})
            new_count += 1
            status_label = "SAFE"
            if not guard_res.is_safe:
                status_label = f"BLOCKED: {guard_res.trigger_reason}"
            print(f"  [{q_id}] [NEW]  '{q_text[:35]}...' -> {status_label} ({dt_ms}ms)")

    tracker.save()
    total_elapsed = round(time.time() - start_t, 2)

    # 3. Xuat ket qua
    out_file = out_path / "sample_guardrail_outputs.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(guardrail_outputs, fp, ensure_ascii=False, indent=2)

    print(f"✅ [Guardrails Hoan Tat] Da xuat {len(guardrail_outputs)} ket qua vao: {out_file} (Moi: {new_count}, Bo qua: {skipped_count}) trong {total_elapsed}s")
    return guardrail_outputs


if __name__ == "__main__":
    run_auto_guardrails()
