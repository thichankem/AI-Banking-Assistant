"""
Automated Business Skills Processing Node with Deduplication.
============================================================================
Features:
1. Reads requests from skills/input/sample_skill_inputs.json.
2. Computes SHA-256 hash of each skill_id and parameter set.
3. Checks StateTracker: Reuses previously executed skills (SKIP).
4. Invokes SkillRegistry for new skill executions.
5. Exports results to sample_skill_outputs.json.

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
from agent_banking.skills.skill_registry import SkillRegistry
from agent_banking.state_tracker import StateTracker


def run_auto_skills(
    input_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    force: bool = False
) -> list[dict]:
    """
    Automatically process skills from input to output with deduplication.
    """
    cfg = Config()
    in_dir = PROJECT_ROOT / "agent_banking" / "skills" / "input"
    in_dir.mkdir(parents=True, exist_ok=True)

    if output_dir is not None:
        out_path = Path(output_dir)
    else:
        out_path = PROJECT_ROOT / "agent_banking" / "skills" / "output"
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
        candidate = in_dir / "sample_skill_inputs.json"
        if candidate.exists():
            target_in_file = candidate

    if target_in_file is None or not target_in_file.exists():
        print(f"⚠️ Skill inputs file not found in: {in_dir}")
        return []

    print(f"\n{'='*75}")
    print(f"▶ [MODULE 11: SKILLS] Input: {target_in_file.name} -> Output: {out_path}")
    print(f"{'='*75}")

    with open(target_in_file, "r", encoding="utf-8") as fp:
        raw_items = json.load(fp)

    if not raw_items:
        print("⚠️ Skill inputs list is empty.")
        return []

    # 2. Read existing results for deduplication
    prev_results = {}
    prev_file = out_path / "sample_skill_outputs.json"
    if prev_file.exists() and not force:
        try:
            with open(prev_file, "r", encoding="utf-8") as fp:
                prev_list = json.load(fp)
                for item in prev_list:
                    sk_id = item.get("skill_id", "")
                    if sk_id:
                        prev_results[sk_id] = item
            if len(prev_results) >= len(raw_items) and len(raw_items) > 0:
                s_hash_map = {}
                for item in raw_items:
                    sk_id = item.get("skill_id", "")
                    params = item.get("params", {})
                    p_str = json.dumps(params, sort_keys=True, ensure_ascii=False)
                    s_hash_map[sk_id] = tracker.compute_text_hash(f"{sk_id}:{p_str}")
                if tracker.are_all_items_processed(s_hash_map):
                    print(f"  ⏭️ [SKIP ALL] All {len(raw_items)} skills previously completed. Skipped in 0.0s.")
                    print(f"✅ [Skills Complete] Exported {len(prev_list)} skill outputs to: {prev_file} in 0.0s")
                    return prev_list
        except Exception:
            prev_results = {}

    registry = SkillRegistry(config=cfg)
    skill_outputs: list[dict] = []
    skipped_count = 0
    new_count = 0
    start_t = time.time()

    for idx, item in enumerate(raw_items, start=1):
        skill_id = item.get("skill_id", "")
        params = item.get("params", {})
        p_str = json.dumps(params, sort_keys=True, ensure_ascii=False)
        req_hash = tracker.compute_text_hash(f"{skill_id}:{p_str}")

        is_done = tracker.is_processed(skill_id, req_hash)

        if is_done and skill_id in prev_results:
            cached_res = dict(prev_results[skill_id])
            skill_outputs.append(cached_res)
            skipped_count += 1
            print(f"  [{idx:02d}] [SKIP] Skill '{skill_id}' (Already processed)")
        else:
            t0 = time.time()
            skill = registry.get_skill(skill_id)
            if skill is not None:
                execution_res = skill.execute(params)
                dt_ms = round((time.time() - t0) * 1000, 2)
                res = {
                    "skill_id": skill_id,
                    "status": execution_res.get("status", "SUCCESS"),
                    "result": execution_res.get("data", {}),
                    "message": execution_res.get("message", execution_res.get("error", "OK")),
                    "latency_ms": dt_ms
                }
            else:
                dt_ms = round((time.time() - t0) * 1000, 2)
                res = {
                    "skill_id": skill_id,
                    "status": "SKILL_NOT_FOUND",
                    "result": {},
                    "message": f"Skill '{skill_id}' not found in SkillRegistry.",
                    "latency_ms": dt_ms
                }

            skill_outputs.append(res)
            tracker.record_processed(skill_id, req_hash, {"status": res["status"]})
            new_count += 1
            print(f"  [{idx:02d}] [NEW]  Skill '{skill_id}' -> {res['status']} ({dt_ms}ms)")

    tracker.save()
    total_elapsed = round(time.time() - start_t, 2)

    # 3. Export output
    out_file = out_path / "sample_skill_outputs.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(skill_outputs, fp, ensure_ascii=False, indent=2)

    print(f"✅ [Skills Complete] Exported {len(skill_outputs)} skill outputs to: {out_file} (New: {new_count}, Skipped: {skipped_count}) in {total_elapsed}s")
    return skill_outputs


if __name__ == "__main__":
    run_auto_skills()
