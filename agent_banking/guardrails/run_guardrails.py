"""
Automated Banking Guardrails Runner (Auto Guardrails Node with Deduplication).
==============================================================================
Features:
1. Reads queries from configured guardrails input file.
2. Computes SHA-256 hash of each query.
3. Checks StateTracker: Reuses previously audited queries (SKIP).
4. Executes Input Safety Guardrail for new queries only.
5. Emits results to configured guardrails output file.

MANDATORY INVARIANTS:
1. NO __INIT__.PY.
3. ZERO TERNARY: Explicit if/else statements.
4. 0% REGEX LAW.
"""

import concurrent.futures
import json
from pathlib import Path
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.guardrails.config.settings import GuardrailsConfig, load_guardrail_prompts
from agent_banking.guardrails.rag_guardrails import BankingGuardrailManager
from agent_banking.state_tracker import StateTracker


_runner_cfg = dict(GuardrailsConfig.RUNNER_SETTINGS or {})
_prompts = load_guardrail_prompts()
_runner_msgs = dict(getattr(_prompts, "RUNNER_MESSAGES", {}) or {})


def _msg(key, **kwargs):
    template = str(_runner_msgs.get(key) or "")
    if not template:
        return ""
    return template.format(**kwargs)


def run_auto_guardrails(
    input_file=None,
    output_dir=None,
    force=False,
    max_workers=None,
    use_threading=None,
    use_concurrency=None,
    use_decision_model=None
):
    """Run automated guardrails input->output evaluation with deduplication and thread pool."""
    if max_workers is None:
        max_workers = int(_runner_cfg.get("max_workers", 0))
    if use_threading is None:
        use_threading = bool(_runner_cfg.get("use_threading", False))
    if use_decision_model is None:
        use_decision_model = bool(_runner_cfg.get("use_decision_model", False))
    if use_concurrency is not None:
        use_threading = use_concurrency

    input_dir_name = str(_runner_cfg.get("input_dir_name") or "input")
    output_dir_name = str(_runner_cfg.get("output_dir_name") or "output")
    input_candidates = list(_runner_cfg.get("input_candidates") or [])
    output_filename = str(_runner_cfg.get("output_filename") or "")
    state_filename = str(_runner_cfg.get("state_filename") or "")
    preview_chars = int(_runner_cfg.get("query_preview_chars", 0))

    in_dir = PROJECT_ROOT / "agent_banking" / "guardrails" / input_dir_name
    in_dir.mkdir(parents=True, exist_ok=True)

    if output_dir is not None:
        out_path = Path(output_dir)
    else:
        out_path = PROJECT_ROOT / "agent_banking" / "guardrails" / output_dir_name
    out_path.mkdir(parents=True, exist_ok=True)

    state_file = out_path / state_filename
    tracker = StateTracker(state_file)
    if force:
        tracker.clear()

    # 1. Locate query input source
    raw_items = None
    target_in_file = None
    if input_file is not None:
        if isinstance(input_file, list):
            raw_items = input_file
        else:
            target_in_file = Path(input_file)
    else:
        for cand_name in input_candidates:
            c = in_dir / cand_name
            if c.exists():
                target_in_file = c
                break

    if raw_items is None:
        if target_in_file is None or not target_in_file.exists():
            print(_msg("missing_input_file", in_dir=in_dir))
            return []

        with open(target_in_file, "r", encoding="utf-8-sig") as fp:
            raw_items = json.load(fp)

    if not raw_items:
        print(_msg("empty_queries"))
        return []

    if target_in_file is not None:
        input_desc = target_in_file.name
    else:
        input_desc = f"<in-memory list: {len(raw_items)} items>"

    print(f"\n{'=' * 75}")
    print(_msg("banner_line_template", input_desc=input_desc, out_path=out_path))
    print(f"{'=' * 75}")

    # 2. Load cached results if available
    prev_results = {}
    prev_file = out_path / output_filename
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
                    print(_msg("skip_all_template", total=len(raw_items)))
                    print(_msg("skip_all_done_template", total=len(prev_list), prev_file=prev_file))
                    return prev_list
        except Exception:
            prev_results = {}

    manager = BankingGuardrailManager(use_decision_model=use_decision_model)
    if hasattr(manager, "semantic_guardrail"):
        if hasattr(manager.semantic_guardrail, "_ensure_initialized"):
            manager.semantic_guardrail._ensure_initialized()

    guardrail_outputs = [None] * len(raw_items)
    skipped_count = 0
    new_count = 0
    start_t = time.time()

    # 3. Classify queries into cached reuse (SKIP) and pending evaluation (NEW)
    items_to_process = []
    for idx, item in enumerate(raw_items):
        if isinstance(item, str):
            q_text = item
            q_id = "q_" + str(idx + 1)
            desc = ""
        else:
            q_text = item.get("query", "")
            q_id = item.get("id", "q_" + str(idx + 1))
            desc = item.get("description", "")

        q_hash = tracker.compute_text_hash(q_text)
        is_done = tracker.is_processed(q_text, q_hash)

        if is_done and q_text in prev_results:
            cached_res = dict(prev_results[q_text])
            cached_res["id"] = q_id
            guardrail_outputs[idx] = cached_res
            skipped_count += 1
            print(_msg("skip_item_template", q_id=q_id, q_preview=q_text[:preview_chars]))
        else:
            items_to_process.append((idx, q_id, q_text, q_hash, desc))

    # 4. Execute guardrail verification on pending queries
    if items_to_process:
        should_use_pool = False
        if use_threading:
            if max_workers > 1:
                if len(items_to_process) > 1:
                    should_use_pool = True

        if should_use_pool:
            workers = min(max_workers, len(items_to_process))

            def _evaluate_item(item_tuple):
                it_idx, it_qid, it_qtext, it_qhash, it_desc = item_tuple
                t0 = time.time()
                guard_res = manager.guard_input(it_qtext, use_decision_model=use_decision_model)
                dt_ms = round((time.time() - t0) * 1000, 2)
                item_res = {
                    "id": it_qid,
                    "query": it_qtext,
                    "is_safe": guard_res.is_safe,
                    "trigger_reason": guard_res.trigger_reason,
                    "refusal_message": guard_res.refusal_message,
                    "sanitized_query": guard_res.sanitized_query,
                    "latency_ms": dt_ms,
                    "description": it_desc
                }
                return (it_idx, it_qid, it_qtext, it_qhash, guard_res, item_res, dt_ms)

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_item = {}
                for it in items_to_process:
                    f = executor.submit(_evaluate_item, it)
                    future_to_item[f] = it

                for future in concurrent.futures.as_completed(future_to_item):
                    it_idx, it_qid, it_qtext, it_qhash, guard_res, item_res, dt_ms = future.result()
                    guardrail_outputs[it_idx] = item_res
                    tracker.record_processed(it_qtext, it_qhash, {"is_safe": guard_res.is_safe})
                    new_count += 1
                    if not guard_res.is_safe:
                        status_label = f"BLOCKED: {guard_res.trigger_reason}"
                    else:
                        status_label = "SAFE"
                    print(_msg("parallel_item_template", q_id=it_qid, q_preview=it_qtext[:preview_chars], status_label=status_label, dt_ms=dt_ms))
        else:
            for it in items_to_process:
                it_idx, it_qid, it_qtext, it_qhash, it_desc = it
                t0 = time.time()
                guard_res = manager.guard_input(it_qtext, use_decision_model=use_decision_model)
                dt_ms = round((time.time() - t0) * 1000, 2)
                item_res = {
                    "id": it_qid,
                    "query": it_qtext,
                    "is_safe": guard_res.is_safe,
                    "trigger_reason": guard_res.trigger_reason,
                    "refusal_message": guard_res.refusal_message,
                    "sanitized_query": guard_res.sanitized_query,
                    "latency_ms": dt_ms,
                    "description": it_desc
                }
                guardrail_outputs[it_idx] = item_res
                tracker.record_processed(it_qtext, it_qhash, {"is_safe": guard_res.is_safe})
                new_count += 1
                if not guard_res.is_safe:
                    status_label = f"BLOCKED: {guard_res.trigger_reason}"
                else:
                    status_label = "SAFE"
                print(_msg("serial_item_template", q_id=it_qid, q_preview=it_qtext[:preview_chars], status_label=status_label, dt_ms=dt_ms))

    tracker.save()
    total_elapsed = round(time.time() - start_t, 2)

    # 5. Export results
    out_file = out_path / output_filename
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(guardrail_outputs, fp, ensure_ascii=False, indent=2)

    print(_msg("done_template", total=len(guardrail_outputs), out_file=out_file, new_count=new_count, skipped_count=skipped_count, elapsed=total_elapsed))
    return guardrail_outputs


if __name__ == "__main__":
    run_auto_guardrails()
