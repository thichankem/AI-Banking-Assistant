"""
Automated LangSmith Tracing Collection Runner (Auto Tracing Node with Deduplication).
=====================================================================================
Features:
1. Reads trace inputs from configured tracing input file.
2. Computes SHA-256 hash of each trace request.
3. Checks StateTracker: Reuses previously traced requests (SKIP).
4. Records LangSmith runs and spans for new trace queries.
5. Emits results to configured tracing output file and flushes LangSmith client.

MANDATORY INVARIANTS:
1. NO __INIT__.PY.
2. NO TYPING LIBRARY.
3. ZERO TERNARY: Explicit if/else statements.
4. 0% REGEX LAW.
"""

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

from agent_banking.core.settings import Config
from agent_banking.state_tracker import StateTracker
from agent_banking.tracing.config.settings import TracingConfig, load_tracing_prompts
from agent_banking.tracing.tracer import BankingTracer


_runner_cfg = dict(TracingConfig.RUNNER_SETTINGS or {})
_prompts = load_tracing_prompts()
_runner_msgs = dict(getattr(_prompts, "RUNNER_MESSAGES", {}) or {})


def _msg(key, **kwargs):
    tpl = str(_runner_msgs.get(key) or "")
    if not tpl:
        return ""
    return tpl.format(**kwargs)


def run_auto_tracing(
    input_file=None,
    output_dir=None,
    force=False
):
    """Run automated LangSmith tracing collection with state deduplication."""
    cfg = Config()
    input_dir_name = str(_runner_cfg.get("input_dir_name") or "input")
    output_dir_name = str(_runner_cfg.get("output_dir_name") or "output")
    input_filename = str(_runner_cfg.get("input_filename") or "")
    output_filename = str(_runner_cfg.get("output_filename") or "")
    state_filename = str(_runner_cfg.get("state_filename") or "")
    default_session_id = str(_runner_cfg.get("default_session_id") or "")
    sim_parser_ms = float(_runner_cfg.get("simulated_parser_duration_ms") or 0.0)
    sim_embed_ms = float(_runner_cfg.get("simulated_embedding_duration_ms") or 0.0)
    sim_embed_dim = int(_runner_cfg.get("simulated_embedding_dimension") or 0)
    preview_chars = int(_runner_cfg.get("query_preview_chars") or 0)

    in_dir = PROJECT_ROOT / "agent_banking" / "tracing" / input_dir_name
    in_dir.mkdir(parents=True, exist_ok=True)

    if output_dir is not None:
        out_path = Path(output_dir)
    else:
        out_path = PROJECT_ROOT / "agent_banking" / "tracing" / output_dir_name
    out_path.mkdir(parents=True, exist_ok=True)

    state_file = out_path / state_filename
    tracker = StateTracker(state_file)
    if force:
        tracker.clear()

    # 1. Locate trace input source
    target_in_file = None
    if input_file is not None:
        target_in_file = Path(input_file)
    else:
        candidate = in_dir / input_filename
        if candidate.exists():
            target_in_file = candidate

    if target_in_file is None or not target_in_file.exists():
        print(_msg("missing_input_file", in_dir=in_dir))
        return []

    print(f"\n{'=' * 75}")
    print(_msg("banner_line_template", input_name=target_in_file.name, out_path=out_path))
    print(f"{'=' * 75}")

    with open(target_in_file, "r", encoding="utf-8-sig") as fp:
        raw_traces = json.load(fp)

    if not raw_traces:
        print(_msg("empty_input"))
        return []

    # 2. Load previous trace outputs for reuse
    prev_results = {}
    prev_file = out_path / output_filename
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
                    sess = item.get("session_id", default_session_id)
                    t_hash_map[tid] = tracker.compute_text_hash(f"{tid}:{q}:{sess}")
                if tracker.are_all_items_processed(t_hash_map):
                    print(_msg("skip_all_template", total=len(raw_traces)))
                    print(_msg("skip_all_done_template", total=len(prev_list), prev_file=prev_file))
                    return prev_list
        except Exception:
            prev_results = {}

    tracer = BankingTracer(config=cfg)
    trace_outputs = []
    skipped_count = 0
    new_count = 0
    start_t = time.time()

    for item in raw_traces:
        tid = item.get("trace_id", "tr_" + str(len(trace_outputs) + 1))
        q = item.get("query", "")
        sess = item.get("session_id", default_session_id)

        trace_hash = tracker.compute_text_hash(f"{tid}:{q}:{sess}")
        is_done = tracker.is_processed(tid, trace_hash)

        if is_done and tid in prev_results:
            cached_res = dict(prev_results[tid])
            trace_outputs.append(cached_res)
            skipped_count += 1
            print(_msg("skip_item_template", tid=tid, q_preview=q[:preview_chars]))
        else:
            t0 = time.time()
            with tracer.trace_query(query=q, session_id=sess) as trace:
                trace.record_query_parser(
                    input_query=q,
                    parsed=type("DummyParsed", (), {"normalized_query": q, "keywords": q.split()})(),
                    duration_ms=sim_parser_ms
                )
                trace.record_embedding(num_texts=1, dimension=sim_embed_dim, duration_ms=sim_embed_ms)
                trace.finish(answer=_msg("simulated_answer"), is_blocked=False)
                recorded_trace_id = trace.trace_id

            dt_ms = round((time.time() - t0) * 1000, 2)
            res = {
                "trace_id": tid,
                "langsmith_trace_id": recorded_trace_id or tid,
                "langsmith_project": tracer.project_name,
                "langsmith_enabled": tracer.is_langsmith_active,
                "session_id": sess,
                "query": q,
                "status": "RECORDED",
                "latency_ms": dt_ms,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            trace_outputs.append(res)
            tracker.record_processed(tid, trace_hash, {"status": "RECORDED"})
            new_count += 1
            print(_msg("new_item_template", tid=tid, q_preview=q[:preview_chars], dt_ms=dt_ms))

    tracer.flush()
    tracker.save()
    total_elapsed = round(time.time() - start_t, 2)

    # 3. Export trace outputs
    out_file = out_path / output_filename
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(trace_outputs, fp, ensure_ascii=False, indent=2)

    print(_msg("done_template", total=len(trace_outputs), out_file=out_file, new_count=new_count, skipped_count=skipped_count, elapsed=total_elapsed))
    return trace_outputs


if __name__ == "__main__":
    run_auto_tracing()
