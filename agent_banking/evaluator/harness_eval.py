"""Run a fresh, resumable cloud evaluation without touching historical benchmark caches."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
import os
import sys
from time import perf_counter

from agent_banking.evaluator.config.settings import (
    EvaluatorConfig,
    get_evaluator_messages,
    get_evaluator_prompt,
    resolve_repo_path,
)

GROUPS = EvaluatorConfig.DATASET_GROUPS
_DS_CFG = EvaluatorConfig.DATASETS
_HARNESS = EvaluatorConfig.HARNESS
_FEEDBACK = get_evaluator_messages("feedback")
_JUDGE_PROMPT = get_evaluator_prompt("HARNESS_JUDGE_PROMPT", "harness_judge.md")


def golden_cases(root):
    cases = []
    template = _DS_CFG["group_path_template"]
    for group in GROUPS:
        path = resolve_repo_path(template.replace("{group}", group), root=root)
        cases.extend({**case, "group": group, "key": f"{group}:{case['id']}"} for case in
                     json.loads(path.read_text(encoding="utf-8-sig")))
    expected_total = int(_DS_CFG["expected_total_cases"])
    if len(cases) != expected_total or len({case["key"] for case in cases}) != expected_total:
        raise ValueError(f"Expected {expected_total} distinct golden cases")
    return cases


def judge_answer(judge, case, observed):
    from agent_banking.agent.model_protocol import object_schema, text_schema

    schema = object_schema({
        "verdict": {"enum": ["pass", "fail"]},
        "feedback": text_schema(int(_HARNESS["judge_feedback_max_chars"]))
    }, ["verdict", "feedback"])
    ref_fields = tuple(_HARNESS["judge_reference_fields"])
    excluded_fields = tuple(_HARNESS["judge_excluded_turn_fields"])
    criteria = {key: case[key] for key in ref_fields if key in case}
    return judge.request(
        _JUDGE_PROMPT,
        {"reference": criteria, "observed": [
            {key: value for key, value in turn.items() if key not in excluded_fields} for turn in observed
        ]}, [schema], max_tokens=int(_HARNESS["judge_max_tokens"])
    )


def load_completed(path):
    if not path.exists():
        return {}
    records = {}
    contents = path.read_text(encoding="utf-8")
    lines = contents.splitlines()
    for pos, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if pos != len(lines) - 1:
                raise
            # Retain complete records if the process stopped during the last write.
            path.write_text("".join(item + "\n" for item in lines[:pos]), encoding="utf-8")
            break
        records[row["key"]] = row
    else:
        if contents and not contents.endswith("\n"):
            with path.open("a", encoding="utf-8") as stream:
                stream.write("\n")
    return records


def comparison(baseline_dir: Path, candidate_dir: Path) -> dict:
    identity_file = str(_HARNESS["identity_filename"])
    results_file = str(_HARNESS["results_filename"])
    baseline_identity = json.loads((baseline_dir / identity_file).read_text(encoding="utf-8"))
    candidate_identity = json.loads((candidate_dir / identity_file).read_text(encoding="utf-8"))
    identity_keys = tuple(_HARNESS["identity_keys"])
    if not baseline_identity.get("snapshot") or any(
        baseline_identity.get(key) != candidate_identity.get(key) for key in identity_keys
    ):
        raise ValueError("The runs must use the same model and knowledge snapshot")
    baseline = load_completed(baseline_dir / results_file)
    candidate = load_completed(candidate_dir / results_file)

    verdict_names = tuple(_HARNESS["verdict_names"])

    def count(records, keep):
        tallies = {name: 0 for name in verdict_names}
        for key, row in records.items():
            if keep(key, row):
                tallies[row["verdict"]] += 1
        return tallies

    base_golden = count(baseline, lambda key, row: not key.startswith("scenario:"))
    new_golden = count(candidate, lambda key, row: not key.startswith("scenario:"))
    scenario_groups = str(_HARNESS["scenario_groups"])
    groups = {letter: count(candidate, lambda key, row: key.startswith("scenario:") and row["group"] == letter)
              for letter in scenario_groups}
    expect_required = tuple(_HARNESS["mandatory_scenarios"])
    mandatory = all(candidate.get("scenario:" + name, {}).get("verdict") == "pass" for name in expect_required)
    expected_total = int(_DS_CFG["expected_total_cases"])
    cases_per_group = int(_HARNESS["cases_per_scenario_group"])
    min_pass_rate = float(_HARNESS["scenario_group_min_pass_rate"])
    complete = (sum(base_golden.values()) == sum(new_golden.values()) == expected_total
                and all(sum(group.values()) == cases_per_group for group in groups.values()))
    passed = complete and mandatory and all(
        group["pass"] / cases_per_group >= min_pass_rate for group in groups.values()
    ) and new_golden["pass"] >= base_golden["pass"]
    changes = [{"key": key, "baseline": baseline[key]["verdict"], "candidate": candidate[key]["verdict"],
                "baseline_trace_ids": trace_ids(baseline[key]), "candidate_trace_ids": trace_ids(candidate[key])}
               for key in sorted(baseline.keys() & candidate.keys()) if baseline[key]["verdict"] != candidate[key]["verdict"]]
    return {"complete": complete, "cloud_gate_passed": passed, "mandatory_cases_passed": mandatory,
            "baseline_785": base_golden, "candidate_785": new_golden, "candidate_a_to_f": groups,
            "snapshot": candidate_identity["snapshot"], "verdict_changes": changes,
            "baseline_metrics": routing_metrics(baseline_dir, baseline),
            "candidate_metrics": routing_metrics(candidate_dir, candidate)}


def trace_ids(row):
    return [turn["trace_id"] for turn in row.get("observed", []) if turn.get("trace_id")]


def routing_metrics(output, results):
    path = resolve_repo_path(_HARNESS["trace_relpath"], root=output)
    wanted = {trace_id for row in results.values() for trace_id in trace_ids(row)}
    latencies = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    routes = {}
    calls = 0
    usage_reports = 0
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.endswith("\n"):
                    continue
                trace = json.loads(line)
                if trace["trace_id"] not in wanted:
                    continue
                latencies.append(trace["duration_ms"])
                for span in trace["spans"]:
                    if span["name"] != "model.attempt":
                        continue
                    data = span["attributes"]
                    if data.get("result") == "skipped":
                        continue
                    calls += 1
                    route = data.get("configured_role", data["role"]) + "." + data["alias"]
                    routes[route] = routes.get(route, 0) + 1
                    reported = data.get("usage", {})
                    usage_reports += bool(reported)
                    usage["prompt_tokens"] += reported.get("prompt_tokens", reported.get("input_tokens", 0))
                    usage["completion_tokens"] += reported.get("completion_tokens", reported.get("output_tokens", 0))
                    usage["total_tokens"] += reported.get("total_tokens", 0)
    latencies.sort()

    def percentile(value):
        if latencies:
            return latencies[ceil(len(latencies) * value) - 1]
        return None

    if usage_reports:
        usage_out = usage
    else:
        usage_out = None
    return {"traced_requests": len(latencies), "p50_ms": percentile(float(_HARNESS["p50_percentile"])),
            "p95_ms": percentile(float(_HARNESS["p95_percentile"])),
            "model_calls": calls, "calls_with_usage": usage_reports,
            "usage": usage_out, "routes": routes}


def run(args):
    root = Path(__file__).resolve().parents[2]
    from agent_banking.core.environment import load_environment
    load_environment()
    if args.code_root:
        # The benchmark runner is shared. All runtime imports below come from the pinned code copy.
        package = sys.modules["agent_banking"]
        package.__path__ = [str(args.code_root.resolve() / "agent_banking")]
        for name in ("agent_banking.core.environment",):
            sys.modules.pop(name, None)

    from agent_banking.agent.model_protocol import StructuredModel
    from agent_banking.agent_pipeline import BankingAgentPipeline
    from agent_banking.core.chat import FallbackChatClient
    from agent_banking.core.settings import Config
    from agent_banking.pipeline import BankingRAGPipeline
    from agent_banking.tools.rag_tool import BankingRAGTool

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = Config()
    config.QDRANT_STORAGE_PATH = str(args.index.resolve())
    config.ABANK_QDRANT_STORAGE_PATH = config.QDRANT_STORAGE_PATH
    snapshot = args.index.resolve() / str(_HARNESS["snapshot_filename"])
    if snapshot.exists():
        snapshot_meta = json.loads(snapshot.read_text(encoding="utf-8"))
    else:
        snapshot_meta = {}
    config.KNOWLEDGE_PUBLICATION_SCOPE = args.scope or snapshot_meta.get("scope", "")
    config.STORAGE_DIR = root / "agent_banking" / "vector_db"
    config.TRACE_DIR = output / "traces"
    config.ENABLE_TRACING = True
    config.LOCAL_TRACE_EXPORT = True
    rag = BankingRAGPipeline(config=config)
    pipeline = BankingAgentPipeline(
        config=config,
        rag_pipeline=rag,
        sqlite_db_path=str(output / str(_HARNESS["session_sqlite_filename"])),
    )
    judge_client = FallbackChatClient(str(_HARNESS["judge_module"]), routed=bool(_HARNESS["judge_routed"]))
    judge = StructuredModel(judge_client, config.AGENT_LLM_MODEL)
    identity = {
        "version": args.version, "models": judge_client.fingerprint, "collection": config.QDRANT_COLLECTION,
        "scope": config.KNOWLEDGE_PUBLICATION_SCOPE,
        "snapshot": snapshot_meta,
    }
    from agent_banking.core import models
    if hasattr(models, "ROUTING_VERSION"):
        identity["routing"] = {"enabled": models.auto_routing_enabled(), "version": models.ROUTING_VERSION,
                               "chains": {role: models.chain_fingerprint(role) for role in models.ROUTED_MODULES}}
    identity_path = output / str(_HARNESS["identity_filename"])
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("Use a new output directory for a different runtime, model, or index")
    identity_path.write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.scenarios_only:
        cases = []
    else:
        cases = golden_cases(root)
    if args.scenarios:
        scenarios = json.loads(args.scenarios.read_text(encoding="utf-8"))
        cases = [{**case, "key": "scenario:" + case["id"]} for case in scenarios] + cases
    if args.limit:
        cases = cases[:args.limit]
    result_path = output / str(_HARNESS["results_filename"])
    completed = load_completed(result_path)
    run_nonce = os.urandom(int(_HARNESS["fixture_nonce_bytes"])).hex()
    fixture_state = ContextVar("scenario_fixture", default=None)
    retrieve = BankingRAGTool.execute

    def execute_search(tool, query, *positional, **keyword):
        fixture = fixture_state.get()
        if fixture is None:
            return retrieve(tool, query, *positional, **keyword)
        fixture["calls"] += 1
        if fixture["calls"] == 1 and fixture.get("temporary_first"):
            raise TimeoutError("Synthetic retrieval timeout")
        return {"status": "success", "query": query, "normalized_query": query,
                "documents": fixture["documents"]}

    BankingRAGTool.execute = execute_search

    observed_turn_fields = tuple(_HARNESS["observed_turn_fields"])
    observed_resume_fields = tuple(_HARNESS["observed_resume_fields"])
    session_prefix = str(_HARNESS["session_prefix"])
    session_hash_len = int(_HARNESS["session_hash_len"])

    def execute(case):
        started = perf_counter()
        key = case["key"]
        session = session_prefix + sha256((run_nonce + key).encode()).hexdigest()[:session_hash_len]
        observed = []
        fixture = case.get("fixture")
        if fixture is not None:
            fixture_val = {**fixture, "calls": 0}
        else:
            fixture_val = None
        token = fixture_state.set(fixture_val)
        try:
            for turn in case.get("turns", [case["question"]]):
                result = pipeline.chat(turn, session_id=session)
                observed.append({name: result[name] for name in observed_turn_fields if name in result})
                if result.get("audit_status") == "ERROR":
                    category = result.get("error_category")
                    verdict = "infrastructure_error"
                    if category == "model_contract":
                        verdict = "fail"
                    return {"key": key, "group": case["group"], "verdict": verdict,
                            "feedback": category or _FEEDBACK["harness_runtime_error"], "observed": observed}
                if result.get("status") == "awaiting_input":
                    # Use a scripted clarification only when supplied by the fixture.
                    reply = case.get("clarifications", {}).get(str(len(observed)))
                    if not reply:
                        break
                    result = pipeline.resume(session, result["interaction"]["id"], reply["choice_id"], reply.get("text"))
                    observed.append({name: result[name] for name in observed_resume_fields if name in result})
            checks = case.get("checks", {})
            calls = [call for turn in observed for call in turn.get("tool_calls", [])]
            answers = " ".join(turn.get("answer", "") for turn in observed)
            interaction_kinds = {turn["interaction"]["kind"] for turn in observed if turn.get("interaction")}
            if ((checks.get("no_tools") and calls)
                or (checks.get("interaction") and checks["interaction"] not in interaction_kinds)
                or (checks.get("retried_read") and not any(call.get("retried") for call in calls))
                or (checks.get("contains") and checks["contains"].casefold() not in answers.casefold())
                or (checks.get("citations") and not any(turn.get("citations") for turn in observed))):
                verdict = {"verdict": "fail", "feedback": _FEEDBACK["harness_missing_behavior"]}
            else:
                verdict = judge_answer(judge, case, observed)
            return {"key": key, "group": case["group"], **verdict, "observed": observed,
                    "seconds": round(perf_counter() - started, 2)}
        except Exception as error:
            return {"key": key, "group": case["group"], "verdict": "infrastructure_error",
                    "feedback": type(error).__name__, "observed": observed}
        finally:
            fixture_state.reset(token)

    remaining = [case for case in cases if case["key"] not in completed or (
        args.retry_errors and completed[case["key"]]["verdict"] == "infrastructure_error"
    )]
    try:
        with result_path.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=args.workers) as workers:
            jobs = [workers.submit(execute, case) for case in remaining]
            for job in as_completed(jobs):
                row = job.result()
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                completed[row["key"]] = row
                print(f"{len(completed)}/{len(cases)} {row['key']} {row['verdict']}", flush=True)
    finally:
        BankingRAGTool.execute = retrieve
        pipeline.graph.checkpointer.conn.close()
        rag.close()

    summary = {}
    for case in cases:
        verdict = completed.get(case["key"], {}).get("verdict", "pending")
        group = summary.setdefault(case["group"], {})
        group[verdict] = group.get(verdict, 0) + 1
    (output / str(_HARNESS["summary_filename"])).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = routing_metrics(output, {case["key"]: completed[case["key"]] for case in cases if case["key"] in completed})
    (output / str(_HARNESS["metrics_filename"])).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BASELINE", "CANDIDATE"))
    parser.add_argument("--index", type=Path)
    parser.add_argument("--scope", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--version")
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--scenarios", type=Path)
    parser.add_argument("--scenarios-only", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        choices=range(int(_HARNESS["workers_min"]), int(_HARNESS["workers_max"]) + 1),
        default=int(_HARNESS["workers_default"]),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    if args.compare:
        print(json.dumps(comparison(*args.compare), ensure_ascii=False, indent=2))
    else:
        if not (args.index and args.output and args.version):
            parser.error("--index, --output and --version are required for an evaluation")
        if args.scenarios_only and not args.scenarios:
            parser.error("--scenarios-only requires --scenarios")
        run(args)
