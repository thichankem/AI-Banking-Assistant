"""Reproducible, isolated evaluation; reference answers only reach the judge."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import sys
import time
from uuid import uuid4

from agent_banking.evaluator.config.settings import EvaluatorConfig, resolve_repo_path

_DS_CFG = EvaluatorConfig.DATASETS
_OUT_CFG = EvaluatorConfig.CACHES_AND_OUTPUTS
_TE_CFG = EvaluatorConfig.TUNING_EVAL


def load_records(path, *, repair_tail=False):
    records = {}
    if path.exists():
        data = path.read_bytes()
        lines = data.splitlines(keepends=True)
        valid_end = 0
        for index, line in enumerate(lines):
            if line.strip():
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    if any(remaining.strip() for remaining in lines[index + 1:]):
                        raise ValueError("Invalid JSONL record before the final interrupted row: " + str(path)) from None
                    if repair_tail:
                        tail = data[valid_end:]
                        backup = path.with_name(path.name + ".interrupted-" + sha256(tail).hexdigest()[:12] + ".bin")
                        if not backup.exists():
                            backup.write_bytes(tail)
                        temporary = path.with_name(path.name + ".recovered.tmp")
                        temporary.write_bytes(data[:valid_end])
                        temporary.replace(path)
                    print("Ignored interrupted final JSONL row: " + str(path), file=sys.stderr)
                    break
                records[record["id"]] = record
            valid_end += len(line)
        if repair_tail and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
            with path.open("ab") as stream:
                stream.write(b"\n")
    return records


def select_cases(root, limit, seed):
    cases = []
    template = _DS_CFG["group_path_template"]
    for group in EvaluatorConfig.DATASET_GROUPS:
        path = resolve_repo_path(template.replace("{group}", group), root=root)
        items = json.loads(path.read_text(encoding="utf-8"))
        random.Random(seed).shuffle(items)
        for item in items:
            item["group"] = group
        cases.append(items)
    # Interleaving keeps every prefix balanced across the four groups.
    selected = []
    for index in range(max(map(len, cases))):
        for items in cases:
            if index < len(items):
                selected.append(items[index])
    return selected[:limit]


def summarize(records, baseline):
    groups = {}
    for record in records.values():
        groups.setdefault(record["group"], Counter())[record["status"]] += 1
    paired = Counter()
    for cid, record in records.items():
        if cid in baseline:
            paired[baseline[cid]["status"] + " -> " + record["status"]] += 1
    return {"total": len(records), "status": dict(Counter(r["status"] for r in records.values())),
            "groups": {key: dict(value) for key, value in groups.items()},
            "paired": dict(paired), "mean_seconds": round(sum(r["seconds"] for r in records.values()) / max(1, len(records)), 2)}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path(str(_OUT_CFG["default_tuning_baseline_relpath"])))
    parser.add_argument("--limit", type=int, default=int(_TE_CFG["default_limit"]))
    parser.add_argument("--seed", type=int, default=int(_TE_CFG["default_seed"]))
    parser.add_argument("--workers", type=int, default=int(_TE_CFG["default_workers"]))
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--priority-ids", nargs="*", default=[],
                        help="Evaluate these cases first while keeping the complete selection.")
    parser.add_argument("--index", type=Path)
    parser.add_argument("--reference-corrections", type=Path,
                        help="Apply source-verified benchmark references only when grading.")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    from langgraph.checkpoint.memory import MemorySaver
    from agent_banking.agent_pipeline import BankingAgentPipeline
    from agent_banking.core.chat import FallbackChatClient
    from agent_banking.core.settings import Config
    from agent_banking.core.models import chain_fingerprint, get_model_chain
    from agent_banking.agent import deep
    from agent_banking.evaluator.master_785_runner import call_llm_judge

    root = Path(__file__).resolve().parents[2]
    config = Config()
    index = args.index.resolve() if args.index else Path(config.ABANK_QDRANT_STORAGE_PATH or config.QDRANT_STORAGE_PATH)
    if not index.is_absolute():
        index = root / index
    inventory = index / "knowledge_inventory.json"
    cases = select_cases(root, args.limit, args.seed)
    if args.ids:
        total_cases = int(_DS_CFG["expected_total_cases"])
        cases = [case for case in select_cases(root, total_cases, args.seed) if case["id"] in args.ids]
    if args.priority_ids:
        by_id = {case["id"]: case for case in cases}
        if not set(args.priority_ids).issubset(by_id):
            parser.error("Every priority ID must be part of the selected cases")
        priority = list(dict.fromkeys(args.priority_ids))
        cases = [by_id[cid] for cid in priority] + [case for case in cases if case["id"] not in priority]
    if args.reference_corrections:
        from agent_banking.evaluator.reference_alignment import validate_corrections

        corrections = validate_corrections(args.reference_corrections,
                                           select_cases(root, int(_DS_CFG["expected_total_cases"]), args.seed),
                                           json.loads(inventory.read_bytes()), config.CHUNKING_INPUT_DIR)
        for case in cases:
            if case["id"] in corrections:
                correction = corrections[case["id"]]
                case["reference_answer"] = correction["reference_answer"]
                case["rubric"] = {**case["rubric"], "must_include": correction["must_include"]}
    code_paths = list(EvaluatorConfig.TUNING_CODE_PATHS)
    package = Path(__file__).resolve().parents[1]
    code_paths.extend(name for name in (
        "tools/extensions/legal_web_search.py", "tools/search_legal_web/tool.yaml", "reranking/rag_reranker.py",
        "tools/config/settings/query_rules.yaml", "tools/tool_registry.py",
        "agent/config/prompts/deep/base.md", "agent/config/prompts/deep/coordinator.md",
        "agent/config/prompts/deep/review.md", "agent/config/prompts/deep/partial_recovery.md",
        "agent/config/prompts/messages.yaml", "evaluator/config/prompts/master_785_judge.md",
    ) if name not in code_paths and (package / name).is_file())
    template = _DS_CFG["group_path_template"]
    metadata = {"case_ids": [case["id"] for case in cases], "seed": args.seed,
                "code": {name: sha256((package / name).read_bytes()).hexdigest() for name in code_paths},
                "runtime_model_chain": chain_fingerprint("agent"),
                "runtime_review_model_chain": chain_fingerprint("agent_review"),
                "judge_model_chain": chain_fingerprint(str(_TE_CFG["judge_module"]), routed=bool(_TE_CFG["judge_routed"])),
                "runtime_models": [candidate.model for candidate in get_model_chain("agent")],
                "judge_models": [candidate.model for candidate in get_model_chain(
                    str(_TE_CFG["judge_module"]), routed=bool(_TE_CFG["judge_routed"])
                )],
                "agent_context_bytes": config.AGENT_CONTEXT_BYTES,
                "cloud_requests_per_minute": float(os.getenv("CLOUD_REQUESTS_PER_MINUTE", str(_TE_CFG["default_cloud_rpm"]))),
                "hybrid_candidate_k": Config.HYBRID_CANDIDATE_K,
                "datasets": {group: sha256(resolve_repo_path(template.replace("{group}", group), root=root).read_bytes()).hexdigest()
                             for group in EvaluatorConfig.DATASET_GROUPS}}
    if inventory.is_file():
        metadata["index_inventory_sha256"] = sha256(inventory.read_bytes()).hexdigest()
    if args.reference_corrections:
        metadata["reference_corrections_sha256"] = sha256(args.reference_corrections.read_bytes()).hexdigest()
    metadata_path = args.output.with_suffix(".metadata.json")
    if metadata_path.exists() and json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Code, case selection or corpus changed; choose a fresh output path")
    records = load_records(args.output, repair_tail=True)
    baseline = load_records(args.baseline)
    pending = [case for case in cases if case["id"] not in records or args.retry_errors
               and records[case["id"]]["status"] in ("ERROR", "JUDGE_ERROR")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if not pending:
        summary = summarize(records, baseline)
        args.output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.index:
        config.ABANK_QDRANT_STORAGE_PATH = str(index)
        config.KNOWLEDGE_PUBLICATION_SCOPE = ""
    checkpointer = MemorySaver()
    pipeline = BankingAgentPipeline(config=config, checkpointer=checkpointer)
    if inventory.is_file():
        expected_chunks = int(json.loads(inventory.read_bytes())["chunks"])
        actual_chunks = pipeline.rag_pipeline.vector_store.count()
        if actual_chunks != expected_chunks:
            pipeline.rag_pipeline.close()
            raise RuntimeError(f"Benchmark index opened with {actual_chunks} chunks; expected {expected_chunks}.")
    judge = FallbackChatClient(module=str(_TE_CFG["judge_module"]), routed=bool(_TE_CFG["judge_routed"]))
    run_id = uuid4().hex
    review_errors = ContextVar("tuning_review_errors", default=None)
    original_validate = deep.validate_review

    def diagnose_review(*values):
        try:
            return original_validate(*values)
        except ValueError as error:
            errors = review_errors.get()
            if errors is not None:
                errors.append(str(error))
            raise

    deep.validate_review = diagnose_review

    judge_retries = int(_TE_CFG["judge_retries"])
    error_prefixes = tuple(_TE_CFG["judge_error_prefixes"])

    def evaluate(case):
        started = time.perf_counter()
        result = {"id": case["id"], "group": case["group"], "status": "ERROR"}
        errors = []
        token = review_errors.set(errors)
        try:
            previous = records.get(case["id"], {})
            if args.retry_errors and previous.get("status") == "JUDGE_ERROR" and previous.get("response"):
                response = previous["response"]
                result["reused_agent_response"] = True
                errors.extend(previous.get("review_validation_errors", []))
            else:
                session_id = "tuning_" + run_id + case["id"]
                try:
                    response = pipeline.chat(case["question"], session_id=session_id)
                finally:
                    # Each benchmark question is an independent, single-turn session.
                    checkpointer.delete_thread(session_id)
            if args.retry_errors and previous:
                result["retry_of_status"] = previous["status"]
            result["response"] = response
            if response.get("status") == "awaiting_input":
                result["status"] = "AWAITING_INPUT"
            elif response.get("audit_status") == "ERROR":
                result["status"] = "ERROR"
            else:
                for attempt in range(judge_retries):
                    score = call_llm_judge(judge, case["question"], case["reference_answer"],
                                           case.get("rubric", {}).get("must_include", []), response.get("answer", ""))
                    if not score.get("feedback", "").startswith(error_prefixes):
                        break
                    if attempt + 1 < judge_retries:
                        time.sleep(2 ** (attempt + 1))
                result["judge"] = score
                if score.get("feedback", "").startswith(error_prefixes):
                    result["status"] = "JUDGE_ERROR"
                else:
                    result["status"] = score["verdict"]
        except Exception as error:
            result["error"] = type(error).__name__ + ": " + str(error)
        finally:
            review_errors.reset(token)
        result["review_validation_errors"] = errors
        result["seconds"] = round(time.perf_counter() - started, 2)
        return result

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(evaluate, case) for case in pending]
            for future in as_completed(futures):
                result = future.result()
                records[result["id"]] = result
                with args.output.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
                print(result["id"], result["status"], result["seconds"],
                      result.get("response", {}).get("review_outcomes", []), flush=True)
                summary = summarize(records, baseline)
                args.output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summarize(records, baseline), ensure_ascii=False, indent=2))
    finally:
        deep.validate_review = original_validate
        pipeline.rag_pipeline.close()


if __name__ == "__main__":
    main()
