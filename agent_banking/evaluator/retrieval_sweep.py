"""Measure candidate-pool sizes without exposing golden labels to retrieval."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

from agent_banking.evaluator.config.settings import EvaluatorConfig

_RS_CFG = EvaluatorConfig.RETRIEVAL_SWEEP


def basename(value):
    return str(value).replace("\\", "/").rsplit("/", 1)[-1].casefold()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=int(_RS_CFG["default_limit"]))
    parser.add_argument("--seed", type=int, default=int(_RS_CFG["default_seed"]))
    parser.add_argument("--workers", type=int, default=int(_RS_CFG["default_workers"]))
    parser.add_argument("--pools", nargs="+", type=int, default=list(_RS_CFG["default_pools"]))
    args = parser.parse_args()
    from agent_banking.core.settings import Config
    from agent_banking.core.cloud_rate_limit import acquire_cloud_slot
    from agent_banking.pipeline import BankingRAGPipeline
    from agent_banking.evaluator.tuning_eval import select_cases, load_records

    config = Config()
    config.ABANK_QDRANT_STORAGE_PATH = str(args.index.resolve())
    config.KNOWLEDGE_PUBLICATION_SCOPE = ""
    pipeline = BankingRAGPipeline(config=config)
    cases = select_cases(Path(__file__).resolve().parents[2], args.limit, args.seed)
    available = {basename(chunk["source"]) for chunk in pipeline.vector_store._bm25_corpus}
    records = load_records(args.output, repair_tail=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rerank_top_k = int(_RS_CFG["rerank_top_k"])
    expected_method = str(_RS_CFG["expected_rerank_method"])

    def evaluate(case):
        started = time.perf_counter()
        parsed = pipeline.parse_query(case["question"])
        targets = {basename(case["document_title"])}
        targets.update(basename(item["source_path"]) for item in case.get("evidence", []) if item.get("source_path"))
        scores = {}
        methods = set()
        for count in args.pools:
            pool_started = time.perf_counter()
            candidates = pipeline.retriever.retrieve(parsed, top_k=count, candidate_k=count, filter_dict={})
            acquire_cloud_slot(pipeline.reranker.model_name, pipeline.reranker.rerank_url)
            selected = pipeline.reranker.rerank(case["question"], candidates, top_k=rerank_top_k)
            methods.update(item.get("rerank_method", "unknown") for item in selected)
            names = [basename(item.get("chunk", item).get("source")) for item in selected]
            scores[str(count)] = {"source_at_1": bool(names and names[0] in targets),
                                  "source_at_5": bool(targets.intersection(names)),
                                  "all_sources_at_5": targets.issubset(names),
                                  "seconds": round(time.perf_counter() - pool_started, 3)}
        return {"id": case["id"], "group": case["group"], "target_sources_available": targets.issubset(available),
                "methods": sorted(methods), "scores": scores, "seconds": round(time.perf_counter() - started, 3)}

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(evaluate, case) for case in cases if case["id"] not in records]
            for future in as_completed(futures):
                record = future.result()
                records[record["id"]] = record
                with args.output.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(record["id"], record["methods"], flush=True)
        verified = [record for record in records.values() if record["methods"] == [expected_method]
                    and record["target_sources_available"]]
        summary = {"total": len(records), "covered_cloud_cases": len(verified),
                   "method": "independent retrieval and cloud reranking for each pool size",
                   "methods": dict(Counter(method for record in records.values() for method in record["methods"])),
                   "pool_scores": {}}
        for count in args.pools:
            summary["pool_scores"][str(count)] = {
                metric: sum(record["scores"][str(count)][metric] for record in verified)
                for metric in ("source_at_1", "source_at_5", "all_sources_at_5")}
            summary["pool_scores"][str(count)]["mean_seconds"] = round(
                sum(record["scores"][str(count)]["seconds"] for record in verified) / max(1, len(verified)), 3)
        args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
