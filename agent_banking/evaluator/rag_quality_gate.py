# -*- coding: utf-8 -*-
"""Automated Quantitative RAG Quality Gate CLI.

Measures cold initialization, BM25 cache footprint, 785 Master Benchmark
(the / compliance / app / specialized) Hit@1/Hit@5/Hit@45, and Excel-350 Hit@45.

COMPLIANCE WITH GEMINI.md:
- 0% REGEX LAW: Pure string methods only, no import re.
- ZERO TERNARY LAW: Explicit if/else statements only.
- NO __INIT__.PY: Direct module imports.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import sys
import time

import openpyxl

from agent_banking.core.search_text import search_key
from agent_banking.core.settings import Config
from agent_banking.evaluator.config.settings import EvaluatorConfig, resolve_repo_path
from agent_banking.tools.rag_tool import BankingRAGTool, _parse_source_code, _source_stem

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_DS_CFG = EvaluatorConfig.DATASETS


def _source_code_key(text: str) -> str:
    stem_key = search_key(_source_stem(str(text or "")))
    parsed = _parse_source_code(stem_key)
    if parsed is None:
        return ""
    return f"{parsed[0]} {parsed[1]}"


def _extract_expected_targets(case: dict) -> tuple:
    names = set()
    codes = set()
    doc_title = str(case.get("document_title") or "").strip()
    if doc_title:
        names.add(_source_stem(doc_title))
        c_key = _source_code_key(doc_title)
        if c_key:
            codes.add(c_key)
    topic_id = str(case.get("topic_id") or "").strip()
    if topic_id:
        c_topic = _source_code_key(topic_id)
        if c_topic:
            codes.add(c_topic)
    for ev in case.get("evidence", ()) or ():
        if isinstance(ev, dict) and ev.get("source_path"):
            sp = str(ev["source_path"]).strip()
            names.add(_source_stem(sp))
            c_ev = _source_code_key(sp)
            if c_ev:
                codes.add(c_ev)
    return names, codes


def _matches_target(source_path: str, exp_names: set, exp_codes: set) -> bool:
    stem = _source_stem(str(source_path or ""))
    if stem in exp_names:
        return True
    code = _source_code_key(source_path)
    if code and code in exp_codes:
        return True
    return False


def _select_balanced_785(root: Path, limit: int, seed: int) -> list:
    groups = []
    template = _DS_CFG["group_path_template"]
    for group in EvaluatorConfig.DATASET_GROUPS:
        path = resolve_repo_path(template.replace("{group}", group), root=root)
        items = json.loads(path.read_text(encoding="utf-8"))
        random.Random(seed).shuffle(items)
        for item in items:
            item["group"] = group
        groups.append(items)
    selected = []
    max_len = 0
    for grp in groups:
        if len(grp) > max_len:
            max_len = len(grp)
    for idx in range(max_len):
        for grp in groups:
            if idx < len(grp):
                selected.append(grp[idx])
    if limit > 0:
        return selected[:limit]
    return selected


def _load_excel_350(root: Path, limit: int) -> list:
    xlsx_path = root / "dataset" / "A_Bank_golden_excel_350_evaluation_llm.xlsx"
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.active
    cases = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or len(row) < 6:
            continue
        qid = str(row[1] or "").strip()
        doc_code = str(row[2] or "").strip()
        criteria = str(row[3] or "").strip()
        question = str(row[5] or "").strip()
        src_path = ""
        if len(row) > 8 and row[8]:
            src_path = str(row[8]).strip()
        if not qid or not question:
            continue
        cases.append({
            "id": qid,
            "doc_code": doc_code,
            "criteria": criteria,
            "question": question,
            "source_path": src_path,
        })
    wb.close()
    if limit > 0:
        return cases[:limit]
    return cases


def main():
    parser = argparse.ArgumentParser(description="Automated RAG Quality Gate Benchmark CLI")
    parser.add_argument("--limit-785", type=int, default=80, help="Stratified sample size from 785 suite (0 = all)")
    parser.add_argument("--limit-excel", type=int, default=350, help="Case count from Excel-350 suite (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-rerank", action="store_true", help="Skip Cloud Cross-Encoder step (faster candidate pool check)")
    parser.add_argument("--output", type=Path, default=Path("agent_banking/evaluator/output/rag_quality_gate_report.json"))
    args = parser.parse_args()

    from agent_banking.pipeline import BankingRAGPipeline

    root = Path(__file__).resolve().parents[2]
    cfg = Config()
    cfg.KNOWLEDGE_PUBLICATION_SCOPE = ""

    t_init0 = time.perf_counter()
    pipeline = BankingRAGPipeline(config=cfg)
    init_ms = round((time.perf_counter() - t_init0) * 1000.0, 1)
    rag_tool = BankingRAGTool(pipeline_instance=pipeline)

    cache_file = Path(pipeline.vector_store.storage_path) / "bm25_runtime_cache.pkl"
    cache_mb = 0.0
    if cache_file.is_file():
        cache_mb = round(cache_file.stat().st_size / (1024.0 * 1024.0), 2)

    try:
        cases_785 = _select_balanced_785(root, args.limit_785, args.seed)
        group_stats = {}
        misses_785 = []
        latencies_785 = []

        def _eval_785(case):
            exp_names, exp_codes = _extract_expected_targets(case)
            t0 = time.perf_counter()
            q = case["question"]
            if args.skip_rerank:
                parsed = pipeline.parse_query(q)
                cands = pipeline.retriever.retrieve(parsed, top_k=45, candidate_k=45, filter_dict={})
                top_sources = [c.get("chunk", c).get("source", "") for c in cands[:5]]
                pool_sources = [c.get("chunk", c).get("source", "") for c in cands]
            else:
                res = rag_tool.execute(q, top_k=5, candidate_k=45)
                top_sources = [d.get("source", "") for d in res.get("documents", [])]
                pool_sources = list(top_sources)
                for rd in res.get("related_documents", []):
                    pool_sources.append(rd.get("source", ""))
                parsed = pipeline.parse_query(q)
                cands = pipeline.retriever.retrieve(parsed, top_k=45, candidate_k=45, filter_dict={})
                for c in cands:
                    pool_sources.append(c.get("chunk", c).get("source", ""))
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            hit1 = bool(top_sources and _matches_target(top_sources[0], exp_names, exp_codes))
            hit5 = any(_matches_target(s, exp_names, exp_codes) for s in top_sources[:5])
            hit45 = any(_matches_target(s, exp_names, exp_codes) for s in pool_sources)
            return {
                "id": case["id"],
                "group": case["group"],
                "hit1": hit1,
                "hit5": hit5,
                "hit45": hit45,
                "ms": elapsed_ms,
                "question": q[:100],
                "expected": sorted(exp_codes or exp_names)[:2],
                "top3": [_source_stem(s)[:50] for s in top_sources[:3]],
            }

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for rec in [f.result() for f in as_completed([pool.submit(_eval_785, c) for c in cases_785])]:
                grp = rec["group"]
                if grp not in group_stats:
                    group_stats[grp] = {"n": 0, "hit1": 0, "hit5": 0, "hit45": 0}
                gs = group_stats[grp]
                gs["n"] += 1
                if rec["hit1"]:
                    gs["hit1"] += 1
                if rec["hit5"]:
                    gs["hit5"] += 1
                if rec["hit45"]:
                    gs["hit45"] += 1
                else:
                    misses_785.append(rec)
                if not rec["hit5"] and rec["hit45"]:
                    misses_785.append(rec)
                latencies_785.append(rec["ms"])

        excel_cases = _load_excel_350(root, args.limit_excel)
        excel_hit45 = 0
        excel_misses = []

        def _eval_excel(case):
            exp_codes = set()
            c_doc = _source_code_key(case["doc_code"])
            if c_doc:
                exp_codes.add(c_doc)
            exp_names = set()
            if case["source_path"]:
                exp_names.add(_source_stem(case["source_path"]))
                c_sp = _source_code_key(case["source_path"])
                if c_sp:
                    exp_codes.add(c_sp)
            from agent_banking.tools.extensions.banking_query import decompose_subqueries
            q = case["question"]
            parsed = pipeline.parse_query(q)
            cands = pipeline.retriever.retrieve(parsed, top_k=45, candidate_k=45, filter_dict={})
            sub_qs = decompose_subqueries(q)
            if sub_qs:
                from agent_banking.tools.rag_tool import _merge_subquery_candidates
                batches = []
                for sq in sub_qs:
                    sp = pipeline.parse_query(sq)
                    sb = pipeline.retriever.retrieve(sp, top_k=24, candidate_k=24, filter_dict={})
                    if sb:
                        batches.append(sb)
                cands = _merge_subquery_candidates(cands, batches, 45)
            pool_sources = [c.get("chunk", c).get("source", "") for c in cands]
            hit45 = any(_matches_target(s, exp_names, exp_codes) for s in pool_sources)
            return {
                "id": case["id"],
                "doc_code": case["doc_code"],
                "criteria": case["criteria"],
                "hit45": hit45,
                "question": q[:100],
                "top3": [_source_stem(s)[:35] for s in pool_sources[:3]],
            }

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for rec in [f.result() for f in as_completed([pool.submit(_eval_excel, c) for c in excel_cases])]:
                if rec["hit45"]:
                    excel_hit45 += 1
                else:
                    excel_misses.append(rec)

        report = {
            "cold_init_ms": init_ms,
            "bm25_cache_mb": cache_mb,
            "corpus_chunks": pipeline.vector_store.count(),
            "master_785_sample": {
                "count": len(cases_785),
                "groups": group_stats,
                "misses": sorted(misses_785, key=lambda x: x["id"])[:25],
            },
            "excel_350": {
                "count": len(excel_cases),
                "hit45": excel_hit45,
                "hit45_pct": round((excel_hit45 / max(1, len(excel_cases))) * 100.0, 2),
                "misses": sorted(excel_misses, key=lambda x: x["id"]),
            },
        }
        out_path = args.output
        if not out_path.is_absolute():
            out_path = root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"COLD INIT: {init_ms} ms | BM25 CACHE: {cache_mb} MB | CHUNKS: {report['corpus_chunks']}")
        for grp, gs in sorted(group_stats.items()):
            n = max(1, gs["n"])
            print(
                f"  785[{grp:11s}] n={gs['n']} | Hit@1={gs['hit1']}/{gs['n']} ({100*gs['hit1']/n:.1f}%)"
                f" | Hit@5={gs['hit5']}/{gs['n']} ({100*gs['hit5']/n:.1f}%)"
                f" | Hit@45={gs['hit45']}/{gs['n']} ({100*gs['hit45']/n:.1f}%)"
            )
        print(
            f"  EXCEL-350: Hit@45={excel_hit45}/{len(excel_cases)} "
            f"({report['excel_350']['hit45_pct']}%)"
        )
        for m in sorted(misses_785, key=lambda x: x["id"])[:10]:
            print("  785 MISS:", m["id"], m["group"], m["expected"], "->", m["top3"])
        for em in sorted(excel_misses, key=lambda x: x["id"])[:12]:
            print("  EX MISS@45:", em["id"], em["doc_code"], em["top3"])
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
