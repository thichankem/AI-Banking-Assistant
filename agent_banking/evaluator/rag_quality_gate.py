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

from agent_banking.chunking.universal.enrichment.contextual_enricher import strip_ui_noise_for_context
from agent_banking.core.search_text import search_key
from agent_banking.core.numeric_tokens import extract_canonical_numeric_tokens
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


def _reference_fee_codes(text: str) -> set[str]:
    codes = set()
    words = search_key(text).split()
    for index, word in enumerate(words):
        if word.startswith("bp"):
            parsed = _parse_source_code(" ".join(words[index:index + 2]))
            if parsed and parsed[0] == "bp":
                codes.add("bp " + parsed[1])
    return codes


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


def _expects_absent_evidence(anchor):
    quote = search_key(str(anchor.get("verbatim_quote") or anchor.get("quote") or ""))
    section = search_key(str(anchor.get("section") or ""))
    marker = "khong co trong tai lieu hien hanh"
    return quote == marker and (section == marker or section.startswith(marker + " "))


def _anchor_hits(case, documents):
    hits = []
    for anchor in case.get("evidence", ()):
        source = anchor.get("source_path", "")
        quote = search_key(str(anchor.get("verbatim_quote") or anchor.get("quote") or ""))
        section = search_key(str(anchor.get("section") or ""))
        if not source or not (quote or section) or _expects_absent_evidence(anchor):
            continue
        code = _source_code_key(source)
        located = False
        for document in documents:
            if not _matches_target(document.get("source", ""), {_source_stem(source)}, {code}):
                continue
            text = search_key(document.get("content", "") + " " + document.get("parent_content", ""))
            heading = search_key(document.get("section_title", "") + " " + document.get("breadcrumb", ""))
            if quote and f" {quote} " in f" {text} " or not quote and section and section in heading:
                located = True
                break
        hits.append(located)
    return hits


_GROUND_STOPWORDS = frozenset({
    "cua", "cho", "trong", "theo", "va", "hay", "hoac", "nhu", "the", "nao",
    "cac", "nhung", "mot", "duoc", "khong", "khi", "sau", "truoc", "tai",
    "ve", "tu", "den", "la", "gi", "co", "voi", "tren", "duoi", "nay", "do",
    "ngan", "hang", "abank", "a", "bank", "lpbank",
})


def _split_ellipsis_segments(raw_quote: str) -> list:
    normalized = str(raw_quote or "").replace("…", "...")
    parts = normalized.split("...")
    segments = []
    for part in parts:
        seg_key = search_key(part)
        if seg_key:
            segments.append(seg_key)
    return segments


def _grounded_anchor_hits(case, documents):
    hits = []
    for anchor in case.get("evidence", ()):
        source = anchor.get("source_path", "")
        raw_quote = str(anchor.get("verbatim_quote") or anchor.get("quote") or "")
        quote = search_key(raw_quote)
        section = search_key(str(anchor.get("section") or ""))
        if not source or not (quote or section) or _expects_absent_evidence(anchor):
            continue
        code = _source_code_key(source)
        target_docs = []
        for document in documents:
            if _matches_target(document.get("source", ""), {_source_stem(source)}, {code}):
                target_docs.append(document)
        if not target_docs:
            hits.append(False)
            continue

        located = False
        per_doc_texts = []
        all_tokens = set()
        for document in target_docs:
            raw_content = str(document.get("content", "") or "")
            raw_parent = str(document.get("parent_content", "") or "")
            clean_content = strip_ui_noise_for_context(raw_content)
            text_raw = search_key(raw_content + " " + raw_parent)
            text_clean = search_key(clean_content + " " + raw_parent)
            heading = search_key(str(document.get("section_title", "") or "") + " " + str(document.get("breadcrumb", "") or ""))
            per_doc_texts.append((text_raw, text_clean, heading))
            all_tokens.update(text_raw.split())
            all_tokens.update(text_clean.split())
            all_tokens.update(heading.split())
            if quote and (f" {quote} " in f" {text_raw} " or f" {quote} " in f" {text_clean} "):
                located = True
                break
            if not quote and section and section in heading:
                located = True
                break

        if not located and quote:
            segments = _split_ellipsis_segments(raw_quote)
            substantive = [seg for seg in segments if len(seg) >= 4]
            if substantive:
                combined_raw = " " + " ".join(t[0] for t in per_doc_texts) + " "
                combined_clean = " " + " ".join(t[1] for t in per_doc_texts) + " "
                if all((f" {seg} " in combined_raw or f" {seg} " in combined_clean) for seg in substantive):
                    located = True
            if not located:
                q_tokens = []
                for tok in quote.split():
                    if (len(tok) >= 2 or tok.isdigit()) and tok not in _GROUND_STOPWORDS:
                        if tok not in q_tokens:
                            q_tokens.append(tok)
                if q_tokens:
                    matched_toks = 0
                    for tok in q_tokens:
                        if tok in all_tokens:
                            matched_toks += 1
                    if (matched_toks / len(q_tokens)) >= 0.80:
                        located = True

        hits.append(located)
    return hits


class _HybridOrder:
    @staticmethod
    def rerank(query, candidates, top_k):
        return [{**item, "rerank_method": "hybrid_order"} for item in candidates[:top_k]]


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
            "reference_answer": str(row[6] or "").strip(),
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
    parser.add_argument("--min-hit45-pct", type=float, default=0.0, help="Exit non-zero if 785 Hit@45%% drops below this threshold")
    parser.add_argument("--min-excel-hit45-pct", type=float, default=0.0, help="Exit non-zero if Excel-350 Hit@45%% drops below this threshold")
    parser.add_argument("--output", type=Path, default=Path("agent_banking/evaluator/output/rag_quality_gate_report.json"))
    args = parser.parse_args()

    from agent_banking.pipeline import BankingRAGPipeline

    root = Path(__file__).resolve().parents[2]
    cfg = Config()
    t_init0 = time.perf_counter()
    pipeline = BankingRAGPipeline(config=cfg)
    if args.skip_rerank:
        pipeline.reranker = _HybridOrder()
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
        evidence_misses = []
        absent_evidence_cases = []
        latencies_785 = []

        corpus_docs = getattr(pipeline.vector_store, "_bm25_corpus", ())
        corpus_numbers_by_target = {}
        for c_doc in corpus_docs:
            if not isinstance(c_doc, dict):
                continue
            c_meta = c_doc.get("business_metadata") or {}
            if not c_meta.get("excel_table_id"):
                continue
            c_src = str(c_doc.get("source") or "")
            c_code = _source_code_key(c_src)
            c_stem = _source_stem(c_src)
            c_nums = set(extract_canonical_numeric_tokens(str(c_doc.get("content") or "")))
            if c_code:
                corpus_numbers_by_target.setdefault(c_code, set()).update(c_nums)
            if c_stem:
                corpus_numbers_by_target.setdefault(c_stem, set()).update(c_nums)

        def _eval_785(case):
            exp_names, exp_codes = _extract_expected_targets(case)
            t0 = time.perf_counter()
            q = case["question"]
            res = rag_tool.execute(q, top_k=5, candidate_k=45, include_candidates=True)
            top = res.get("documents", [])
            top_sources = [d.get("source", "") for d in top]
            pool_sources = [d.get("source", "") for d in res.get("candidate_documents", [])[:45]]
            combined_docs = [*top, *res.get("related_documents", [])]
            anchors = _anchor_hits(case, combined_docs)
            grounded_anchors = _grounded_anchor_hits(case, combined_docs)
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
                "anchor_count": len(anchors),
                "anchor_hits": sum(anchors),
                "grounded_anchor_hits": sum(grounded_anchors),
                "absent_evidence_count": sum(_expects_absent_evidence(anchor) for anchor in case.get("evidence", [])),
                "ms": elapsed_ms,
                "question": q[:100],
                "expected": sorted(exp_codes or exp_names)[:2],
                "top3": [_source_stem(s)[:50] for s in top_sources[:3]],
            }

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for rec in [f.result() for f in as_completed([pool.submit(_eval_785, c) for c in cases_785])]:
                grp = rec["group"]
                if grp not in group_stats:
                    group_stats[grp] = {
                        "n": 0, "hit1": 0, "hit5": 0, "hit45": 0,
                        "anchor_count": 0, "anchor_hits": 0, "grounded_anchor_hits": 0,
                    }
                gs = group_stats[grp]
                gs["n"] += 1
                gs["anchor_count"] += rec["anchor_count"]
                gs["anchor_hits"] += rec["anchor_hits"]
                gs["grounded_anchor_hits"] += rec["grounded_anchor_hits"]
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
                if rec["anchor_hits"] < rec["anchor_count"]:
                    evidence_misses.append({
                        key: rec[key]
                        for key in ("id", "group", "anchor_hits", "grounded_anchor_hits", "anchor_count", "expected")
                    })
                if rec["absent_evidence_count"]:
                    absent_evidence_cases.append({key: rec[key] for key in ("id", "group", "absent_evidence_count", "expected")})
                latencies_785.append(rec["ms"])

        excel_cases = _load_excel_350(root, args.limit_excel)
        excel_hit45 = 0
        excel_table_hits = 0
        excel_numeric_cases = 0
        excel_numeric_hits = 0
        excel_source_numeric_cases = 0
        excel_source_numeric_hits = 0
        excel_misses = []
        excel_evidence_misses = []

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
            q = case["question"]
            result = rag_tool.execute(q, top_k=5, candidate_k=45, include_candidates=True)
            pool_sources = [d.get("source", "") for d in result.get("candidate_documents", [])[:45]]
            hit45 = any(_matches_target(s, exp_names, exp_codes) for s in pool_sources)
            tables = []
            for document in [*result.get("documents", []), *result.get("related_documents", [])]:
                meta = document["business_metadata"]
                citation = meta.get("citation", {})
                if (meta.get("excel_table_id") and citation.get("sheet_name") and citation.get("cell_ranges")
                        and meta.get("temporal", {}).get("status") not in {"reference_only", "conflict"}
                        and _matches_target(document["source"], exp_names, exp_codes)):
                    tables.append(document)
            expected_numbers = set(extract_canonical_numeric_tokens(case["reference_answer"]))
            returned_numbers = set()
            for document in tables:
                returned_numbers.update(extract_canonical_numeric_tokens(document["content"]))
            missing_numbers = expected_numbers - returned_numbers
            corpus_target_nums = set()
            for key_t in (*exp_codes, *exp_names):
                corpus_target_nums.update(corpus_numbers_by_target.get(key_t, ()))
            source_expected_numbers = expected_numbers & corpus_target_nums
            question_numbers = set(extract_canonical_numeric_tokens(q))
            reference = case["reference_answer"]
            alternate_codes = _reference_fee_codes(reference) - exp_codes
            returned_top3 = [_source_stem(d.get("source", ""))[:35] for d in result.get("documents", [])[:3]]
            return {
                "id": case["id"],
                "doc_code": case["doc_code"],
                "criteria": case["criteria"],
                "hit45": hit45,
                "table_hit": bool(tables),
                "numeric_checked": bool(expected_numbers),
                "numeric_hit": bool(expected_numbers) and expected_numbers <= returned_numbers,
                "source_numeric_checked": bool(source_expected_numbers),
                "source_numeric_hit": bool(source_expected_numbers) and source_expected_numbers <= returned_numbers,
                "missing_reference_numbers": sorted(missing_numbers),
                "missing_numbers_given_in_question": sorted(missing_numbers & question_numbers),
                "alternate_fee_codes_in_reference": sorted(alternate_codes),
                "reference_contains_formula": "=" in reference and any(op in reference for op in ("*", "×")),
                "question": q[:100],
                "top3": [_source_stem(s)[:35] for s in pool_sources[:3]],
                "returned_top3": returned_top3,
            }

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for rec in [f.result() for f in as_completed([pool.submit(_eval_excel, c) for c in excel_cases])]:
                excel_table_hits += rec["table_hit"]
                excel_numeric_cases += rec["numeric_checked"]
                excel_numeric_hits += rec["numeric_hit"]
                excel_source_numeric_cases += rec["source_numeric_checked"]
                excel_source_numeric_hits += rec["source_numeric_hit"]
                if rec["hit45"]:
                    excel_hit45 += 1
                else:
                    excel_misses.append(rec)
                if not rec["table_hit"] or (rec["numeric_checked"] and not rec["numeric_hit"]):
                    excel_evidence_misses.append(rec)

        report = {
            "cold_init_ms": init_ms,
            "bm25_cache_mb": cache_mb,
            "corpus_chunks": pipeline.vector_store.count(),
            "master_785_sample": {
                "count": len(cases_785),
                "groups": group_stats,
                "misses": sorted(misses_785, key=lambda x: x["id"])[:25],
                "evidence_misses": sorted(evidence_misses, key=lambda x: x["id"]),
                "absent_evidence_cases": sorted(absent_evidence_cases, key=lambda x: x["id"]),
            },
            "excel_350": {
                "count": len(excel_cases),
                "hit45": excel_hit45,
                "hit45_pct": round((excel_hit45 / max(1, len(excel_cases))) * 100.0, 2),
                "citable_table_hits": excel_table_hits,
                "numeric_reference_cases": excel_numeric_cases,
                "numeric_reference_hits": excel_numeric_hits,
                "source_numeric_cases": excel_source_numeric_cases,
                "source_numeric_hits": excel_source_numeric_hits,
                "misses": sorted(excel_misses, key=lambda x: x["id"]),
                "evidence_misses": sorted(excel_evidence_misses, key=lambda x: x["id"]),
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
                f" | Evidence anchors={gs['anchor_hits']}/{gs['anchor_count']}"
                f" (grounded={gs['grounded_anchor_hits']}/{gs['anchor_count']})"
            )
        print(
            f"  EXCEL-350: Hit@45={excel_hit45}/{len(excel_cases)} "
            f"({report['excel_350']['hit45_pct']}%) | tables={excel_table_hits}/{len(excel_cases)} "
            f"| reference figures={excel_numeric_hits}/{excel_numeric_cases} "
            f"(source_figures={excel_source_numeric_hits}/{excel_source_numeric_cases})"
        )
        for m in sorted(misses_785, key=lambda x: x["id"])[:10]:
            print("  785 MISS:", m["id"], m["group"], m["expected"], "->", m["top3"])
        for em in sorted(excel_misses, key=lambda x: x["id"])[:12]:
            print("  EX MISS@45:", em["id"], em["doc_code"], em["top3"])
    finally:
        pipeline.close()

    total_785 = sum(gs["n"] for gs in group_stats.values())
    hit45_785_pct = (sum(gs["hit45"] for gs in group_stats.values()) * 100.0) / max(1, total_785)
    gate_failures = []
    if args.min_hit45_pct > 0.0 and total_785 > 0 and hit45_785_pct < args.min_hit45_pct:
        gate_failures.append(f"785 Hit@45 {hit45_785_pct:.2f}% < min {args.min_hit45_pct:.2f}%")
    if args.min_excel_hit45_pct > 0.0 and len(excel_cases) > 0 and report["excel_350"]["hit45_pct"] < args.min_excel_hit45_pct:
        gate_failures.append(f"Excel Hit@45 {report['excel_350']['hit45_pct']:.2f}% < min {args.min_excel_hit45_pct:.2f}%")
    if gate_failures:
        raise SystemExit("Quality gate failed: " + "; ".join(gate_failures))


if __name__ == "__main__":
    main()
