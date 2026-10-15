"""
Module Task 0 Reproduce (Tái Lập và Đối Soát Kết Quả Master Ablation)
=====================================================================
Mục tiêu Task 0:
1. Chạy lại toàn bộ 8 cấu hình (Config 0 -> Config 5C) trên 50 câu Golden QA.
2. Kiểm tra chặt chẽ: Mỗi chỉ số (Recall@5, Recall@10, MRR@10, NDCG@10) phải khớp báo cáo cũ trong khoảng +-0.005.
3. Xuất kết quả chi tiết từng câu hỏi (per-question) ra `benchmarks/results/{config_slug}.jsonl`.
4. Ghi nhận seed cố định, git commit hash và metadata đầy đủ.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.ablation_runners.run_ablation import (
    BaselineNaiveChunker,
    PipelineV1Chunker,
    PipelineV2Chunker,
    PipelineUpgradeOCRRunner,
    PipelineFastTableRunner,
    PipelineV3UltimateRunner,
    PROFILE_A,
    PROFILE_B,
    PROFILE_C,
    TableFormatType
)
from benchmarks.evaluators.retrieval_evaluator import (
    RetrievalEvaluator,
    compute_token_containment
)


def get_git_commit_hash() -> str:
    try:
        res = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT))
        return res.decode("utf-8").strip()
    except Exception:
        return "unknown_commit"


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "_", cleaned)


def run_task0_reproduction(config_yaml_path: str | Path) -> bool:
    yaml_file = Path(config_yaml_path)
    if not yaml_file.exists():
        yaml_file = PROJECT_ROOT / config_yaml_path

    with open(yaml_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["meta"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    git_hash = get_git_commit_hash()
    print(f"\n{'='*78}")
    print(f"[TASK 0] TÁI LẬP KẾT QUẢ PHASE 1 BENCHMARK (REPRODUCIBILITY CHECK)")
    print(f"Git Commit Hash: {git_hash} | Seed: {seed}")
    print(f"{'='*78}\n")

    qa_path = PROJECT_ROOT / cfg["dataset"]["golden_qa_path"]
    with open(qa_path, "r", encoding="utf-8") as f:
        golden_qa = json.load(f)

    corpus_paths = [PROJECT_ROOT / p for p in cfg["dataset"]["corpus_files"]]
    tolerance = float(cfg.get("tolerance", 0.005))
    target_baselines = cfg.get("target_baselines", {})

    results_dir = PROJECT_ROOT / "experiments" / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    configs_to_run = {
        "Config 0 - Baseline (Naive Chunking)": BaselineNaiveChunker(),
        "Config 1 - Pipeline v1 (Structure + Breadcrumb)": PipelineV1Chunker(),
        "Config 2 - Pipeline v2 (Current Production)": PipelineV2Chunker(),
        "Config 3 - Upgrade OCR & Scan Table (v2 + exp1)": PipelineUpgradeOCRRunner(),
        "Config 4 - Upgrade Fast Vision & Table Formats (Config 3 + exp2)": PipelineFastTableRunner(format_type=TableFormatType.MARKDOWN_TABLE),
        "Config 5A - Pipeline v3 (Profile A: 100-200)": PipelineV3UltimateRunner(profile=PROFILE_A),
        "Config 5B - Pipeline v3 Ultimate (Profile B: 150-400)": PipelineV3UltimateRunner(profile=PROFILE_B),
        "Config 5C - Pipeline v3 (Profile C: Parent-Child 200/700)": PipelineV3UltimateRunner(profile=PROFILE_C)
    }

    evaluator = RetrievalEvaluator(containment_threshold=0.65)
    reproduced_metrics: dict[str, dict[str, float]] = {}
    mismatch_detected = False
    mismatch_report = []

    summary_records = []

    for cfg_name, runner in configs_to_run.items():
        print(f">>> Đang chạy tái lập: [{cfg_name}]...")
        t0 = time.time()
        chunks = []
        for p in corpus_paths:
            chunks.extend(runner.chunk_file(p))

        parent_map = getattr(runner, "parent_map", None)
        evaluator.retriever.index(chunks)

        slug = slugify(cfg_name)
        per_q_file = results_dir / f"{slug}_per_question.jsonl"

        recalls_5, recalls_10 = [], []
        mrrs, ndcgs = [], []
        per_q_records = []

        for q_idx, sample in enumerate(golden_qa):
            query = sample.get("question", "")
            target_ids = set(sample.get("expected_chunk_ids", []))
            target_span = sample.get("ground_truth_context_span") or sample.get("ground_truth_answer", "")
            target_bc = sample.get("expected_breadcrumb_prefix", "")
            cat = sample.get("category", "general")

            top10_docs = evaluator.retriever.search(query, top_k=10)

            ranked_hits = []
            top10_info = []
            for rank_pos, (doc, score) in enumerate(top10_docs, start=1):
                is_rel = evaluator.is_chunk_relevant(
                    retrieved_chunk=doc,
                    expected_chunk_ids=target_ids,
                    expected_context_span=target_span,
                    expected_bc_prefix=target_bc,
                    parent_map=parent_map
                )
                ranked_hits.append(1 if is_rel else 0)
                top10_info.append({
                    "rank": rank_pos,
                    "chunk_id": doc.get("chunk_id", ""),
                    "score": round(float(score), 4),
                    "is_relevant": is_rel,
                    "breadcrumb": doc.get("breadcrumb", ""),
                    "snippet": doc.get("content", "")[:120].replace("\n", " ")
                })

            r5 = 1.0 if any(ranked_hits[:5]) else 0.0
            r10 = 1.0 if any(ranked_hits[:10]) else 0.0
            recalls_5.append(r5)
            recalls_10.append(r10)

            # MRR
            first_hit_rank = None
            mrr_val = 0.0
            for rank_pos, rel in enumerate(ranked_hits[:10], start=1):
                if rel == 1:
                    first_hit_rank = rank_pos
                    mrr_val = 1.0 / rank_pos
                    break
            mrrs.append(mrr_val)

            # NDCG
            dcg = 0.0
            for rank_pos, rel in enumerate(ranked_hits[:10], start=1):
                if rel == 1:
                    dcg += 1.0 / math.log2(rank_pos + 1)
            num_rel = sum(ranked_hits[:10])
            if num_rel > 0:
                idcg = sum(1.0 / math.log2(i + 1) for i in range(1, num_rel + 1))
                ndcg_val = round(min(dcg / idcg, 1.0), 4)
            else:
                ndcg_val = 0.0
            ndcgs.append(ndcg_val)

            per_q_record = {
                "question_idx": q_idx,
                "question": query,
                "category": cat,
                "first_hit_rank": first_hit_rank,
                "hit_at_5": bool(r5 > 0),
                "hit_at_10": bool(r10 > 0),
                "mrr_at_10": round(mrr_val, 4),
                "ndcg_at_10": round(ndcg_val, 4),
                "top10_chunks": top10_info
            }
            per_q_records.append(per_q_record)

        # Lưu per-question jsonl
        with open(per_q_file, "w", encoding="utf-8") as pf:
            for item in per_q_records:
                pf.write(json.dumps(item, ensure_ascii=False) + "\n")

        measured = {
            "recall_at_5": round(sum(recalls_5) / len(recalls_5), 4),
            "recall_at_10": round(sum(recalls_10) / len(recalls_10), 4),
            "mrr_at_10": round(sum(mrrs) / len(mrrs), 4),
            "ndcg_at_10": round(sum(ndcgs) / len(ndcgs), 4),
            "total_chunks": len(chunks),
            "elapsed_sec": round(time.time() - t0, 2)
        }
        reproduced_metrics[cfg_name] = measured

        # Kiểm tra tolerance so với target baseline
        target = target_baselines.get(cfg_name, {})
        diffs = {}
        for m_key in ["recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10"]:
            if m_key in target:
                t_val = target[m_key]
                m_val = measured[m_key]
                diff = abs(m_val - t_val)
                diffs[m_key] = {"measured": m_val, "target": t_val, "diff": round(diff, 4)}
                if diff > tolerance:
                    mismatch_detected = True
                    mismatch_report.append({
                        "config": cfg_name,
                        "metric": m_key,
                        "measured": m_val,
                        "target": t_val,
                        "diff": round(diff, 4),
                        "tolerance": tolerance
                    })

        summary_records.append({
            "config": cfg_name,
            "chunks": len(chunks),
            "measured": measured,
            "target": target,
            "status": "PASS" if all(d["diff"] <= tolerance for d in diffs.values()) else "FAIL"
        })

        status_str = "PASS" if all(d["diff"] <= tolerance for d in diffs.values()) else "FAIL"
        print(f"   [{status_str}] R@5: {measured['recall_at_5']} (target: {target.get('recall_at_5')}) | "
              f"R@10: {measured['recall_at_10']} (target: {target.get('recall_at_10')}) | "
              f"MRR@10: {measured['mrr_at_10']} (target: {target.get('mrr_at_10')}) | "
              f"NDCG@10: {measured['ndcg_at_10']} (target: {target.get('ndcg_at_10')})")

    # Xuất báo cáo tổng kết Task 0
    summary_path = results_dir / "task0_reproduction_summary.json"
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump({
            "meta": {
                "git_commit_hash": git_hash,
                "seed": seed,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "tolerance": tolerance
            },
            "overall_status": "ALL_PASS" if not mismatch_detected else "MISMATCH_DETECTED",
            "configs": summary_records,
            "mismatches": mismatch_report
        }, sf, ensure_ascii=False, indent=2)

    print(f"\n{'='*78}")
    if mismatch_detected:
        print("[TASK 0 THẤT BẠI] Phát hiện chênh lệch vượt ngưỡng +-0.005:")
        for m in mismatch_report:
            print(f"  - {m['config']} -> {m['metric']}: Measured={m['measured']} vs Target={m['target']} (Diff={m['diff']})")
        print(f"{'='*78}\n")
        return False
    else:
        print("[TASK 0 THÀNH CÔNG] Toàn bộ 8 cấu hình khớp 100% với báo cáo Phase 1 trong dung sai +-0.005!")
        print(f"Dữ liệu per-question đã lưu tại: {results_dir}/*.jsonl")
        print(f"Bản tổng hợp kết quả tại:        {summary_path}")
        print(f"{'='*78}\n")
        return True


if __name__ == "__main__":
    cfg_arg = "benchmarks/configs/task0_reproduction.yaml"
    if len(sys.argv) > 1:
        cfg_arg = sys.argv[1]
    success = run_task0_reproduction(cfg_arg)
    sys.exit(0 if success else 1)
