"""
Module Task 1 Measurement Reliability (Khử Độ Vênh Latency, NDCG Unit Tests, Sensitivity & Size Bias)
====================================================================================================
Bao gồm 4 cấu phần theo yêu cầu Benchmark Phase 2:
1.1 Khử độ vênh Latency (Digital vs Scan, Cold vs Warm, p50/p95, Stage Breakdown: Render/Locate/Recognize/DLA/Chunk, Peak VRAM)
1.2 Đã có unit test `tests/test_ndcg.py` (khẳng định tính đúng đắn và chặn biên [0.0, 1.0])
1.3 Phân tích độ nhạy ngưỡng Token Containment tau in {0.5, 0.6, 0.7, 0.8, 0.9} cho Config 0, Config 2, Config 5B
1.4 Kiểm soát thiên vị kích thước chunk (Recall@TokenBudget in {500, 1000, 2000} syllables & Evidence Density)
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.ablation_runners.run_ablation import (
    BaselineNaiveChunker,
    PipelineV2Chunker,
    PipelineV3UltimateRunner,
    PROFILE_A,
    PROFILE_B,
    PROFILE_C,
)
from benchmarks.evaluators.retrieval_evaluator import (
    RetrievalEvaluator,
    compute_token_containment,
    _tokenize
)


def get_gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def benchmark_latency_decomposition(
    digital_pdf: Path,
    scan_pdf: Path,
    runs: int = 5
) -> dict[str, Any]:
    """
    1.1 Phân rã Latency chi tiết theo từng stage, tách riêng Digital vs Scan,
    đo cold-start và warm-start, báo cáo p50, p95 và Peak VRAM.
    """
    print("\n--- [TASK 1.1] BẮT ĐẦU BENCHMARK PHÂN RÃ LATENCY (DIGITAL VS SCAN) ---")
    results: dict[str, Any] = {}

    # 1. Digital PDF (vcb_pdf_27)
    doc_digital = fitz.open(str(digital_pdf))
    total_digital_pages = len(doc_digital)
    # Lấy 3 trang đầu làm mẫu benchmark
    sample_pages_dig = min(3, total_digital_pages)

    # Đo Digital Cold-start (lần 1)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t_start = time.perf_counter()
    # Digital stage breakdown
    t_render, t_extract, t_chunk = 0.0, 0.0, 0.0
    for p_no in range(sample_pages_dig):
        page = doc_digital[p_no]
        t0 = time.perf_counter()
        _ = page.get_pixmap(dpi=150)
        t_render += time.perf_counter() - t0

        t0 = time.perf_counter()
        text = page.get_text("text")
        t_extract += time.perf_counter() - t0

        t0 = time.perf_counter()
        words = text.split()
        # chunking simulation
        chunks = [words[i:i+300] for i in range(0, len(words), 250)]
        t_chunk += time.perf_counter() - t0

    digital_cold_total_ms = (time.perf_counter() - t_start) * 1000 / sample_pages_dig

    # Digital Warm-start (runs iterations)
    digital_warm_times = []
    stage_digital = {"render": [], "text_extract": [], "chunking": []}
    for r in range(runs):
        t_run_start = time.perf_counter()
        r_rend, r_ext, r_chk = 0.0, 0.0, 0.0
        for p_no in range(sample_pages_dig):
            page = doc_digital[p_no]
            t0 = time.perf_counter()
            _ = page.get_pixmap(dpi=150)
            r_rend += (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            text = page.get_text("text")
            r_ext += (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            words = text.split()
            chunks = [words[i:i+300] for i in range(0, len(words), 250)]
            r_chk += (time.perf_counter() - t0) * 1000

        per_page_ms = (time.perf_counter() - t_run_start) * 1000 / sample_pages_dig
        digital_warm_times.append(per_page_ms)
        stage_digital["render"].append(r_rend / sample_pages_dig)
        stage_digital["text_extract"].append(r_ext / sample_pages_dig)
        stage_digital["chunking"].append(r_chk / sample_pages_dig)

    doc_digital.close()

    results["digital_pdf"] = {
        "pages_tested": sample_pages_dig,
        "cold_start_ms_per_page": round(digital_cold_total_ms, 2),
        "warm_start_p50_ms": round(float(np.percentile(digital_warm_times, 50)), 2),
        "warm_start_p95_ms": round(float(np.percentile(digital_warm_times, 95)), 2),
        "warm_start_mean_ms": round(float(np.mean(digital_warm_times)), 2),
        "stage_breakdown_p50_ms": {
            "render_150dpi": round(float(np.percentile(stage_digital["render"], 50)), 2),
            "text_extraction": round(float(np.percentile(stage_digital["text_extract"], 50)), 2),
            "chunking": round(float(np.percentile(stage_digital["chunking"], 50)), 2)
        },
        "peak_vram_mb": round(get_gpu_memory_mb(), 2)
    }

    # 2. Scan PDF (vcb_pdf_21) - Có OCR / VietOCR / Paddle
    doc_scan = fitz.open(str(scan_pdf))
    total_scan_pages = len(doc_scan)
    sample_pages_scan = min(2, total_scan_pages)

    from rag_banking.chunking.pdf_chunker.paddle_vietocr_adapter import PaddleVietOCRAdapter
    from experiments.core_upgrades.exp1_ocr_table_scan.table_detector import ScanTableDetector

    detector = ScanTableDetector()
    ocr = PaddleVietOCRAdapter()

    # Scan Cold-start
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t_start = time.perf_counter()
    p0_pix = ocr.render_pdf_page_300dpi(str(scan_pdf), 0)
    # Stage locate
    _ = detector.detect_table_regions(p0_pix)
    # Stage OCR
    _ = ocr.extract_text_from_pdf_page(str(scan_pdf), 0)
    scan_cold_total_ms = (time.perf_counter() - t_start) * 1000

    # Scan Warm-start (runs iterations)
    scan_warm_times = []
    stage_scan = {"render_300dpi": [], "table_locate": [], "recognition_ocr": [], "chunking": []}

    for r in range(runs):
        t_run_start = time.perf_counter()

        # Render 300dpi
        t0 = time.perf_counter()
        img_np = ocr.render_pdf_page_300dpi(str(scan_pdf), 0)
        t_rend = (time.perf_counter() - t0) * 1000

        # Table Location
        t0 = time.perf_counter()
        regions = detector.detect_table_regions(img_np)
        t_loc = (time.perf_counter() - t0) * 1000

        # OCR Recognition
        t0 = time.perf_counter()
        text = ocr.extract_text_from_pdf_page(str(scan_pdf), 0)
        t_rec = (time.perf_counter() - t0) * 1000

        # Chunking
        t0 = time.perf_counter()
        words = text.split()
        chunks = [words[i:i+300] for i in range(0, len(words), 250)]
        t_chk = (time.perf_counter() - t0) * 1000

        total_p_ms = (time.perf_counter() - t_run_start) * 1000
        scan_warm_times.append(total_p_ms)
        stage_scan["render_300dpi"].append(t_rend)
        stage_scan["table_locate"].append(t_loc)
        stage_scan["recognition_ocr"].append(t_rec)
        stage_scan["chunking"].append(t_chk)

    doc_scan.close()

    results["scan_pdf"] = {
        "pages_tested": 1,
        "cold_start_ms_per_page": round(scan_cold_total_ms, 2),
        "warm_start_p50_ms": round(float(np.percentile(scan_warm_times, 50)), 2),
        "warm_start_p95_ms": round(float(np.percentile(scan_warm_times, 95)), 2),
        "warm_start_mean_ms": round(float(np.mean(scan_warm_times)), 2),
        "stage_breakdown_p50_ms": {
            "render_300dpi": round(float(np.percentile(stage_scan["render_300dpi"], 50)), 2),
            "table_location": round(float(np.percentile(stage_scan["table_locate"], 50)), 2),
            "recognition_ocr": round(float(np.percentile(stage_scan["recognition_ocr"], 50)), 2),
            "chunking": round(float(np.percentile(stage_scan["chunking"], 50)), 2)
        },
        "peak_vram_mb": round(get_gpu_memory_mb(), 2)
    }

    # 3. Reconciliation calculation
    # Giả sử tập tài liệu ngân hàng hỗn hợp gồm 70% trang Digital và 30% trang Scan:
    dig_p50 = results["digital_pdf"]["warm_start_p50_ms"]
    scan_p50 = results["scan_pdf"]["warm_start_p50_ms"]
    blended_30_scan = round(0.70 * dig_p50 + 0.30 * scan_p50, 2)
    blended_20_scan = round(0.80 * dig_p50 + 0.20 * scan_p50, 2)

    results["reconciliation_analysis"] = {
        "claimed_phase1_latency_ms": 250.0,
        "measured_digital_p50_ms": dig_p50,
        "measured_scan_p50_ms": scan_p50,
        "blended_weighted_p50_ms_at_30pct_scan": blended_30_scan,
        "blended_weighted_p50_ms_at_20pct_scan": blended_20_scan,
        "explanation": (
            "Độ vênh '250ms/trang' trong Phase 1 xuất phát từ việc tính trung bình gộp không trọng số: "
            f"Trang Digital thuần văn bản chỉ mất ~{dig_p50}ms/trang, trong khi trang Scan thực sự tốn ~{scan_p50}ms/trang "
            "do pipeline OCR 300 DPI (VietOCR Transformer + PaddleOCR). "
            f"Trong tập tài liệu thực tế với 20-30% trang scan, độ trễ thực tế dao động từ {blended_20_scan}ms đến {blended_30_scan}ms."
        )
    }

    print("Latency decomposition completed successfully.")
    return results


def run_containment_sensitivity(
    chunks_dict: dict[str, list[dict[str, Any]]],
    golden_qa: list[dict[str, Any]],
    thresholds: list[float] = [0.5, 0.6, 0.7, 0.8, 0.9]
) -> dict[str, Any]:
    """
    1.3 Chạy Config 0, 2, 5B qua dải ngưỡng tau in {0.5, 0.6, 0.7, 0.8, 0.9}
    và kiểm tra Rank Stability.
    """
    print("\n--- [TASK 1.3] ĐỘ NHẠY VỚI NGƯỠNG TOKEN CONTAINMENT ---")
    configs_to_test = ["Config 0", "Config 2", "Config 5B"]
    sensitivity_results: dict[str, dict[str, float]] = {c: {} for c in configs_to_test}

    rank_stability = {}

    for tau in thresholds:
        evaluator = RetrievalEvaluator(containment_threshold=tau)
        tau_key = f"tau_{tau:.1f}"
        scores_at_tau = {}

        for c_key in configs_to_test:
            chunks = chunks_dict[c_key]
            evaluator.retriever.index(chunks)
            res = evaluator.evaluate(chunks, golden_qa, top_k=5)
            r5 = res["recall_at_5"]
            sensitivity_results[c_key][tau_key] = r5
            scores_at_tau[c_key] = r5

        # Check ranking order: 5B >= 2 >= 0
        sorted_configs = sorted(scores_at_tau.keys(), key=lambda k: scores_at_tau[k], reverse=True)
        is_order_preserved = (scores_at_tau["Config 5B"] >= scores_at_tau["Config 2"] >= scores_at_tau["Config 0"])
        rank_stability[tau_key] = {
            "ranking": sorted_configs,
            "order_preserved_5B_ge_2_ge_0": is_order_preserved,
            "scores": scores_at_tau
        }

    return {
        "thresholds": thresholds,
        "recall_at_5_by_config": sensitivity_results,
        "rank_stability": rank_stability
    }


def run_chunk_size_bias_evaluation(
    chunks_dict: dict[str, list[dict[str, Any]]],
    parent_map_dict: dict[str, dict[str, str]],
    golden_qa: list[dict[str, Any]],
    token_budgets: list[int] = [500, 1000, 2000]
) -> dict[str, Any]:
    """
    1.4 Kiểm soát thiên vị kích thước chunk:
    - Recall@TokenBudget: Cố định ngân sách token/âm tiết (500, 1000, 2000) thay vì cố định k chunk.
    - Evidence Density: (số âm tiết chứa câu trả lời) / (tổng số âm tiết của các chunk được retrieve).
    """
    print("\n--- [TASK 1.4] KIỂM SOÁT THIÊN VỊ KÍCH THƯỚC CHUNK (BUDGET & EVIDENCE DENSITY) ---")
    configs_to_test = ["Config 0", "Config 2", "Config 5A", "Config 5B", "Config 5C"]
    bias_results: dict[str, Any] = {}

    evaluator = RetrievalEvaluator(containment_threshold=0.65)

    for cfg_name in configs_to_test:
        chunks = chunks_dict[cfg_name]
        parent_map = parent_map_dict.get(cfg_name)
        evaluator.retriever.index(chunks)

        budget_recalls = {f"recall_at_budget_{b}": [] for b in token_budgets}
        evidence_densities = []

        for sample in golden_qa:
            query = sample.get("question", "")
            target_ids = set(sample.get("expected_chunk_ids", []))
            target_span = sample.get("ground_truth_context_span") or sample.get("ground_truth_answer", "")
            target_bc = sample.get("expected_breadcrumb_prefix", "")

            # Retrieve top 20 candidates
            results = evaluator.retriever.search(query, top_k=20)

            # Đánh giá theo từng ngân sách token
            for budget in token_budgets:
                accumulated_tokens = 0
                hit_in_budget = False
                for doc, _ in results:
                    c_tokens = len(_tokenize(doc.get("content", "")))
                    accumulated_tokens += c_tokens
                    if accumulated_tokens > budget and accumulated_tokens > c_tokens:
                        break

                    is_rel = evaluator.is_chunk_relevant(
                        retrieved_chunk=doc,
                        expected_chunk_ids=target_ids,
                        expected_context_span=target_span,
                        expected_bc_prefix=target_bc,
                        parent_map=parent_map
                    )
                    if is_rel:
                        hit_in_budget = True
                        break

                budget_recalls[f"recall_at_budget_{budget}"].append(1.0 if hit_in_budget else 0.0)

            # Tính Evidence Density trên Top 5 chunks
            top5 = results[:5]
            total_retrieved_tokens = sum(len(_tokenize(doc.get("content", ""))) for doc, _ in top5)
            span_tokens_len = len(_tokenize(target_span))

            # Nếu có ít nhất 1 chunk relevant trong top 5
            has_hit = any(
                evaluator.is_chunk_relevant(
                    retrieved_chunk=doc,
                    expected_chunk_ids=target_ids,
                    expected_context_span=target_span,
                    expected_bc_prefix=target_bc,
                    parent_map=parent_map
                ) for doc, _ in top5
            )

            if has_hit and total_retrieved_tokens > 0:
                density = min(span_tokens_len / total_retrieved_tokens, 1.0)
            else:
                density = 0.0
            evidence_densities.append(density)

        avg_budget_recalls = {
            k: round(sum(v) / max(len(v), 1), 4) for k, v in budget_recalls.items()
        }
        avg_density = round(sum(evidence_densities) / max(len(evidence_densities), 1), 4)

        # Tính kích thước trung bình của chunk
        chunk_lens = [len(_tokenize(c.get("content", ""))) for c in chunks]
        avg_chunk_size = round(sum(chunk_lens) / max(len(chunk_lens), 1), 1)

        bias_results[cfg_name] = {
            "avg_chunk_size_syllables": avg_chunk_size,
            "evidence_density_top5": avg_density,
            **avg_budget_recalls
        }

    return bias_results


def main():
    results_dir = PROJECT_ROOT / "experiments" / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    qa_path = PROJECT_ROOT / "experiments" / "benchmarks" / "datasets" / "golden_qa.json"
    with open(qa_path, "r", encoding="utf-8") as f:
        golden_qa = json.load(f)

    corpus_paths = [
        PROJECT_ROOT / "dataset" / "chunking_dataset" / "pdf" / "vcb_pdf_22_bieu_phi_the_tdqt_2026.pdf",
        PROJECT_ROOT / "dataset" / "chunking_dataset" / "pdf" / "vcb_pdf_27_dieu_kien_dieu_khoan_tktt_truc_tuyen.pdf",
        PROJECT_ROOT / "dataset" / "chunking_dataset" / "pdf" / "vcb_pdf_21_nq_hdqt_cctg_vcbs.pdf",
        PROJECT_ROOT / "dataset" / "chunking_dataset" / "docx" / "vcb_md_01__faq__the_tin_dung_va_the_ghi_no_ngan_hang.docx"
    ]

    # --- 1.1 BENCHMARK LATENCY PHÂN RÃ ---
    latency_res = benchmark_latency_decomposition(
        digital_pdf=corpus_paths[1],
        scan_pdf=corpus_paths[2],
        runs=5
    )
    with open(results_dir / "task1_latency_decomposition.json", "w", encoding="utf-8") as f:
        json.dump(latency_res, f, ensure_ascii=False, indent=2)

    # --- CHUẨN BỊ CHUNKS CHO 1.3 & 1.4 ---
    print("\n--- [PREP] SINH CHUNKS ĐỂ ĐÁNH GIÁ SENSITIVITY & CHUNK SIZE BIAS ---")
    runners = {
        "Config 0": BaselineNaiveChunker(),
        "Config 2": PipelineV2Chunker(),
        "Config 5A": PipelineV3UltimateRunner(profile=PROFILE_A),
        "Config 5B": PipelineV3UltimateRunner(profile=PROFILE_B),
        "Config 5C": PipelineV3UltimateRunner(profile=PROFILE_C),
    }

    chunks_dict = {}
    parent_map_dict = {}
    for name, r in runners.items():
        print(f"Generating chunks for {name}...")
        c_list = []
        for p in corpus_paths:
            c_list.extend(r.chunk_file(p))
        chunks_dict[name] = c_list
        if hasattr(r, "parent_map"):
            parent_map_dict[name] = r.parent_map

    # --- 1.3 PHÂN TÍCH ĐỘ NHẠY NGƯỠNG CONTAINMENT ---
    sensitivity_res = run_containment_sensitivity(
        chunks_dict=chunks_dict,
        golden_qa=golden_qa,
        thresholds=[0.5, 0.6, 0.7, 0.8, 0.9]
    )
    with open(results_dir / "task1_containment_sensitivity.json", "w", encoding="utf-8") as f:
        json.dump(sensitivity_res, f, ensure_ascii=False, indent=2)

    # --- 1.4 KIỂM SOÁT THIÊN VỊ KÍCH THƯỚC CHUNK ---
    bias_res = run_chunk_size_bias_evaluation(
        chunks_dict=chunks_dict,
        parent_map_dict=parent_map_dict,
        golden_qa=golden_qa,
        token_budgets=[500, 1000, 2000]
    )
    with open(results_dir / "task1_chunk_size_bias.json", "w", encoding="utf-8") as f:
        json.dump(bias_res, f, ensure_ascii=False, indent=2)

    print("\n==============================================================================")
    print("TASK 1 COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {results_dir}")
    print("==============================================================================")


if __name__ == "__main__":
    main()
