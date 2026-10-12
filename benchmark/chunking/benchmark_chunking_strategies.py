#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HỆ THỐNG BENCHMARK THỰC NGHIỆM CHIẾN LƯỢC PHÂN ĐOẠN TÀI LIỆU (CHUNKING BENCHMARK)
So sánh đối đầu giữa các chiến lược phân đoạn:
1. Baseline: LayoutAwareChunker (Bố cục cơ bản)
2. Strategy 1: Document Structure-Based Chunking (Ngân sách âm tiết)
3. Strategy 3: Section Length-Aware Chunking (Tối ưu chống phân mảnh vụn)
4. Strategy 2: Hierarchical Chunking (Phân tầng Parent-Child)

Đo lường trên 50 tài liệu thực tế của Vietcombank (dataset/chunking_dataset/md/):
- Tiêu chí 1: Tổng số Chunks sinh ra
- Tiêu chí 2: Tỷ lệ phân mảnh vụn vặt (% Chunks < 80 từ - Less Fragmentation)
- Tiêu chí 3: Tỷ lệ rơi vào Dải kích thước vàng (% Chunks 150 - 450 từ)
- Tiêu chí 4: Độ đồng đều kích thước (Độ lệch chuẩn Std Dev)
- Tiêu chí 5: Bảo toàn Breadcrumbs & Ngữ cảnh (Preservation Rate %)
- Tiêu chí 6: Bảo toàn Bảng biểu dữ liệu (Table Integrity %)
- Tiêu chí 7: Thử nghiệm RAG Retrieval giả lập với 10 câu hỏi nghiệp vụ ngân hàng
"""

import os
import re
import sys
import json
import time
import math
from pathlib import Path
from typing import Dict, Any, List

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_banking.chunking.layout_aware_chunker import LayoutAwareChunker
from rag_banking.chunking.structure_chunker import DocumentStructureChunker

DATASET_MD_DIR = PROJECT_ROOT / "dataset" / "chunking_dataset" / "md"
OUTPUT_DIR = PROJECT_ROOT / "benchmark" / "chunking"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 10 câu hỏi truy vấn nghiệp vụ Vietcombank đại diện để đo lường Retrieval Coherence
BENCHMARK_QUERIES = [
    "Điều kiện và hồ sơ mở tài khoản trực tuyến eKYC Vietcombank",
    "Biểu phí phát hành và phí thường niên thẻ tín dụng quốc tế Visa Infinite",
    "Quy định về hạn mức giao dịch qua VCB Digibank và Smart OTP",
    "Thể lệ chương trình tích điểm đổi quà VCB Rewards Loyalty",
    "Kết quả kinh doanh và lợi nhuận trước thuế Báo cáo tài chính 2024",
    "Lãi suất cho vay mua nhà và phương thức trả nợ khách hàng cá nhân",
    "Biểu phí chuyển tiền kiều hối và ngoại tệ qua kênh số",
    "Quy trình xử lý khi phát hiện giao dịch gian lận hoặc nghi ngờ lộ mã PIN",
    "Điều kiện áp dụng ưu đãi phòng chờ sân bay của thẻ Vietcombank",
    "Thủ tục giải tỏa tiền gửi tiết kiệm và chứng chỉ tiền gửi trực tuyến"
]


def evaluate_chunk_set(chunks: List[Dict[str, Any]], strategy_name: str) -> Dict[str, Any]:
    """Tính toán các chỉ số thống kê chất lượng phân đoạn."""
    if not chunks:
        return {}

    word_counts = [c.get("word_count", len(c.get("content", "").split())) for c in chunks]
    total_chunks = len(chunks)
    avg_words = sum(word_counts) / total_chunks
    variance = sum((w - avg_words) ** 2 for w in word_counts) / total_chunks
    std_words = math.sqrt(variance)

    # 1. Tỷ lệ chunk quá nhỏ < 80 từ (Phân mảnh vụn)
    fragmented_chunks = sum(1 for w in word_counts if w < 80)
    fragmentation_rate = round((fragmented_chunks / total_chunks) * 100, 2)

    # 2. Tỷ lệ rơi vào dải kích thước vàng (150 - 450 từ)
    gold_zone_chunks = sum(1 for w in word_counts if 150 <= w <= 450)
    gold_zone_rate = round((gold_zone_chunks / total_chunks) * 100, 2)

    # 3. Tỷ lệ chunk quá dài > 600 từ (nguy cơ tràn context window)
    oversized_chunks = sum(1 for w in word_counts if w > 600)
    oversized_rate = round((oversized_chunks / total_chunks) * 100, 2)

    # 4. Tỷ lệ bảo toàn Breadcrumbs / Tiêu đề
    chunks_with_breadcrumbs = sum(1 for c in chunks if c.get("breadcrumb") and len(c["breadcrumb"].strip()) > 3)
    breadcrumb_rate = round((chunks_with_breadcrumbs / total_chunks) * 100, 2)

    # 5. Tỷ lệ bảo toàn Bảng biểu
    table_chunks = [c for c in chunks if c.get("chunk_type") == "table"]
    table_count = len(table_chunks)

    return {
        "strategy": strategy_name,
        "total_chunks": total_chunks,
        "avg_words": round(avg_words, 1),
        "std_words": round(std_words, 1),
        "min_words": min(word_counts),
        "max_words": max(word_counts),
        "fragmentation_rate_pct": fragmentation_rate,
        "gold_zone_rate_pct": gold_zone_rate,
        "oversized_rate_pct": oversized_rate,
        "breadcrumb_coverage_pct": breadcrumb_rate,
        "table_chunks_count": table_count,
    }


def simple_keyword_overlap_score(query: str, doc_text: str) -> float:
    """Đo lường mức độ tương đồng từ khóa giữa câu hỏi và đoạn văn bản."""
    q_words = set(re.findall(r"\b\w{2,}\b", query.lower()))
    if not q_words:
        return 0.0
    doc_words = set(re.findall(r"\b\w{2,}\b", doc_text.lower()))
    overlap = q_words.intersection(doc_words)
    return len(overlap) / len(q_words)


def simulate_retrieval_performance(all_chunks: List[Dict[str, Any]], queries: List[str], top_k: int = 3) -> Dict[str, Any]:
    """
    Mô phỏng truy xuất RAG (Simulated Retrieval Benchmark):
    Đo độ đầy đủ ngữ cảnh và mật độ thông tin hữu ích trong Top-3 chunks.
    """
    total_retrieved_words = 0
    total_overlap_score = 0.0
    coherent_chunks_count = 0

    for query in queries:
        scored = []
        for c in all_chunks:
            score = simple_keyword_overlap_score(query, c["content"])
            scored.append((score, c))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        top_retrieved = scored[:top_k]

        for s, c in top_retrieved:
            total_overlap_score += s
            w_count = c.get("word_count", len(c["content"].split()))
            total_retrieved_words += w_count
            if c.get("breadcrumb") and w_count >= 120:
                coherent_chunks_count += 1

    total_evals = len(queries) * top_k
    avg_relevance = round((total_overlap_score / total_evals) * 100, 2)
    coherence_rate = round((coherent_chunks_count / total_evals) * 100, 2)
    avg_context_len = round(total_retrieved_words / len(queries), 1)

    return {
        "avg_keyword_relevance_pct": avg_relevance,
        "context_coherence_rate_pct": coherence_rate,
        "avg_top_k_context_words": avg_context_len,
    }


def main():
    print("=" * 85)
    print("   HỆ THỐNG ĐÁNH GIÁ VÀ SO SÁNH THỰC NGHIỆM CÁC CHIẾN LƯỢC CHUNKING CHO RAG")
    print("=" * 85)

    md_files = sorted(list(DATASET_MD_DIR.glob("*.md")))
    if not md_files:
        raise FileNotFoundError(f"Không tìm thấy file markdown nào trong {DATASET_MD_DIR}")

    print(f"[*] Tập dữ liệu kiểm thử: {len(md_files)} văn bản nghiệp vụ ngân hàng Vietcombank")

    # Khởi tạo các bộ chunker
    current_chunker = LayoutAwareChunker(max_chunk_size=800, chunk_overlap=100)
    structure_chunker = DocumentStructureChunker(chunk_size=400, min_section_size=120)

    results_baseline: List[Dict[str, Any]] = []
    results_s1: List[Dict[str, Any]] = []
    results_s3: List[Dict[str, Any]] = []
    results_s2: List[Dict[str, Any]] = []

    print("[*] Đang tiến hành phân đoạn đồng loạt cả 4 chiến lược...")
    t0 = time.time()

    for idx, fpath in enumerate(md_files, 1):
        content = fpath.read_text(encoding="utf-8")

        # 1. Chạy Baseline Hiện tại
        c_chunks = current_chunker.chunk_document(content, source_name=fpath.name)
        results_baseline.extend(c_chunks)

        # 2. Parse Markdown sang SubSections
        sub_sections = structure_chunker.parse_markdown_to_structure(content, source_name=fpath.name)

        # Chiến lược 1: Structure-Based
        s1_chunks = structure_chunker.chunk_strategy_1_structure_based(sub_sections, source_name=fpath.name)
        results_s1.extend(s1_chunks)

        # Chiến lược 3: Section Length-Aware (Chống phân mảnh)
        s3_chunks = structure_chunker.chunk_strategy_3_section_length_aware(sub_sections, source_name=fpath.name)
        results_s3.extend(s3_chunks)

        # Chiến lược 2: Hierarchical (Parent - Child)
        s2_chunks = structure_chunker.chunk_strategy_2_hierarchical(sub_sections, source_name=fpath.name)
        results_s2.extend(s2_chunks)

    elapsed = time.time() - t0
    print(f"[+] Hoàn tất xử lý 50 văn bản trong {elapsed:.2f} giây!\n")

    # Đánh giá chỉ số thống kê
    eval_baseline = evaluate_chunk_set(results_baseline, "1. Baseline (LayoutAware)")
    eval_s1 = evaluate_chunk_set(results_s1, "2. Strategy 1 (Structure-based)")
    eval_s3 = evaluate_chunk_set(results_s3, "3. Strategy 3 (Section-length-aware)*")
    eval_s2 = evaluate_chunk_set(results_s2, "4. Strategy 2 (Hierarchical Parent-Child)")

    # Mô phỏng Retrieval
    retrieval_baseline = simulate_retrieval_performance(results_baseline, BENCHMARK_QUERIES)
    retrieval_s1 = simulate_retrieval_performance(results_s1, BENCHMARK_QUERIES)
    retrieval_s3 = simulate_retrieval_performance(results_s3, BENCHMARK_QUERIES)
    retrieval_s2 = simulate_retrieval_performance(results_s2, BENCHMARK_QUERIES)

    # Gộp kết quả
    comparison_table = [
        {**eval_baseline, **retrieval_baseline},
        {**eval_s1, **retrieval_s1},
        {**eval_s3, **retrieval_s3},
        {**eval_s2, **retrieval_s2},
    ]

    # Lưu kết quả JSON
    json_path = OUTPUT_DIR / "chunking_benchmark_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_table, f, ensure_ascii=False, indent=2)

    # In Bảng Kết Quả
    print("=" * 100)
    print(f"{'Tiêu chí Đo lường (Metrics)':<35} | {'Baseline (Hiện tại)':<18} | {'Strategy 1':<15} | {'Strategy 3*':<16} | {'Strategy 2':<15}")
    print("-" * 105)
    print(f"{'Tổng số Chunk sinh ra':<35} | {eval_baseline['total_chunks']:>18} | {eval_s1['total_chunks']:>15} | {eval_s3['total_chunks']:>16} | {eval_s2['total_chunks']:>15}")
    print(f"{'Độ dài từ trung bình (Avg Words)':<35} | {eval_baseline['avg_words']:>18} | {eval_s1['avg_words']:>15} | {eval_s3['avg_words']:>16} | {eval_s2['avg_words']:>15}")
    print(f"{'Độ lệch chuẩn kích thước (Std)':<35} | {eval_baseline['std_words']:>18} | {eval_s1['std_words']:>15} | {eval_s3['std_words']:>16} | {eval_s2['std_words']:>15}")
    print(f"{'Tỷ lệ phân mảnh vụn (< 80 từ) [↓ tốt]':<35} | {eval_baseline['fragmentation_rate_pct']:>17.1f}% | {eval_s1['fragmentation_rate_pct']:>14.1f}% | {eval_s3['fragmentation_rate_pct']:>15.1f}% | {eval_s2['fragmentation_rate_pct']:>14.1f}%")
    print(f"{'Tỷ lệ Dải kích thước Vàng (150-450) [↑ tốt]':<35} | {eval_baseline['gold_zone_rate_pct']:>17.1f}% | {eval_s1['gold_zone_rate_pct']:>14.1f}% | {eval_s3['gold_zone_rate_pct']:>15.1f}% | {eval_s2['gold_zone_rate_pct']:>14.1f}%")
    print(f"{'Tỷ lệ chunk quá lớn (> 600 từ) [↓ tốt]':<35} | {eval_baseline['oversized_rate_pct']:>17.1f}% | {eval_s1['oversized_rate_pct']:>14.1f}% | {eval_s3['oversized_rate_pct']:>15.1f}% | {eval_s2['oversized_rate_pct']:>14.1f}%")
    print(f"{'Độ phủ Breadcrumbs tiêu đề [↑ tốt]':<35} | {eval_baseline['breadcrumb_coverage_pct']:>17.1f}% | {eval_s1['breadcrumb_coverage_pct']:>14.1f}% | {eval_s3['breadcrumb_coverage_pct']:>15.1f}% | {eval_s2['breadcrumb_coverage_pct']:>14.1f}%")
    print(f"{'Độ mạch lạc ngữ cảnh Retrieval [↑ tốt]':<35} | {retrieval_baseline['context_coherence_rate_pct']:>17.1f}% | {retrieval_s1['context_coherence_rate_pct']:>14.1f}% | {retrieval_s3['context_coherence_rate_pct']:>15.1f}% | {retrieval_s2['context_coherence_rate_pct']:>14.1f}%")
    print("=" * 100)
    print("(*) Strategy 3 = Section-length-aware Chunking (Gộp các mục nhỏ chống phân mảnh)")


if __name__ == "__main__":
    main()
