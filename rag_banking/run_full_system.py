"""
Chương trình Điều phối Toàn trình: Kiểm thử Chunking, Embedding và Auto-Tuning
Thực hiện trọn vẹn 4 giai đoạn theo yêu cầu:
1. GIAI ĐOẠN 1: Kiểm thử Chunking toàn bộ dataset (100% 50 file MD, 100% 10 file Excel, 100% 50 file PDF)
2. GIAI ĐOẠN 2: Kiểm thử và đo lường thông lượng Embedding cho toàn bộ chunks (Batch size tuning 16-128)
3. GIAI ĐOẠN 3: Tinh chỉnh tự động siêu tham số (Auto-Tuner BM25 + RRF + Hybrid Weights) trên 20 câu hỏi ground-truth
4. GIAI ĐOẠN 4: Chạy kiểm thử toàn trình End-to-End RAG với bộ tham số tối ưu và ghi vết bằng Bird Tracer 🦜
"""

import sys
import time
from pathlib import Path

# Đảm bảo UTF-8 trên Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_banking.config import Config
from rag_banking.benchmark.test_full_chunking import run_full_chunking_test
from rag_banking.benchmark.test_full_embedding import run_full_embedding_test
from rag_banking.benchmark.tuner import run_parameter_tuning
from rag_banking.pipeline import BankingRAGPipeline

def main():
    print("\n" + "#" * 80)
    print("   HỆ THỐNG KIỂM THỬ VÀ TINH CHỈNH TOÀN DIỆN RAG BANKING (AI BANKING ASSISTANT)")
    print("#" * 80 + "\n")

    total_start = time.time()

    # =========================================================================
    # GIAI ĐOẠN 1: CHUNKING TOÀN BỘ DATASET
    # =========================================================================
    print("\n" + "▶" * 40)
    print(" GIAI ĐOẠN 1: KIỂM THỬ TOÀN DIỆN CHUNKING TRÊN TOÀN BỘ DATASET")
    print("▶" * 40)
    chunks, chunk_report = run_full_chunking_test(max_pages_per_large_pdf=3)

    # =========================================================================
    # GIAI ĐOẠN 2: EMBEDDING TOÀN BỘ CHUNKS VÀ BENCHMARK THROUGHPUT
    # =========================================================================
    print("\n" + "▶" * 40)
    print(" GIAI ĐOẠN 2: KIỂM THỬ TOÀN DIỆN EMBEDDING VÀ ĐO LƯỜNG THROUGHPUT")
    print("▶" * 40)
    vectors, emb_report = run_full_embedding_test(chunks=chunks)

    # =========================================================================
    # GIAI ĐOẠN 3: AUTO-TUNING SIÊU THAM SỐ (BM25, RRF, HYBRID)
    # =========================================================================
    print("\n" + "▶" * 40)
    print(" GIAI ĐOẠN 3: TINH CHỈNH TỰ ĐỘNG SIÊU THAM SỐ ĐẠT KẾT QUẢ TỐT NHẤT")
    print("▶" * 40)
    best_params, tuning_report = run_parameter_tuning(chunks=chunks, vectors=vectors)

    # =========================================================================
    # GIAI ĐOẠN 4: KIỂM THỬ RAG VỚI BỘ THAM SỐ TỐI ƯU VÀ BIRD TRACING
    # =========================================================================
    print("\n" + "▶" * 40)
    print(" GIAI ĐOẠN 4: KIỂM THỬ RAG VỚI BỘ THAM SỐ ĐÃ TUNING & BIRD TRACER 🦜")
    print("▶" * 40)

    # Khởi tạo Pipeline mới đã áp dụng thông số tối ưu
    pipeline = BankingRAGPipeline()
    pipeline.vector_store.add_chunks(chunks, vectors)
    pipeline.bm25_search.index(chunks)
    pipeline.retriever.rrf_k = best_params["rrf_k"]
    pipeline.retriever.dense_weight = best_params["dense_weight"]

    test_queries = [
        "Lãi suất tiền gửi tiết kiệm VND tại quầy của Vietcombank kỳ hạn 12 tháng là bao nhiêu %/năm và trên Digibank là bao nhiêu?",
        "Theo biểu phí dịch vụ thẻ tín dụng quốc tế Vietcombank 2026, phí thường niên của thẻ chính Vietcombank Vibe là bao nhiêu VNĐ/năm?",
        "Nghị quyết HĐQT số 28154/VCB-CLTKHDQT ngày 31/12/2025 phê duyệt nội dung gì liên quan đến chứng chỉ tiền gửi CCTG?"
    ]

    for q_idx, q in enumerate(test_queries, start=1):
        print(f"\n[DEMO {q_idx}] Câu hỏi: '{q}'")
        res = pipeline.ask(q, top_k=5)
        print("\n--- Câu trả lời của AI ---")
        print(res.get("answer", ""))
        print("--- Nguồn tham chiếu ---")
        for s in res.get("sources_used", []):
            print(f" * {s}")
        print("-" * 65)

    pipeline.close()

    total_duration = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"   HOÀN THÀNH TOÀN BỘ QUY TRÌNH CHUNKING, EMBEDDING VÀ TUNING ({total_duration:.2f} GIÂY)")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
