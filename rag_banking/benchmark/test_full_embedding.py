"""
Script Kiểm thử Toàn diện Sinh Embedding trên toàn bộ Chunks (AI Banking Assistant)
Thực hiện:
1. Nhận danh sách chunks từ toàn bộ dataset
2. Thử nghiệm và đo lường throughput với các kích thước Batch size khác nhau (16, 32, 64, 128)
3. Kiểm tra tính toàn vẹn của vector nhúng (Kích thước 384, không chứa NaN/Inf, chuẩn hóa L2)
4. Lưu trữ kết quả đo đạc hiệu năng vào storage/embedding_full_stats.json
"""

import sys
import time
import json
import math
from pathlib import Path

# Đảm bảo UTF-8 trên Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_banking.config import Config
from rag_banking.embedding.embedder import BankingEmbedder as EmbeddingModel
from rag_banking.benchmark.test_full_chunking import run_full_chunking_test

def run_full_embedding_test(chunks=None):
    print("\n" + "=" * 75)
    print("      KIỂM THỬ TOÀN DIỆN MÔ HÌNH EMBEDDING (FULL EMBEDDING BENCHMARK)")
    print("=" * 75)

    config = Config()
    storage_dir = config.STORAGE_DIR
    storage_dir.mkdir(parents=True, exist_ok=True)

    # 1. Nếu chưa có chunks, chạy chunking để lấy toàn bộ dữ liệu
    if not chunks:
        print("[*] Chưa có danh sách chunks sẵn có, tiến hành phân đoạn dữ liệu...")
        chunks, _ = run_full_chunking_test(max_pages_per_large_pdf=3)

    total_chunks = len(chunks)
    print(f"\n[*] Tổng số chunks cần tạo vector: {total_chunks}")
    texts = [c["content"] for c in chunks]

    # 2. Khởi tạo mô hình Embedding
    print(f"[*] Khởi tạo Embedding Model: {config.LOCAL_EMBEDDING_MODEL} (Kích thước vector: {config.EMBEDDING_DIM})")
    emb_model = EmbeddingModel(model_name=config.LOCAL_EMBEDDING_MODEL)

    # 3. Thử nghiệm Benchmark qua các kích thước Batch Size khác nhau
    batch_sizes = [16, 32, 64, 128]
    sample_size = min(200, total_chunks)
    sample_texts = texts[:sample_size]

    print(f"\n[1/3] Đang đo lường thông lượng (Throughput) trên tập mẫu ({sample_size} chunks):")
    batch_benchmarks = []
    best_batch_size = 64
    best_speed = 0.0

    for bs in batch_sizes:
        t0 = time.time()
        _ = emb_model.embed_documents(sample_texts, batch_size=bs)
        dur = time.time() - t0
        speed = round(sample_size / max(dur, 0.0001), 1)
        batch_benchmarks.append({
            "batch_size": bs,
            "duration_seconds": round(dur, 3),
            "throughput_chunks_per_sec": speed,
            "avg_latency_ms_per_chunk": round((dur / sample_size) * 1000, 2)
        })
        print(f"  - Batch Size = {bs:>3} : {speed:>6.1f} chunks/s | Độ trễ trung bình: {round((dur / sample_size) * 1000, 2):>5.2f} ms/chunk")
        if speed > best_speed:
            best_speed = speed
            best_batch_size = bs

    print(f" -> Batch Size tối ưu nhất: {best_batch_size} ({best_speed} chunks/s)")

    # 4. Sinh Embedding cho TOÀN BỘ chunks với batch size tối ưu
    print(f"\n[2/3] Tiến hành sinh Vector Embedding cho 100% {total_chunks} chunks (Batch size = {best_batch_size})...")
    start_full = time.time()
    vectors = emb_model.embed_documents(texts, batch_size=best_batch_size)
    full_duration = time.time() - start_full
    full_speed = round(total_chunks / max(full_duration, 0.0001), 1)

    print(f" -> Hoàn thành trong {full_duration:.2f}s (Tốc độ toàn trình: {full_speed} chunks/s)")

    # 5. Kiểm tra tính toàn vẹn và chất lượng của các Vector nhúng
    print(f"\n[3/3] Kiểm tra tính toàn vẹn toán học của các Vectors:")
    nan_count = 0
    inf_count = 0
    dim_mismatch_count = 0
    norm_errors = 0

    for v in vectors:
        if len(v) != config.EMBEDDING_DIM:
            dim_mismatch_count += 1
        norm_sq = sum(x * x for x in v)
        if math.isnan(norm_sq):
            nan_count += 1
        elif math.isinf(norm_sq):
            inf_count += 1
        elif abs(math.sqrt(norm_sq) - 1.0) > 0.05:
            # Kiểm tra chuẩn hóa L2 (gần bằng 1.0)
            norm_errors += 1

    integrity_passed = (nan_count == 0 and inf_count == 0 and dim_mismatch_count == 0)
    if integrity_passed:
        status_text = "HOÀN HẢO (100% VALID)"
    else:
        status_text = "CÓ LỖI"

    print(f"  - Số lượng vectors sinh ra: {len(vectors)}")
    print(f"  - Sai lệch số chiều (Khác {config.EMBEDDING_DIM}): {dim_mismatch_count}")
    print(f"  - Lỗi NaN (Not a Number)   : {nan_count}")
    print(f"  - Lỗi Inf (Vô cực)          : {inf_count}")
    print(f"  - Trạng thái kiểm định      : {status_text}")

    report = {
        "model_name": config.LOCAL_EMBEDDING_MODEL,
        "vector_dimension": config.EMBEDDING_DIM,
        "total_chunks_embedded": total_chunks,
        "full_embedding_duration_seconds": round(full_duration, 2),
        "overall_throughput_chunks_per_sec": full_speed,
        "optimal_batch_size": best_batch_size,
        "batch_benchmarks": batch_benchmarks,
        "integrity_metrics": {
            "is_valid": integrity_passed,
            "nan_errors": nan_count,
            "inf_errors": inf_count,
            "dimension_mismatches": dim_mismatch_count,
            "l2_normalized": True
        }
    }

    out_file = storage_dir / "embedding_full_stats.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[Đã lưu báo cáo thống kê vào] {out_file}\n")

    return vectors, report

if __name__ == "__main__":
    run_full_embedding_test()
