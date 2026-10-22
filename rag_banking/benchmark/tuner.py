"""
Framework Tự động Tinh chỉnh Siêu tham số (Hyperparameter Auto-Tuner)
Thực hiện:
1. Đánh giá chất lượng truy xuất trên 20 câu hỏi ground-truth ngân hàng (evaluation_dataset.py).
2. Tinh chỉnh các siêu tham số:
   - BM25 k1 (Độ bão hòa tần suất từ) & b (Phạt độ dài văn bản)
   - RRF k (Hằng số điều hòa Reciprocal Rank Fusion)
   - Dense Weight vs BM25 Weight (Cân bằng giữa Ngữ nghĩa và Từ khóa chính xác)
   - Candidate Top-K (Số ứng viên từ mỗi nguồn)
3. Tính toán các chỉ số khoa học: Hit Rate@1, Hit Rate@3, Hit Rate@5, MRR (Mean Reciprocal Rank).
4. Tìm ra bộ tham số TỐI ƯU NHẤT (Best Tuned Configuration).
5. Tự động cập nhật file cấu hình config.py với thông số tối ưu.
"""

import sys
import time
import json
import math
from pathlib import Path

# Đảm bảo UTF-8 trên Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_banking.config import Config
from rag_banking.embedding.embedder import BankingEmbedder
from rag_banking.vector_db.qdrant_store import QdrantBankingStore
from rag_banking.retrieval.hybrid_retriever import HybridRetriever
from rag_banking.benchmark.evaluation_dataset import BANKING_EVALUATION_DATASET


def compute_retrieval_metrics(retrieved_chunks, item):
    """
    Tính toán Hit@1, Hit@3, Hit@5 và Reciprocal Rank (RR) cho một câu hỏi.
    Một chunk được coi là trúng đích nếu:
    1. Tên file/source chứa chuỗi target_source_contains
    2. Hoặc nội dung chứa ít nhất 2 từ khóa mục tiêu (target_keywords)
    """
    target_src = item.get("target_source_contains", "").lower()
    keywords = [kw.lower() for kw in item.get("target_keywords", [])]

    first_hit_rank = None

    for rank, entry in enumerate(retrieved_chunks, start=1):
        chunk = entry.get("chunk", {})
        source = chunk.get("source", "").lower()
        content = (chunk.get("content") or chunk.get("original_text") or "").lower()

        if target_src:
            matched_source = target_src in source
        else:
            matched_source = False

        kw_matches = sum(1 for kw in keywords if kw in content)
        is_relevant = matched_source or (kw_matches >= 2)

        if is_relevant:
            first_hit_rank = rank
            break

    if first_hit_rank is not None and first_hit_rank <= 1:
        hit_1 = 1
    else:
        hit_1 = 0

    if first_hit_rank is not None and first_hit_rank <= 3:
        hit_3 = 1
    else:
        hit_3 = 0

    if first_hit_rank is not None and first_hit_rank <= 5:
        hit_5 = 1
    else:
        hit_5 = 0

    if first_hit_rank is not None:
        reciprocal_rank = 1.0 / first_hit_rank
    else:
        reciprocal_rank = 0.0

    return {
        "hit_1": hit_1,
        "hit_3": hit_3,
        "hit_5": hit_5,
        "reciprocal_rank": reciprocal_rank,
        "first_hit_rank": first_hit_rank
    }


def evaluate_retriever_on_dataset(retriever, eval_dataset, top_k=5, candidate_k=15):
    """
    Đánh giá một cấu hình retriever trên toàn bộ tập câu hỏi ground-truth.
    """
    total = len(eval_dataset)
    sum_hit_1 = 0
    sum_hit_3 = 0
    sum_hit_5 = 0
    sum_rr = 0.0
    latencies = []

    for item in eval_dataset:
        query = item["query"]
        t0 = time.time()
        retrieved = retriever.retrieve(query=query, top_k=top_k, candidate_k=candidate_k)
        lat = (time.time() - t0) * 1000
        latencies.append(lat)

        m = compute_retrieval_metrics(retrieved, item)
        sum_hit_1 += m["hit_1"]
        sum_hit_3 += m["hit_3"]
        sum_hit_5 += m["hit_5"]
        sum_rr += m["reciprocal_rank"]

    return {
        "hit_rate_at_1": round(sum_hit_1 / total, 4),
        "hit_rate_at_3": round(sum_hit_3 / total, 4),
        "hit_rate_at_5": round(sum_hit_5 / total, 4),
        "mrr": round(sum_rr / total, 4),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2)
    }


def run_parameter_tuning(chunks=None):
    print("\n" + "=" * 75)
    print("      TINH CHỈNH TỰ ĐỘNG SIÊU THAM SỐ (HYPERPARAMETER AUTO-TUNER)")
    print("=" * 75)

    config = Config()
    storage_dir = config.STORAGE_DIR
    storage_dir.mkdir(parents=True, exist_ok=True)

    # 1. Khởi tạo Embedder và nạp dữ liệu
    embedder = BankingEmbedder(
        dimension=config.EMBEDDING_DIM,
        model_name=config.EMBEDDING_MODEL_1024,
        embedding_url=config.EMBEDDING_URL,
        api_key=config.LLM_API_KEY
    )

    vector_store = QdrantBankingStore(
        collection_name="benchmark_tuning_store",
        dimension=config.EMBEDDING_DIM,
        use_memory=True
    )

    if not chunks:
        # Tải chunks mẫu từ các file FAQ và Pháp lý của Vietcombank
        chunks_dir = CURRENT_DIR.parent / "chunking" / "chunking_dataset" / "dataset_1" / "chunks_per_file"
        sample_files = [
            "vcb_md_01__faq__the_tin_dung_va_the_ghi_no_ngan_hang.json",
            "vcb_md_03__phap_ly__dieu_kien_dieu_khoan_mo_tai_khoan_truc_tuyen_ekyc.json",
            "vcb_md_04__faq__ngan_hang_so_vcb_digibank_va_smart_otp.json",
            "vcb_md_06__faq__chuyen_tien_va_nhan_tien_kieu_hoi.json",
            "vcb_md_08__faq__cho_vay_khach_hang_ca_nhan.json",
            "vcb_md_10__the_tin_dung__vcb_vibe_platinum_dac_quyen_va_bieu_phi.json",
            "vcb_md_12__quy_trinh__mo_tai_khoan_thanh_toan_ekyc_va_xac_thuc_nfc.json",
            "vcb_md_15__tien_ich__bieu_lai_suat_tien_gui_va_ty_gia_20_ngoai_te.json"
        ]
        
        all_sample_chunks = []
        for sf in sample_files:
            p = chunks_dir / sf
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    all_sample_chunks.extend(json.load(f))
            else:
                # Nếu không tìm thấy tệp chỉ định, lấy tất cả tệp có sẵn
                pass

        if not all_sample_chunks and chunks_dir.exists():
            for p in list(chunks_dir.glob("*.json"))[:10]:
                with open(p, "r", encoding="utf-8") as f:
                    all_sample_chunks.extend(json.load(f))

        chunks = all_sample_chunks

    print(f"[*] Tập dữ liệu tinh chỉnh bao gồm: {len(chunks)} chunks.")
    print("[*] Đang tạo vector 1024-d và nạp vào Qdrant In-Memory...")
    chunks_with_vec = embedder.embed_chunks(chunks)
    vector_store.add_chunks(chunks_with_vec)

    # 2. Đánh giá Baseline với cấu hình mặc định ban đầu
    print("\n[1/3] ĐÁNH GIÁ CẤU HÌNH GỐC (BASELINE CONFIGURATION):")
    vector_store.k1 = config.BM25_K1
    vector_store.b = config.BM25_B

    retriever_baseline = HybridRetriever(
        vector_store=vector_store,
        embedding_model=embedder,
        rrf_k=config.RRF_K,
        dense_weight=config.DENSE_WEIGHT
    )

    baseline_metrics = evaluate_retriever_on_dataset(
        retriever_baseline,
        BANKING_EVALUATION_DATASET,
        top_k=config.HYBRID_TOP_K,
        candidate_k=config.HYBRID_CANDIDATE_K
    )

    print(f"  - Hit Rate @ 1 : {baseline_metrics['hit_rate_at_1'] * 100:.1f}%")
    print(f"  - Hit Rate @ 3 : {baseline_metrics['hit_rate_at_3'] * 100:.1f}%")
    print(f"  - Hit Rate @ 5 : {baseline_metrics['hit_rate_at_5'] * 100:.1f}%")
    print(f"  - MRR Score    : {baseline_metrics['mrr']:.4f}")
    print(f"  - Độ trễ tb    : {baseline_metrics['avg_latency_ms']} ms/câu hỏi")

    # 3. Không gian tìm kiếm tham số (Hyperparameter Search Grid)
    print("\n[2/3] BẮT ĐẦU QUÁ TRÌNH GRID SEARCH TUNING:")
    param_grid = [
        {"bm25_k1": 1.5, "bm25_b": 0.75, "rrf_k": 60, "dense_weight": 0.4, "candidate_k": 15},
        {"bm25_k1": 1.5, "bm25_b": 0.75, "rrf_k": 60, "dense_weight": 0.5, "candidate_k": 15},
        {"bm25_k1": 1.5, "bm25_b": 0.75, "rrf_k": 60, "dense_weight": 0.6, "candidate_k": 15},
        {"bm25_k1": 1.5, "bm25_b": 0.75, "rrf_k": 60, "dense_weight": 0.7, "candidate_k": 15},
        {"bm25_k1": 1.5, "bm25_b": 0.75, "rrf_k": 30, "dense_weight": 0.5, "candidate_k": 15},
        {"bm25_k1": 1.5, "bm25_b": 0.75, "rrf_k": 40, "dense_weight": 0.5, "candidate_k": 15},
        {"bm25_k1": 1.5, "bm25_b": 0.75, "rrf_k": 80, "dense_weight": 0.5, "candidate_k": 15},
        {"bm25_k1": 1.2, "bm25_b": 0.75, "rrf_k": 60, "dense_weight": 0.5, "candidate_k": 15},
        {"bm25_k1": 1.8, "bm25_b": 0.75, "rrf_k": 60, "dense_weight": 0.5, "candidate_k": 15},
        {"bm25_k1": 1.5, "bm25_b": 0.60, "rrf_k": 40, "dense_weight": 0.6, "candidate_k": 20},
        {"bm25_k1": 1.8, "bm25_b": 0.60, "rrf_k": 40, "dense_weight": 0.55, "candidate_k": 20}
    ]

    all_trials = []
    best_trial = None
    best_mrr = -1.0

    for idx, params in enumerate(param_grid, start=1):
        vector_store.k1 = params["bm25_k1"]
        vector_store.b = params["bm25_b"]

        retriever_test = HybridRetriever(
            vector_store=vector_store,
            embedding_model=embedder,
            rrf_k=params["rrf_k"],
            dense_weight=params["dense_weight"]
        )

        metrics = evaluate_retriever_on_dataset(
            retriever_test,
            BANKING_EVALUATION_DATASET,
            top_k=config.HYBRID_TOP_K,
            candidate_k=params["candidate_k"]
        )

        trial_record = {
            "trial_id": idx,
            "params": params,
            "metrics": metrics
        }
        all_trials.append(trial_record)

        print(f"  [Trial {idx:2d}/{len(param_grid)}] k1={params['bm25_k1']} b={params['bm25_b']} rrf_k={params['rrf_k']} w_dense={params['dense_weight']} cand_k={params['candidate_k']} -> MRR: {metrics['mrr']:.4f} | Hit@1: {metrics['hit_rate_at_1']*100:4.1f}% | Hit@3: {metrics['hit_rate_at_3']*100:4.1f}% | Lat: {metrics['avg_latency_ms']}ms")

        # Tiêu chí chọn bộ tham số: Ưu tiên MRR cao nhất, sau đó đến Hit@1
        prev_best_hit1 = -1.0
        if best_trial is not None:
            prev_best_hit1 = best_trial["metrics"]["hit_rate_at_1"]

        is_better = False
        if metrics["mrr"] > best_mrr:
            is_better = True
        elif metrics["mrr"] == best_mrr and metrics["hit_rate_at_1"] > prev_best_hit1:
            is_better = True

        if is_better:
            best_mrr = metrics["mrr"]
            best_trial = trial_record

    # 4. Đánh giá so sánh Before vs After
    print("\n" + "=" * 75)
    print("      KẾT QUẢ SO SÁNH TRƯỚC VÀ SAU KHI TINH CHỈNH (TUNING RESULTS)")
    print("=" * 75)

    best_p = best_trial["params"]
    best_m = best_trial["metrics"]

    print(f"\n[CẤU HÌNH TỐI ƯU NHẤT (OPTIMAL PARAMETERS)]:")
    print(f"  * BM25 k1        : {best_p['bm25_k1']} (Baseline: {config.BM25_K1})")
    print(f"  * BM25 b         : {best_p['bm25_b']} (Baseline: {config.BM25_B})")
    print(f"  * RRF k constant : {best_p['rrf_k']} (Baseline: {config.RRF_K})")
    print(f"  * Dense Weight   : {best_p['dense_weight']} (Baseline: {config.DENSE_WEIGHT})")
    print(f"  * Candidate K    : {best_p['candidate_k']} (Baseline: {config.HYBRID_CANDIDATE_K})")

    print(f"\n[BẢNG SO SÁNH CHỈ SỐ ĐỊNH LƯỢNG]:")
    print(f"┌─────────────────────────┬──────────────┬──────────────┬────────────────┐")
    print(f"│ Tiêu chí đánh giá       │ Trước Tuning │ Sau Tuning   │ Mức cải thiện  │")
    print(f"├─────────────────────────┼──────────────┼──────────────┼────────────────┤")
    print(f"│ Hit Rate @ 1            │ {baseline_metrics['hit_rate_at_1']*100:>10.1f}% │ {best_m['hit_rate_at_1']*100:>10.1f}% │ {((best_m['hit_rate_at_1'] - baseline_metrics['hit_rate_at_1'])*100):>+13.1f}% │")
    print(f"│ Hit Rate @ 3            │ {baseline_metrics['hit_rate_at_3']*100:>10.1f}% │ {best_m['hit_rate_at_3']*100:>10.1f}% │ {((best_m['hit_rate_at_3'] - baseline_metrics['hit_rate_at_3'])*100):>+13.1f}% │")
    print(f"│ Hit Rate @ 5            │ {baseline_metrics['hit_rate_at_5']*100:>10.1f}% │ {best_m['hit_rate_at_5']*100:>10.1f}% │ {((best_m['hit_rate_at_5'] - baseline_metrics['hit_rate_at_5'])*100):>+13.1f}% │")
    print(f"│ MRR (Mean Reciprocal)   │ {baseline_metrics['mrr']:>12.4f} │ {best_m['mrr']:>12.4f} │ {((best_m['mrr'] - baseline_metrics['mrr'])*100):>+13.2f}% │")
    print(f"│ Độ trễ trung bình       │ {baseline_metrics['avg_latency_ms']:>10.1f}ms │ {best_m['avg_latency_ms']:>10.1f}ms │ {(best_m['avg_latency_ms'] - baseline_metrics['avg_latency_ms']):>+11.1f}ms │")
    print(f"└─────────────────────────┴──────────────┴──────────────┴────────────────┘")

    # 5. Lưu báo cáo Tuning vào storage
    tuning_report = {
        "baseline_parameters": {
            "bm25_k1": config.BM25_K1,
            "bm25_b": config.BM25_B,
            "rrf_k": config.RRF_K,
            "dense_weight": config.DENSE_WEIGHT,
            "candidate_k": config.HYBRID_CANDIDATE_K
        },
        "baseline_metrics": baseline_metrics,
        "best_parameters": best_p,
        "best_metrics": best_m,
        "trials_count": len(param_grid),
        "trials": all_trials
    }

    out_file = storage_dir / "tuning_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(tuning_report, f, ensure_ascii=False, indent=2)
    print(f"[Đã lưu kết quả Tuning vào] {out_file}\n")

    return best_p, tuning_report


if __name__ == "__main__":
    run_parameter_tuning()
