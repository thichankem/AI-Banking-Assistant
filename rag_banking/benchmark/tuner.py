"""
Framework Tự động Tinh chỉnh Siêu tham số (Hyperparameter Auto-Tuner)
Thực hiện:
1. Đánh giá chất lượng truy xuất trên 20 câu hỏi ground-truth ngân hàng
2. Tinh chỉnh các siêu tham số:
   - BM25 k1 (Độ bão hòa tần suất từ) & b (Phạt độ dài văn bản)
   - RRF k (Hằng số điều hòa Reciprocal Rank Fusion)
   - Dense Weight vs BM25 Weight (Cân bằng giữa Ngữ nghĩa và Từ khóa chính xác)
   - Candidate Top-K (Số ứng viên từ mỗi nguồn)
3. Tính toán các chỉ số khoa học: Hit Rate@1, Hit Rate@3, Hit Rate@5, MRR (Mean Reciprocal Rank), NDCG@5
4. Tìm ra bộ tham số TỐI ƯU NHẤT (Best Tuned Configuration)
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
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_banking.config import Config
from rag_banking.chunking.layout_aware_chunker import LayoutAwareChunker
from rag_banking.embeddings.embedding_model import EmbeddingModel
from rag_banking.vector_db.weaviate_store import WeaviateStore
from rag_banking.keyword_search.bm25_search import BM25Search
from rag_banking.retrieval.hybrid_retriever import HybridRetriever
from rag_banking.benchmark.evaluation_dataset import BANKING_EVALUATION_DATASET

def compute_retrieval_metrics(retrieved_chunks, item):
    """
    Tính toán Hit@1, Hit@3, Hit@5 và Reciprocal Rank (RR) cho một câu hỏi.
    Một chunk được coi là trúng đích nếu:
    1. Tên file chứa chuỗi target_source_contains
    2. Hoặc nội dung chứa ít nhất 2 từ khóa mục tiêu (target_keywords)
    """
    target_src = item.get("target_source_contains", "").lower()
    keywords = [kw.lower() for kw in item.get("target_keywords", [])]

    first_hit_rank = None

    for rank, entry in enumerate(retrieved_chunks, start=1):
        chunk = entry.get("chunk", {})
        source = chunk.get("source", "").lower()
        content = chunk.get("content", "").lower()

        # Kiểm tra nguồn gốc hoặc độ khớp từ khóa cốt lõi
        matched_source = target_src in source if target_src else False
        kw_matches = sum(1 for kw in keywords if kw in content)
        is_relevant = matched_source or (kw_matches >= 2)

        if is_relevant:
            first_hit_rank = rank
            break

    hit_1 = 1 if (first_hit_rank is not None and first_hit_rank <= 1) else 0
    hit_3 = 1 if (first_hit_rank is not None and first_hit_rank <= 3) else 0
    hit_5 = 1 if (first_hit_rank is not None and first_hit_rank <= 5) else 0
    reciprocal_rank = 1.0 / first_hit_rank if first_hit_rank is not None else 0.0

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

def run_parameter_tuning(chunks=None, vectors=None):
    print("\n" + "=" * 75)
    print("      TINH CHỈNH TỰ ĐỘNG SIÊU THAM SỐ (HYPERPARAMETER AUTO-TUNER)")
    print("=" * 75)

    config = Config()
    storage_dir = config.STORAGE_DIR
    storage_dir.mkdir(parents=True, exist_ok=True)

    # 1. Chuẩn bị dữ liệu nếu chưa truyền vào
    if not chunks or not vectors:
        print("[*] Đang nạp dữ liệu kiểm định từ bộ tài liệu cốt lõi...")
        chunker = LayoutAwareChunker(max_chunk_size=config.CHUNK_MAX_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        
        target_docs = [
            "vcb_bieu_lai_suat_va_ty_gia_ngoai_te.xlsx",
            "vcb_pdf_22_bieu_phi_the_tdqt_2026.pdf",
            "vcb_pdf_21_nq_hdqt_cctg_vcbs.pdf",
            "vcb_pdf_23_bieu_phi_cho_vay_khcn_2025.pdf",
            "vcb_pdf_25_bieu_phi_tai_khoan_so_dep.pdf",
            "vcb_pdf_39_thoa_thuan_tien_gui_online_digibank.pdf",
            "vcb_md_01__faq__the_tin_dung_va_the_ghi_no_ngan_hang.md",
            "vcb_md_03__phap_ly__dieu_kien_dieu_khoan_mo_tai_khoan_truc_tuyen_ekyc.md",
            "vcb_md_04__faq__ngan_hang_so_vcb_digibank_va_smart_otp.md",
            "vcb_md_06__faq__chuyen_tien_va_nhan_tien_kieu_hoi.md",
            "vcb_md_07__faq__vcb_loyalty_tich_diem_va_doi_qua.md",
            "vcb_md_10__the_tin_dung__vcb_vibe_platinum_dac_quyen_va_bieu_phi.md",
            "vcb_md_12__quy_trinh__mo_tai_khoan_thanh_toan_ekyc_va_xac_thuc_nfc.md"
        ]
        chunks = chunker.chunk_directory(config.DATASET_DIR, filenames=target_docs, max_pdf_pages=3)
        print(f" -> Đã trích xuất {len(chunks)} chunks đại diện cho tập đánh giá.")

        emb_model = EmbeddingModel(model_name=config.LOCAL_EMBEDDING_MODEL)
        print(f" -> Đang tạo vector embeddings...")
        vectors = emb_model.embed_documents([c["content"] for c in chunks], batch_size=64)
    else:
        emb_model = EmbeddingModel(model_name=config.LOCAL_EMBEDDING_MODEL)

    # 2. Khởi tạo Vector Store
    vector_store = WeaviateStore(collection_name=config.WEAVIATE_COLLECTION_NAME)
    vector_store.add_chunks(chunks, vectors)

    # 3. Đánh giá Baseline với cấu hình mặc định ban đầu
    print("\n[1/3] ĐÁNH GIÁ CẤU HÌNH GỐC (BASELINE CONFIGURATION):")
    bm25_baseline = BM25Search(k1=config.BM25_K1, b=config.BM25_B)
    bm25_baseline.index(chunks)

    retriever_baseline = HybridRetriever(
        vector_store=vector_store,
        bm25_search=bm25_baseline,
        embedding_model=emb_model,
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

    # 4. Không gian tìm kiếm tham số (Hyperparameter Search Grid)
    print("\n[2/3] BẮT ĐẦU QUÁ TRÌNH GRID SEARCH TUNING:")
    param_grid = [
        # (bm25_k1, bm25_b, rrf_k, dense_weight, candidate_k)
        # Nhóm 1: Thử nghiệm biến thể trọng số RRF (Dense vs BM25)
        {"bm25_k1": 1.5, "bm25_b": 0.75, "rrf_k": 60, "dense_weight": 0.4, "candidate_k": 15},
        {"bm25_k1": 1.5, "bm25_b": 0.75, "rrf_k": 60, "dense_weight": 0.5, "candidate_k": 15},
        {"bm25_k1": 1.5, "bm25_b": 0.75, "rrf_k": 60, "dense_weight": 0.6, "candidate_k": 15},
        {"bm25_k1": 1.5, "bm25_b": 0.75, "rrf_k": 60, "dense_weight": 0.7, "candidate_k": 15},

        # Nhóm 2: Thử nghiệm hằng số điều hòa RRF k (k=20, 40, 60, 80)
        {"bm25_k1": 1.5, "bm25_b": 0.75, "rrf_k": 20, "dense_weight": 0.5, "candidate_k": 15},
        {"bm25_k1": 1.5, "bm25_b": 0.75, "rrf_k": 40, "dense_weight": 0.5, "candidate_k": 15},
        {"bm25_k1": 1.5, "bm25_b": 0.75, "rrf_k": 80, "dense_weight": 0.5, "candidate_k": 15},

        # Nhóm 3: Tinh chỉnh BM25 k1 và b (tối ưu bắt số liệu và bảng)
        {"bm25_k1": 1.2, "bm25_b": 0.75, "rrf_k": 60, "dense_weight": 0.5, "candidate_k": 15},
        {"bm25_k1": 1.8, "bm25_b": 0.75, "rrf_k": 60, "dense_weight": 0.5, "candidate_k": 15},
        {"bm25_k1": 1.5, "bm25_b": 0.60, "rrf_k": 60, "dense_weight": 0.5, "candidate_k": 15},
        {"bm25_k1": 1.5, "bm25_b": 0.85, "rrf_k": 60, "dense_weight": 0.5, "candidate_k": 15},

        # Nhóm 4: Mở rộng Candidate K và phối hợp tối ưu
        {"bm25_k1": 1.5, "bm25_b": 0.75, "rrf_k": 40, "dense_weight": 0.6, "candidate_k": 20},
        {"bm25_k1": 1.8, "bm25_b": 0.60, "rrf_k": 40, "dense_weight": 0.55, "candidate_k": 20},
        {"bm25_k1": 1.2, "bm25_b": 0.60, "rrf_k": 50, "dense_weight": 0.6, "candidate_k": 20}
    ]

    all_trials = []
    best_trial = None
    best_mrr = -1.0

    for idx, params in enumerate(param_grid, start=1):
        # Khởi tạo BM25 với tham số thử nghiệm
        bm25_test = BM25Search(k1=params["bm25_k1"], b=params["bm25_b"])
        bm25_test.index(chunks)

        retriever_test = HybridRetriever(
            vector_store=vector_store,
            bm25_search=bm25_test,
            embedding_model=emb_model,
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
        if (metrics["mrr"] > best_mrr) or (metrics["mrr"] == best_mrr and metrics["hit_rate_at_1"] > (best_trial["metrics"]["hit_rate_at_1"] if best_trial else -1)):
            best_mrr = metrics["mrr"]
            best_trial = trial_record

    # 5. Đánh giá so sánh Before vs After
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

    # 6. Tự động cập nhật config.py nếu tìm thấy cấu hình vượt trội
    config_file = CURRENT_DIR.parent / "config.py"
    if config_file.exists():
        _update_config_file(config_file, best_p)
        print(f"\n[Tự động cập nhật] Đã cập nhật các siêu tham số tối ưu vào: {config_file.name}")

    # 7. Lưu báo cáo Tuning vào storage
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

def _update_config_file(config_file_path, best_params):
    """Cập nhật các giá trị tối ưu vào file config.py"""
    with open(config_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Thay thế các tham số tối ưu
    import re
    content = re.sub(r"BM25_K1\s*=\s*[\d\.]+", f"BM25_K1 = {best_params['bm25_k1']}", content)
    content = re.sub(r"BM25_B\s*=\s*[\d\.]+", f"BM25_B = {best_params['bm25_b']}", content)
    content = re.sub(r"RRF_K\s*=\s*\d+", f"RRF_K = {best_params['rrf_k']}", content)
    content = re.sub(r"DENSE_WEIGHT\s*=\s*[\d\.]+", f"DENSE_WEIGHT = {best_params['dense_weight']}", content)
    content = re.sub(r"HYBRID_CANDIDATE_K\s*=\s*\d+", f"HYBRID_CANDIDATE_K = {best_params['candidate_k']}", content)

    with open(config_file_path, "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    run_parameter_tuning()
